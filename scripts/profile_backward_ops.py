import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state, knn
from utils.loss_utils import l1_loss, ssim


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = OmegaConf.load(args.config)

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            if hasattr(args, key):
                setattr(args, key, host[key])

    for key in cfg.keys():
        recursive_merge(key, cfg)
    return args


def fetch_next_batch(data_iter, dataloader):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch = next(data_iter)
    return batch, data_iter


def build_loss(batch_data, gaussians, pipe, background, opt, batch_size):
    losses = []
    for batch_idx in range(batch_size):
        gt_image, viewpoint_cam = batch_data[batch_idx]
        gt_image = gt_image.cuda()
        viewpoint_cam = viewpoint_cam.cuda()

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        alpha = render_pkg["alpha"]
        opacity_t = render_pkg["opacity_t"]
        sigma = render_pkg["sigma"]

        ll1 = l1_loss(image, gt_image)
        lssim = 1.0 - ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * lssim

        if opt.lambda_opa_mask > 0:
            opacity = alpha.clamp(1e-6, 1 - 1e-6)
            sky = 1 - viewpoint_cam.gt_alpha_mask
            lopa_mask = (-sky * torch.log(1 - opacity)).mean()
            loss = loss + opt.lambda_opa_mask * lopa_mask

        if opt.lambda_rigid > 0:
            k = 20
            xyz_cur = gaussians.get_xyz
            idx, dist = knn(
                xyz_cur[None].contiguous().detach(),
                xyz_cur[None].contiguous().detach(),
                k,
            )
            _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
            weight = torch.exp(-100 * dist)
            vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
            lrigid = (weight * vel_dist).sum() / k / xyz_cur.shape[0]
            loss = loss + opt.lambda_rigid * lrigid

        if opt.lambda_motion > 0:
            _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
            lmotion = velocity.norm(p=2, dim=1).mean()
            loss = loss + opt.lambda_motion * lmotion

        losses.append(loss / batch_size)
    return sum(losses)


def main():
    parser = argparse.ArgumentParser(description="Profile backward operator breakdown from a saved checkpoint.")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--profile_iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--ply_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    args = apply_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_seed(args.seed)
    safe_state(True)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    pipe.profile_rasterizer = False
    pipe.current_iteration = -1

    gaussians = GaussianModel(
        dataset.sh_degree,
        gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration,
        rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d,
        sh_degree_t=2 if pipe.eval_shfs_4d else 0,
    )
    scene = Scene(
        dataset,
        gaussians,
        num_pts=args.num_pts,
        num_pts_ratio=args.num_pts_ratio,
        time_duration=args.time_duration,
        ply_path=args.ply_path,
    )
    gaussians.training_setup(opt)
    model_params, _ = torch.load(args.checkpoint)
    gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    training_dataset = scene.getTrainCameras()
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if dataset.dataloader else 0,
        collate_fn=lambda x: x,
        drop_last=True,
    )
    data_iter = iter(training_dataloader)

    for _ in range(args.warmup_iters):
        batch_data, data_iter = fetch_next_batch(data_iter, training_dataloader)
        gaussians.optimizer.zero_grad(set_to_none=True)
        loss = build_loss(batch_data, gaussians, pipe, background, opt, args.batch_size)
        loss.backward()
        gaussians.optimizer.zero_grad(set_to_none=True)

    backward_times_ms = []
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for step in range(args.profile_iters):
            pipe.current_iteration = step
            batch_data, data_iter = fetch_next_batch(data_iter, training_dataloader)
            gaussians.optimizer.zero_grad(set_to_none=True)
            loss = build_loss(batch_data, gaussians, pipe, background, opt, args.batch_size)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            loss.backward()
            end.record()
            torch.cuda.synchronize()
            backward_times_ms.append(start.elapsed_time(end))
            gaussians.optimizer.zero_grad(set_to_none=True)

    key_averages = prof.key_averages()
    operator_cuda_ms = {}
    for event in key_averages:
        operator_cuda_ms[event.key] = float(event.cuda_time_total) / 1000.0

    norm_backward_ms = operator_cuda_ms.get("LinalgVectorNormBackward0", 0.0)
    rasterizer_backward_ms = operator_cuda_ms.get("_RasterizeGaussiansBackward", 0.0)
    total_backward_ms = float(sum(backward_times_ms))
    avg_backward_ms = total_backward_ms / max(len(backward_times_ms), 1)
    avg_norm_backward_ms = norm_backward_ms / max(args.profile_iters, 1)
    avg_rasterizer_backward_ms = rasterizer_backward_ms / max(args.profile_iters, 1)

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "profile_iters": args.profile_iters,
        "warmup_iters": args.warmup_iters,
        "avg_backward_ms": avg_backward_ms,
        "avg_norm_backward_ms": avg_norm_backward_ms,
        "avg_rasterizer_backward_ms": avg_rasterizer_backward_ms,
        "norm_share_of_backward": avg_norm_backward_ms / avg_backward_ms if avg_backward_ms > 0 else 0.0,
        "rasterizer_share_of_backward": avg_rasterizer_backward_ms / avg_backward_ms if avg_backward_ms > 0 else 0.0,
        "norm_vs_rasterizer": avg_norm_backward_ms / avg_rasterizer_backward_ms if avg_rasterizer_backward_ms > 0 else 0.0,
        "top_cuda_ops_ms_total": sorted(operator_cuda_ms.items(), key=lambda item: item[1], reverse=True)[:20],
    }

    summary_path = output_dir / "backward_op_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table_path = output_dir / "backward_op_table.txt"
    table_path.write_text(key_averages.table(sort_by="cuda_time_total", row_limit=40), encoding="utf-8")

    text_lines = [
        f"checkpoint: {summary['checkpoint']}",
        f"profile_iters: {summary['profile_iters']}",
        f"warmup_iters: {summary['warmup_iters']}",
        f"avg_backward_ms: {summary['avg_backward_ms']:.4f}",
        f"avg_norm_backward_ms: {summary['avg_norm_backward_ms']:.4f}",
        f"avg_rasterizer_backward_ms: {summary['avg_rasterizer_backward_ms']:.4f}",
        f"norm_share_of_backward: {summary['norm_share_of_backward'] * 100:.2f}%",
        f"rasterizer_share_of_backward: {summary['rasterizer_share_of_backward'] * 100:.2f}%",
        f"norm_vs_rasterizer: {summary['norm_vs_rasterizer']:.4f}",
    ]
    text_path = output_dir / "backward_op_summary.txt"
    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print(text_path)


if __name__ == "__main__":
    main()
