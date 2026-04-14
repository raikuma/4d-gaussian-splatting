#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import csv
import time
import torch
from torch import nn
from utils.loss_utils import l1_loss, ssim, msssim
from gaussian_renderer import get_rasterizer_backward_profile_dict, render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, knn
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap
from utils.focusgs_utils import (
    build_error_threshold_active_tile_metadata,
    build_random_active_tile_metadata,
    pixel_mask_from_tile_metadata,
    masked_l1_loss,
    masked_psnr,
    masked_ssim_loss,
)
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import make_grid, save_image
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def measure_cuda_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = fn()
    end.record()
    end.synchronize()
    return output, float(start.elapsed_time(end))


def maybe_measure_cuda_ms(enabled, fn):
    if enabled:
        return measure_cuda_ms(fn)
    return fn(), 0.0


def measure_wall_ms(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = fn()
    torch.cuda.synchronize()
    return output, (time.perf_counter() - start) * 1000.0


def maybe_measure_wall_ms(enabled, fn):
    if enabled:
        return measure_wall_ms(fn)
    return fn(), 0.0


def average_profile_dicts(profile_dicts, keys):
    averaged = {}
    for key in keys:
        values = [profile.get(key, 0.0) for profile in profile_dicts if profile]
        averaged[key] = float(sum(values) / len(values)) if values else 0.0
    return averaged

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint, debug_from,
             gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, batch_size, num_workers, ply_path,
             enable_focusgs, focusgs_tile_size, focusgs_tile_ratio, focusgs_sampling_mode, focusgs_error_threshold, focusgs_warmup_iters,
             enable_profiling, profile_rasterizer):
    
    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    scene = Scene(dataset, gaussians, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration, ply_path=ply_path)
    gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    best_psnr = 0.0
    ema_loss_for_log = 0.0
    ema_l1loss_for_log = 0.0
    ema_ssimloss_for_log = 0.0
    ema_tile_ratio_for_log = 1.0
    lambda_all = [key for key in opt.__dict__.keys() if key.startswith('lambda') and key!='lambda_dssim']
    for lambda_name in lambda_all:
        vars()[f"ema_{lambda_name.replace('lambda_','')}_for_log"] = 0.0
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
        
    if pipe.env_map_res:
        env_map = nn.Parameter(torch.zeros((3,pipe.env_map_res, pipe.env_map_res),dtype=torch.float, device="cuda").requires_grad_(True))
        env_map_optimizer = torch.optim.Adam([env_map], lr=opt.feature_lr, eps=1e-15)
    else:
        env_map = None
        
    gaussians.env_map = env_map
        
    training_dataset = scene.getTrainCameras()
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers if dataset.dataloader else 0, collate_fn=lambda x: x, drop_last=True)
    focusgs_sampling_mode = focusgs_sampling_mode.lower()
    focusgs_enabled = enable_focusgs and (
        (focusgs_sampling_mode == "random" and focusgs_tile_ratio < 1.0)
        or focusgs_sampling_mode == "error_threshold"
    )
    profiling_enabled = enable_profiling or profile_rasterizer
    profile_rasterizer = profiling_enabled and profile_rasterizer
    timing_log_path = os.path.join(dataset.model_path, "timing_log.csv")
    timing_log_fields = [
        "iteration",
        "focusgs_enabled",
        "focusgs_sampling_mode",
        "avg_tile_ratio",
        "iter_time_ms",
        "iter_wall_ms",
        "data_prep_wall_ms",
        "mask_update_ms",
        "mask_build_ms",
        "full_render_ms",
        "loss_wall_ms",
        "train_render_ms",
        "backward_ms",
        "densify_wall_ms",
        "optimizer_ms",
        "gpu_mem_alloc_mb",
        "gpu_mem_reserved_mb",
        "num_points",
        "psnr",
        "l1",
        "ssim_loss",
    ]
    raster_forward_keys = [
        "preprocess_ms",
        "active_filter_ms",
        "scan_ms",
        "duplicate_ms",
        "sort_ms",
        "ranges_ms",
        "render_ms",
        "total_ms",
    ]
    raster_backward_keys = [
        "render_ms",
        "preprocess_ms",
        "total_ms",
    ]
    forward_profile_fields = [f"raster_forward_{key}" for key in raster_forward_keys]
    backward_profile_fields = [f"raster_backward_{key}" for key in raster_backward_keys]
    if profiling_enabled:
        timing_log_fields.extend(forward_profile_fields)
        timing_log_fields.extend([f"full_render_{key}" for key in forward_profile_fields])
        timing_log_fields.extend(backward_profile_fields)
        timing_log_file = open(timing_log_path, "w", newline="")
        timing_writer = csv.DictWriter(timing_log_file, fieldnames=timing_log_fields)
        timing_writer.writeheader()
    else:
        timing_log_file = None
        timing_writer = None
     
    iteration = first_iter
    while iteration < opt.iterations + 1:
        for batch_data in training_dataloader:
            iteration += 1
            if iteration > opt.iterations:
                break

            iter_wall_start = time.perf_counter()
            iter_start.record()
            gaussians.update_learning_rate(iteration)
            
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % opt.sh_increase_interval == 0:
                gaussians.oneupSHdegree()
                
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            
            batch_point_grad = []
            batch_visibility_filter = []
            batch_radii = []
            batch_tile_ratios = []
            batch_mask_update_ms = []
            batch_mask_build_ms = []
            batch_full_render_ms = []
            batch_train_render_ms = []
            batch_loss_wall_ms = []
            batch_backward_ms = []
            batch_data_prep_wall_ms = []
            batch_raster_forward_profiles = []
            batch_full_render_raster_profiles = []
            batch_raster_backward_profiles = []
            
            for batch_idx in range(batch_size):
                (gt_image, viewpoint_cam), data_prep_wall_ms = maybe_measure_wall_ms(
                    profiling_enabled,
                    lambda: (
                        batch_data[batch_idx][0].cuda(),
                        batch_data[batch_idx][1].cuda(),
                    ),
                )
                active_tiles = None
                pixel_mask = None
                tile_ratio_for_iter = 1.0
                mask_update_ms = 0.0
                mask_build_ms = 0.0
                full_render_ms = 0.0
                loss_wall_ms = 0.0
                raster_forward_profile = {}
                full_render_raster_profile = {}
                raster_backward_profile = {}
                if focusgs_enabled and iteration > focusgs_warmup_iters:
                    if focusgs_sampling_mode == "random":
                        active_tiles, mask_update_ms = maybe_measure_cuda_ms(
                            profiling_enabled,
                            lambda: build_random_active_tile_metadata(
                                viewpoint_cam.image_height,
                                viewpoint_cam.image_width,
                                focusgs_tile_size,
                                focusgs_tile_ratio,
                                gt_image.device,
                            )
                        )
                    elif focusgs_sampling_mode == "error_threshold":
                        with torch.no_grad():
                            full_render_pkg, full_render_ms = maybe_measure_wall_ms(
                                profiling_enabled,
                                lambda: render(
                                    viewpoint_cam,
                                    gaussians,
                                    pipe,
                                    background,
                                    profile_rasterizer=profile_rasterizer,
                                )
                            )
                        full_render_raster_profile = full_render_pkg.get("rasterizer_forward_profile") or {}
                        error_map = torch.abs(full_render_pkg["render"].detach() - gt_image).mean(dim=0)
                        active_tiles, mask_update_ms = maybe_measure_cuda_ms(
                            profiling_enabled,
                            lambda: build_error_threshold_active_tile_metadata(
                                error_map,
                                viewpoint_cam.image_height,
                                viewpoint_cam.image_width,
                                focusgs_tile_size,
                                focusgs_error_threshold,
                            )
                        )
                    else:
                        raise ValueError(f"Unsupported FocusGS sampling mode: {focusgs_sampling_mode}")

                    pixel_mask, mask_build_ms = maybe_measure_cuda_ms(
                        profiling_enabled,
                        lambda: pixel_mask_from_tile_metadata(
                            active_tiles,
                            viewpoint_cam.image_height,
                            viewpoint_cam.image_width,
                            focusgs_tile_size,
                            gt_image.device,
                        )
                    )
                    tile_ratio_for_iter = active_tiles["tile_ratio"]

                render_pkg, train_render_ms = maybe_measure_wall_ms(
                    profiling_enabled,
                    lambda: render(
                        viewpoint_cam,
                        gaussians,
                        pipe,
                        background,
                        active_tiles=active_tiles,
                        profile_rasterizer=profile_rasterizer,
                    )
                )
                raster_forward_profile = render_pkg.get("rasterizer_forward_profile") or {}
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                depth = render_pkg["depth"]
                alpha = render_pkg["alpha"]
                opacity_t = render_pkg["opacity_t"]
                sigma = render_pkg["sigma"]

                # Loss
                def build_loss_terms():
                    if pixel_mask is None:
                        local_l1 = l1_loss(image, gt_image)
                        local_ssim = 1.0 - ssim(image, gt_image)
                    else:
                        local_l1 = masked_l1_loss(image, gt_image, pixel_mask)
                        local_ssim = masked_ssim_loss(image, gt_image, pixel_mask)
                    return local_l1, local_ssim

                (Ll1, Lssim), loss_wall_ms = maybe_measure_wall_ms(profiling_enabled, build_loss_terms)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
                
                ###### opa mask Loss ######
                if opt.lambda_opa_mask > 0:
                    o = alpha.clamp(1e-6, 1-1e-6)
                    sky = 1 - viewpoint_cam.gt_alpha_mask
                    if pixel_mask is None:
                        Lopa_mask = (- sky * torch.log(1 - o)).mean()
                    else:
                        weighted_sky = sky * pixel_mask.to(dtype=sky.dtype)
                        Lopa_mask = (- weighted_sky * torch.log(1 - o)).sum() / weighted_sky.sum().clamp_min(1.0)

                    # lambda_opa_mask = opt.lambda_opa_mask * (1 - 0.99 * min(1, iteration/opt.iterations))
                    lambda_opa_mask = opt.lambda_opa_mask
                    loss = loss + lambda_opa_mask * Lopa_mask
                ###### opa mask Loss ######
                
                ###### rigid loss ######
                if opt.lambda_rigid > 0:
                    k = 20
                    # cur_time = viewpoint_cam.timestamp
                    # _, delta_mean = gaussians.get_current_covariance_and_mean_offset(1.0, cur_time)
                    xyz_mean = gaussians.get_xyz
                    xyz_cur =  xyz_mean #  + delta_mean
                    idx, dist = knn(xyz_cur[None].contiguous().detach(), 
                                    xyz_cur[None].contiguous().detach(), 
                                    k)
                    _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                    weight = torch.exp(-100 * dist)
                    # cur_marginal_t = gaussians.get_marginal_t(cur_time).detach().squeeze(-1)
                    # marginal_weights = cur_marginal_t[idx] * cur_marginal_t[None,:,None]
                    # weight *= marginal_weights
                    
                    # mean_t, cov_t = gaussians.get_t, gaussians.get_cov_t(scaling_modifier=1)
                    # mean_t_nn, cov_t_nn = mean_t[idx], cov_t[idx]
                    # weight *= torch.exp(-0.5*(mean_t[None, :, None]-mean_t_nn)**2/cov_t[None, :, None]/cov_t_nn*(cov_t[None, :, None]+cov_t_nn)).squeeze(-1).detach()
                    vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
                    Lrigid = (weight * vel_dist).sum() / k / xyz_cur.shape[0]
                    loss = loss + opt.lambda_rigid * Lrigid
                ########################
                
                ###### motion loss ######
                if opt.lambda_motion > 0:
                    _, velocity = gaussians.get_current_covariance_and_mean_offset(1.0, gaussians.get_t + 0.1)
                    Lmotion = velocity.norm(p=2, dim=1).mean()
                    loss = loss + opt.lambda_motion * Lmotion
                ########################

                loss = loss / batch_size
                _, backward_ms = maybe_measure_wall_ms(profiling_enabled, loss.backward)
                if profile_rasterizer:
                    raster_backward_profile = get_rasterizer_backward_profile_dict() or {}
                batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                batch_radii.append(radii)
                batch_visibility_filter.append(visibility_filter)
                batch_tile_ratios.append(tile_ratio_for_iter)
                batch_data_prep_wall_ms.append(data_prep_wall_ms)
                batch_mask_update_ms.append(mask_update_ms)
                batch_mask_build_ms.append(mask_build_ms)
                batch_full_render_ms.append(full_render_ms)
                batch_train_render_ms.append(train_render_ms)
                batch_loss_wall_ms.append(loss_wall_ms)
                batch_backward_ms.append(backward_ms)
                batch_raster_forward_profiles.append(raster_forward_profile)
                batch_full_render_raster_profiles.append(full_render_raster_profile)
                batch_raster_backward_profiles.append(raster_backward_profile)

            if batch_size > 1:
                visibility_count = torch.stack(batch_visibility_filter,1).sum(1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii,1).max(1)[0]
                
                batch_viewspace_point_grad = torch.stack(batch_point_grad,1).sum(1)
                batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)
                
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone()[:,0].detach()
                    batch_t_grad[visibility_filter] = batch_t_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                    batch_t_grad = batch_t_grad.unsqueeze(1)
            else:
                if gaussians.gaussian_dim == 4:
                    batch_t_grad = gaussians._t.grad.clone().detach()

            avg_mask_update_ms = float(sum(batch_mask_update_ms) / max(1, len(batch_mask_update_ms)))
            avg_mask_build_ms = float(sum(batch_mask_build_ms) / max(1, len(batch_mask_build_ms)))
            avg_full_render_ms = float(sum(batch_full_render_ms) / max(1, len(batch_full_render_ms)))
            avg_train_render_ms = float(sum(batch_train_render_ms) / max(1, len(batch_train_render_ms)))
            avg_loss_wall_ms = float(sum(batch_loss_wall_ms) / max(1, len(batch_loss_wall_ms)))
            avg_backward_ms = float(sum(batch_backward_ms) / max(1, len(batch_backward_ms)))
            avg_data_prep_wall_ms = float(sum(batch_data_prep_wall_ms) / max(1, len(batch_data_prep_wall_ms)))
            
            iter_end.record()
            loss_dict = {"Ll1": Ll1,
                        "Lssim": Lssim}
            iter_wall_ms = (time.perf_counter() - iter_wall_start) * 1000.0
            avg_raster_forward_profile = average_profile_dicts(batch_raster_forward_profiles, raster_forward_keys)
            avg_full_render_raster_profile = average_profile_dicts(batch_full_render_raster_profiles, raster_forward_keys)
            avg_raster_backward_profile = average_profile_dicts(batch_raster_backward_profiles, raster_backward_keys)

            with torch.no_grad():
                if pixel_mask is None:
                    psnr_for_log = psnr(image, gt_image).mean().double()
                else:
                    psnr_for_log = masked_psnr(image, gt_image, pixel_mask).double()
                avg_tile_ratio = float(sum(batch_tile_ratios) / max(1, len(batch_tile_ratios)))
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_l1loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_l1loss_for_log
                ema_ssimloss_for_log = 0.4 * Lssim.item() + 0.6 * ema_ssimloss_for_log
                ema_tile_ratio_for_log = 0.4 * avg_tile_ratio + 0.6 * ema_tile_ratio_for_log
                iter_time_ms = float(iter_start.elapsed_time(iter_end))
                
                for lambda_name in lambda_all:
                    if opt.__dict__[lambda_name] > 0:
                        ema = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                        vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"] = 0.4 * vars()[f"L{lambda_name.replace('lambda_', '')}"].item() + 0.6*ema
                        loss_dict[lambda_name.replace("lambda_", "L")] = vars()[lambda_name.replace("lambda_", "L")]
                        
                if iteration % 10 == 0:
                    postfix = {"Loss": f"{ema_loss_for_log:.{7}f}",
                                            "PSNR": f"{psnr_for_log:.{2}f}",
                                            "Ll1": f"{ema_l1loss_for_log:.{4}f}",
                                            "Lssim": f"{ema_ssimloss_for_log:.{4}f}",
                                            "#": f"{gaussians.get_xyz.shape[0]}",}
                    if focusgs_enabled:
                        postfix["tile"] = f"{ema_tile_ratio_for_log:.{3}f}"
                        postfix["mode"] = focusgs_sampling_mode
                    if profiling_enabled:
                        postfix["mask"] = f"{avg_mask_update_ms + avg_mask_build_ms:.1f}ms"
                        postfix["render"] = f"{avg_train_render_ms:.1f}ms"
                        postfix["bw"] = f"{avg_backward_ms:.1f}ms"
                    
                    for lambda_name in lambda_all:
                        if opt.__dict__[lambda_name] > 0:
                            ema_loss = vars()[f"ema_{lambda_name.replace('lambda_', '')}_for_log"]
                            postfix[lambda_name.replace("lambda_", "L")] = f"{ema_loss:.{4}f}"
                            
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                if tb_writer and focusgs_enabled:
                    tb_writer.add_scalar('focusgs/tile_ratio', avg_tile_ratio, iteration)
                if tb_writer and profiling_enabled:
                    tb_writer.add_scalar('timing/iter_ms', iter_time_ms, iteration)
                    tb_writer.add_scalar('timing/iter_wall_ms', iter_wall_ms, iteration)
                    tb_writer.add_scalar('timing/data_prep_wall_ms', avg_data_prep_wall_ms, iteration)
                    tb_writer.add_scalar('timing/train_render_ms', avg_train_render_ms, iteration)
                    tb_writer.add_scalar('timing/loss_wall_ms', avg_loss_wall_ms, iteration)
                    tb_writer.add_scalar('timing/backward_ms', avg_backward_ms, iteration)
                    tb_writer.add_scalar('timing/gpu_mem_alloc_mb', torch.cuda.memory_allocated() / (1024 ** 2), iteration)
                    tb_writer.add_scalar('timing/gpu_mem_reserved_mb', torch.cuda.memory_reserved() / (1024 ** 2), iteration)
                    if focusgs_enabled:
                        tb_writer.add_scalar('focusgs/mask_update_ms', avg_mask_update_ms, iteration)
                        tb_writer.add_scalar('focusgs/mask_build_ms', avg_mask_build_ms, iteration)
                        tb_writer.add_scalar('focusgs/full_render_ms', avg_full_render_ms, iteration)

                # Log and save
                optimizer_ms = 0.0
                test_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time_ms, testing_iterations, scene, render, (pipe, background), loss_dict)
                if (iteration in testing_iterations):
                    if test_psnr >= best_psnr:
                        best_psnr = test_psnr
                        print("\n[ITER {}] Saving best checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_best.pth")
                        
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                densify_wall_ms = 0.0
                if iteration < opt.densify_until_iter and (opt.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < opt.densify_until_num_points):
                    def run_densify():
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        if batch_size == 1:
                            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None)
                        else:
                            gaussians.add_densification_stats_grad(batch_viewspace_point_grad, visibility_filter, batch_t_grad if gaussians.gaussian_dim == 4 else None,
                                                                   da_densification=pipe.da_densification, opacity_t=opacity_t, sigma=sigma)

                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            gaussians.densify_and_prune(opt.densify_grad_threshold, opt.thresh_opa_prune, scene.cameras_extent, size_threshold, opt.densify_grad_t_threshold)

                        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()
                    _, densify_wall_ms = maybe_measure_wall_ms(profiling_enabled, run_densify)
                        
                # Optimizer step
                if iteration < opt.iterations:
                    _, optimizer_ms = maybe_measure_wall_ms(profiling_enabled, gaussians.optimizer.step)
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    if pipe.env_map_res and iteration < pipe.env_optimize_until:
                        _, env_optimizer_ms = maybe_measure_wall_ms(profiling_enabled, env_map_optimizer.step)
                        optimizer_ms += env_optimizer_ms
                        env_map_optimizer.zero_grad(set_to_none = True)

                if timing_writer is not None:
                    row = {
                        "iteration": iteration,
                        "focusgs_enabled": int(focusgs_enabled),
                        "focusgs_sampling_mode": focusgs_sampling_mode if focusgs_enabled else "disabled",
                        "avg_tile_ratio": avg_tile_ratio,
                        "iter_time_ms": iter_time_ms,
                        "iter_wall_ms": iter_wall_ms,
                        "data_prep_wall_ms": avg_data_prep_wall_ms,
                        "mask_update_ms": avg_mask_update_ms,
                        "mask_build_ms": avg_mask_build_ms,
                        "full_render_ms": avg_full_render_ms,
                        "loss_wall_ms": avg_loss_wall_ms,
                        "train_render_ms": avg_train_render_ms,
                        "backward_ms": avg_backward_ms,
                        "densify_wall_ms": densify_wall_ms,
                        "optimizer_ms": optimizer_ms,
                        "gpu_mem_alloc_mb": torch.cuda.memory_allocated() / (1024 ** 2),
                        "gpu_mem_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
                        "num_points": int(gaussians.get_xyz.shape[0]),
                        "psnr": float(psnr_for_log.item() if hasattr(psnr_for_log, "item") else psnr_for_log),
                        "l1": float(Ll1.item()),
                        "ssim_loss": float(Lssim.item()),
                    }
                    for key in raster_forward_keys:
                        row[f"raster_forward_{key}"] = avg_raster_forward_profile.get(key, 0.0)
                        row[f"full_render_raster_forward_{key}"] = avg_full_render_raster_profile.get(key, 0.0)
                    for key in raster_backward_keys:
                        row[f"raster_backward_{key}"] = avg_raster_backward_profile.get(key, 0.0)
                    timing_writer.writerow(row)
                    timing_log_file.flush()

    if timing_log_file is not None:
        timing_log_file.close()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_dict=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        if loss_dict is not None:
            if "Lrigid" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/rigid_loss', loss_dict['Lrigid'].item(), iteration)
            if "Ldepth" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/depth_loss', loss_dict['Ldepth'].item(), iteration)
            if "Ltv" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/tv_loss', loss_dict['Ltv'].item(), iteration)
            if "Lopa" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/opa_loss', loss_dict['Lopa'].item(), iteration)
            if "Lptsopa" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/pts_opa_loss', loss_dict['Lptsopa'].item(), iteration)
            if "Lsmooth" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/smooth_loss', loss_dict['Lsmooth'].item(), iteration)
            if "Llaplacian" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/laplacian_loss', loss_dict['Llaplacian'].item(), iteration)

    psnr_test_iter = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        save_folder = os.path.join(scene.model_path, f"test_{iteration}_renders")
        os.makedirs(save_folder, exist_ok=True)

        validation_configs = (
            # {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
            {'name': 'test', 'cameras' : [scene.getTestCameras()[idx] for idx in range(len(scene.getTestCameras()))]},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                msssim_test = 0.0
                for idx, batch_data in enumerate(tqdm(config['cameras'])):
                    gt_image, viewpoint = batch_data
                    gt_image = gt_image.cuda()
                    viewpoint = viewpoint.cuda()
                    
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    
                    depth = easy_cmap(render_pkg['depth'][0])
                    alpha = torch.clamp(render_pkg['alpha'], 0.0, 1.0).repeat(3,1,1)
                    if tb_writer and (idx < 5):
                        grid = [gt_image, image, alpha, depth]
                        grid = make_grid(grid, nrow=2)
                        tb_writer.add_images(config['name'] + "_view_{}/gt_vs_render".format(viewpoint.image_name), grid[None], global_step=iteration)
                            
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    msssim_test += msssim(image[None].cpu(), gt_image[None].cpu())

                    # save_image(image, os.path.join(save_folder, f'{idx:04d}.png'))
                    save_image(image, os.path.join(save_folder, viewpoint.image_name.split('/')[-1]+'.png'))

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 
                ssim_test /= len(config['cameras'])     
                msssim_test /= len(config['cameras'])        
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - msssim', msssim_test, iteration)
                if config['name'] == 'test':
                    psnr_test_iter = psnr_test.item()
                    
    torch.cuda.empty_cache()
    return psnr_test_iter

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--ply_path", type=str, default=None, help="Path to a PLY file to use instead of the one in the scene folder")
    parser.add_argument("--enable_focusgs", action="store_true")
    parser.add_argument("--focusgs_tile_size", type=int, default=16)
    parser.add_argument("--focusgs_tile_ratio", type=float, default=1.0)
    parser.add_argument("--focusgs_sampling_mode", type=str, default="random")
    parser.add_argument("--focusgs_error_threshold", type=float, default=0.05)
    parser.add_argument("--focusgs_warmup_iters", type=int, default=100)
    parser.add_argument("--enable_profiling", action="store_true")
    parser.add_argument("--profile_rasterizer", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
        
    cfg = OmegaConf.load(args.config)
    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)

    args.save_iterations.append(args.iterations)
        
    if args.exhaust_test:
        args.test_iterations = args.test_iterations + [i for i in range(0,op.iterations,500)]
    
    setup_seed(args.seed)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from,
             args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d, args.batch_size, args.num_workers, args.ply_path,
             args.enable_focusgs, args.focusgs_tile_size, args.focusgs_tile_ratio, args.focusgs_sampling_mode, args.focusgs_error_threshold, args.focusgs_warmup_iters,
             args.enable_profiling, args.profile_rasterizer)

    # All done
    print("\nTraining complete.")
