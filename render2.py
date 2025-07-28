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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams,OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
# import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def pick(img, x, y):
    for i in range(-1, 2):
        for j in range(-1, 2):
            img[0, y+i, x+j] = 1
            img[1, y+i, x+j] = 0
            img[2, y+i, x+j] = 0


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    t_list = []
    d_list = []
    p = (790, 895)
    # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    for idx, view in enumerate(views):
        if idx >= 100:
            continue
        # if idx < 100 or idx >= 200:
            continue
        # print(idx)
        torch.cuda.synchronize(); t0 = time.time()
        results = render(view[1].cuda(), gaussians, pipeline, background, scaling_modifier=1.0)
        rendering = torch.clamp(results["render"],0.0,1.0)
        torch.cuda.synchronize(); t1 = time.time()
        depth = results["depth"]
        d_list.append(depth[0,p[1],p[0]].item()) # sear steak
        # d_list.append(depth[0,780,648].item()) # cook_spinach
        pick(rendering, p[0], p[1])
        t_list.append(t1-t0)
        # gt = view[0][0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
    # exit(0)

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    plt.plot(d_list)
    plt.savefig("depth.png")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=4, rot_4d=True)
        scene = Scene(dataset, gaussians, shuffle=False)

        # print(gaussians._xyz.shape)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--config", type=str)
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")

    args = get_combined_args(parser)
    # from omegaconf import OmegaConf
    # from omegaconf.dictconfig import DictConfig
    # cfg = OmegaConf.load(args.config)
    # def recursive_merge(key, host):
    #     if isinstance(host[key], DictConfig):
    #         for key1 in host[key].keys():
    #             recursive_merge(key1, host[key])
    #     else:
    #         assert hasattr(args, key), key
    #         setattr(args, key, host[key])
    # for k in cfg.keys():
    #     recursive_merge(k, cfg)
    print("Rendering " + args.model_path)

    args.source_path = args.source_path.replace("/root/home/bbangsik/workdirs/4d-gaussian-splatting/data/N3V", "/data/N3DV")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)