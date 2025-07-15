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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
# from utils.loss_utils import ssim
# from lpipsPyTorch import lpips
import lpips
import json
from tqdm import tqdm
# from utils.image_utils import psnr
from argparse import ArgumentParser

import models
from skimage.metrics import structural_similarity
import numpy as np
import cv2
import math

def psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)

    num_valid = np.sum(mask) + 1e-8

    mse = np.sum((img1 - img2)**2 * mask) / num_valid
    
    if mse == 0:
        return 0 #float('inf')

    return 10 * math.log10(1./mse)

def ssim(img1, img2, mask):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 1]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    # _, ssim_map = structural_similarity(img1, img2, multichannel=True, full=True, win_size=11)
    _, ssim_map = structural_similarity(img1, img2, win_size=11, data_range=1, channel_axis=2, full=True)
    num_valid = np.sum(mask) + 1e-8

    return np.sum(ssim_map * mask) / num_valid

def open_img(path):
    img = cv2.imread(path)[:, :, ::-1]
    img = np.float32(img) / 255
    return img

def open_mask(path, resize=None):
    mask = np.float32(cv2.imread(path) > 1e-3)
    # mask = np.float32(mask) / 255

    # dilation
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    if resize is not None:
        mask = cv2.resize(mask, (resize[1], resize[0]), interpolation=cv2.INTER_NEAREST)

    return mask

def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def readImages(renders_dir, gt_dir, mask_dir):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(renders_dir):
        renders.append(renders_dir / fname)
        idx = int(fname.split(".")[0])
        fname = f"{idx:05d}.png"
        gts.append(gt_dir / fname)
        masks.append(mask_dir / fname)
        # render = Image.open(renders_dir / fname)
        # gt = Image.open(gt_dir / fname)
        # renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        # gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, masks, image_names

def evaluate(model_paths, gt_path, mask_path):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        # try:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"
        mask_dir = Path(mask_path)
        gt_dir = Path(gt_path)

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            # gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            # mask_dir = method_dir / "mask"
            renders, gts, masks, image_names = readImages(renders_dir, gt_dir, mask_dir)

            ssims = []
            psnrs = []
            lpipss = []
            alexs = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                if idx >= 300:
                    break
                # render = tf.to_tensor(Image.open(renders[idx])).unsqueeze(0)[:, :3, :, :].cuda()
                # gt = tf.to_tensor(Image.open(gts[idx])).unsqueeze(0)[:, :3, :, :].cuda()
                # mask = tf.to_tensor(Image.open(masks[idx])).unsqueeze(0).cuda()
                render = open_img(renders[idx])
                gt = open_img(gts[idx])
                mask = open_mask(masks[idx], resize=(render.shape[0], render.shape[1]))
                ssims.append(ssim(render, gt, mask))
                psnrs.append(psnr(render, gt, mask))
                # lpipss.append(lpips_fn(render, gt).detach())
                lpipss.append(0.0)
                # alexs.append(alex_fn(render, gt).detach())
                alexs.append(alex_fn(im2tensor(render), im2tensor(gt), torch.Tensor(mask[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))).item())
                # ssims.append(ssim(renders[idx], gts[idx]))
                # psnrs.append(psnr(renders[idx], gts[idx]))
                # lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("  ALEX : {:>12.7f}".format(torch.tensor(alexs).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item(),
                                                    "ALEX": torch.tensor(alexs).mean().item()})
            per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                        "ALEX": {name: alex for alex, name in zip(torch.tensor(alexs).tolist(), image_names)}})

        with open(scene_dir + "/results_mask.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        print(full_dict[scene_dir])
        # with open(scene_dir + "/per_view.json", 'w') as fp:
        #     json.dump(per_view_dict[scene_dir], fp, indent=True)
        # except Exception as e:
        #     print("Unable to compute metrics for model", scene_dir)
        #     print(e)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    # alex_fn = lpips.LPIPS(net='alex').to(device)
    alex_fn = models.PerceptualLoss(model='net-lin',net='alex',
                                      use_gpu=True,version=0.1)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--mask_path', type=str, default=None)
    parser.add_argument('--gt_path', type=str, default=None)
    args = parser.parse_args()
    evaluate(args.model_paths, args.gt_path, args.mask_path)
