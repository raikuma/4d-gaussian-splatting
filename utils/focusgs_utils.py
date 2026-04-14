import math

import torch
import torch.nn.functional as F

from utils.loss_utils import ssim


def _tile_grid_shape(image_height, image_width, tile_size):
    tile_h = (int(image_height) + tile_size - 1) // tile_size
    tile_w = (int(image_width) + tile_size - 1) // tile_size
    return tile_h, tile_w


def _metadata_from_flat_mask(active_mask_flat, tile_h, tile_w):
    num_tiles = tile_h * tile_w
    active_tile_ids = active_mask_flat.nonzero(as_tuple=False).flatten().to(dtype=torch.int32)
    active_tile_xy = torch.stack(
        (
            active_tile_ids.remainder(tile_w),
            torch.div(active_tile_ids, tile_w, rounding_mode="floor"),
        ),
        dim=1,
    ).to(dtype=torch.int32)

    dense_rank_map = torch.full((num_tiles,), -1, dtype=torch.int32, device=active_mask_flat.device)
    if active_tile_ids.numel() > 0:
        dense_rank_map[active_tile_ids.long()] = torch.arange(
            active_tile_ids.shape[0],
            dtype=torch.int32,
            device=active_mask_flat.device,
        )

    return {
        "ids": active_tile_ids,
        "xy": active_tile_xy,
        "mask_dense": active_mask_flat,
        "rank_map": dense_rank_map,
        "tile_shape": (tile_h, tile_w),
        "tile_ratio": active_tile_ids.shape[0] / float(num_tiles),
    }


def _tile_scores_from_error_map(error_map, image_height, image_width, tile_size):
    if error_map.dim() == 2:
        error_map = error_map.unsqueeze(0)
    if error_map.dim() != 3:
        raise ValueError("error_map must have shape [H, W] or [C, H, W]")

    error_map = error_map.mean(dim=0, keepdim=True).unsqueeze(0)
    pad_h = (tile_size - (image_height % tile_size)) % tile_size
    pad_w = (tile_size - (image_width % tile_size)) % tile_size
    padded_error = F.pad(error_map, (0, pad_w, 0, pad_h), mode="replicate")
    return F.avg_pool2d(padded_error, kernel_size=tile_size, stride=tile_size).squeeze(0).squeeze(0)


def build_random_active_tile_metadata(image_height, image_width, tile_size, tile_ratio, device):
    tile_h, tile_w = _tile_grid_shape(image_height, image_width, tile_size)
    num_tiles = tile_h * tile_w
    active_tiles = max(1, min(num_tiles, int(math.ceil(tile_ratio * num_tiles))))

    active_mask_flat = torch.zeros((num_tiles,), dtype=torch.bool, device=device)
    if active_tiles >= num_tiles:
        active_mask_flat.fill_(True)
    else:
        sampled_ids = torch.randperm(num_tiles, device=device)[:active_tiles]
        active_mask_flat[sampled_ids] = True

    return _metadata_from_flat_mask(active_mask_flat, tile_h, tile_w)


def build_error_threshold_active_tile_metadata(error_map, image_height, image_width, tile_size, error_threshold):
    tile_scores = _tile_scores_from_error_map(error_map, image_height, image_width, tile_size)
    tile_h, tile_w = tile_scores.shape
    flat_scores = tile_scores.reshape(-1)
    active_mask_flat = flat_scores > error_threshold

    if not bool(active_mask_flat.any()):
        active_mask_flat[torch.argmax(flat_scores)] = True

    return _metadata_from_flat_mask(active_mask_flat, tile_h, tile_w)


def pixel_mask_from_tile_metadata(active_tiles, image_height, image_width, tile_size, device):
    tile_h, tile_w = active_tiles["tile_shape"]
    pixel_mask = active_tiles["mask_dense"].to(device=device, dtype=torch.bool).view(tile_h, tile_w)
    pixel_mask = pixel_mask.repeat_interleave(tile_size, dim=0).repeat_interleave(tile_size, dim=1)
    return pixel_mask[:image_height, :image_width].unsqueeze(0)


def masked_l1_loss(image, gt_image, pixel_mask):
    expanded_mask = pixel_mask.to(dtype=image.dtype).expand_as(image)
    denom = expanded_mask.sum().clamp_min(1.0)
    return torch.abs(image - gt_image).mul(expanded_mask).sum() / denom


def masked_ssim_loss(image, gt_image, pixel_mask):
    mask = pixel_mask.to(dtype=image.dtype)
    return 1.0 - ssim(image * mask, gt_image * mask)


def masked_psnr(image, gt_image, pixel_mask):
    expanded_mask = pixel_mask.to(dtype=image.dtype).expand_as(image)
    denom = expanded_mask.sum().clamp_min(1.0)
    mse = ((image - gt_image) ** 2).mul(expanded_mask).sum() / denom
    return 20 * torch.log10(1.0 / torch.sqrt(mse.clamp_min(1e-8)))
