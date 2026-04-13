import math

import torch


def get_tile_layout(image_height, image_width, tile_size):
    tiles_y = math.ceil(image_height / tile_size)
    tiles_x = math.ceil(image_width / tile_size)
    return tiles_y, tiles_x, tiles_y * tiles_x


def build_tile_selection(image_height, image_width, tile_size, tile_ratio, device, selection_mode="random"):
    tiles_y, tiles_x, total_tiles = get_tile_layout(image_height, image_width, tile_size)
    clamped_ratio = max(0.0, min(1.0, float(tile_ratio)))

    if selection_mode == "full" or clamped_ratio >= 1.0:
        active_tile_ids = torch.arange(total_tiles, device=device, dtype=torch.int32)
    elif selection_mode == "random":
        num_active_tiles = max(1, int(round(total_tiles * clamped_ratio)))
        active_tile_ids = torch.randperm(total_tiles, device=device)[:num_active_tiles].to(torch.int32)
        active_tile_ids, _ = torch.sort(active_tile_ids)
    else:
        raise ValueError(f"Unsupported tile selection mode: {selection_mode}")

    tile_mask = torch.zeros(total_tiles, device=device, dtype=torch.int32)
    tile_mask[active_tile_ids.long()] = 1

    pixel_mask = tile_mask.view(tiles_y, tiles_x).to(torch.bool)
    pixel_mask = pixel_mask.repeat_interleave(tile_size, dim=0).repeat_interleave(tile_size, dim=1)
    pixel_mask = pixel_mask[:image_height, :image_width].unsqueeze(0)

    num_active_tiles = int(active_tile_ids.numel())
    active_ratio = float(num_active_tiles) / float(total_tiles) if total_tiles > 0 else 1.0

    return {
        "tile_mask": tile_mask,
        "active_tile_ids": active_tile_ids,
        "pixel_mask": pixel_mask,
        "num_active_tiles": num_active_tiles,
        "total_tiles": total_tiles,
        "active_ratio": active_ratio,
        "tiles_y": tiles_y,
        "tiles_x": tiles_x,
    }
