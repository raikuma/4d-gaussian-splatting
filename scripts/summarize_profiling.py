import argparse
import csv
from pathlib import Path


PYTHON_KEYS = [
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
]

RASTER_FORWARD_KEYS = [
    "raster_forward_preprocess_ms",
    "raster_forward_active_filter_ms",
    "raster_forward_scan_ms",
    "raster_forward_duplicate_ms",
    "raster_forward_sort_ms",
    "raster_forward_ranges_ms",
    "raster_forward_render_ms",
    "raster_forward_total_ms",
]

FULL_RENDER_FORWARD_KEYS = [
    "full_render_raster_forward_preprocess_ms",
    "full_render_raster_forward_active_filter_ms",
    "full_render_raster_forward_scan_ms",
    "full_render_raster_forward_duplicate_ms",
    "full_render_raster_forward_sort_ms",
    "full_render_raster_forward_ranges_ms",
    "full_render_raster_forward_render_ms",
    "full_render_raster_forward_total_ms",
]

RASTER_BACKWARD_KEYS = [
    "raster_backward_render_ms",
    "raster_backward_preprocess_ms",
    "raster_backward_total_ms",
]


def load_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def mean(rows, key):
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    return sum(values) / len(values) if values else 0.0


def print_section(title, rows, keys):
    print(title)
    for key in keys:
        print(f"  {key}: {mean(rows, key):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Summarize profiling CSV.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--skip-first", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows(args.csv_path)[args.skip_first:]
    if not rows:
        raise ValueError("No profiling rows left after skip_first")

    print(f"rows: {len(rows)}")
    print(f"avg_tile_ratio: {mean(rows, 'avg_tile_ratio'):.4f}")
    print(f"psnr: {mean(rows, 'psnr'):.4f}")
    print(f"gpu_mem_alloc_mb: {mean(rows, 'gpu_mem_alloc_mb'):.2f}")
    print()
    print_section("python_breakdown", rows, PYTHON_KEYS)
    print()
    print_section("raster_forward_breakdown", rows, RASTER_FORWARD_KEYS)
    print()
    print_section("full_render_forward_breakdown", rows, FULL_RENDER_FORWARD_KEYS)
    print()
    print_section("raster_backward_breakdown", rows, RASTER_BACKWARD_KEYS)


if __name__ == "__main__":
    main()
