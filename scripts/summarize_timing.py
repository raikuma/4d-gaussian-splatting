import argparse
import csv
from pathlib import Path


METRICS = [
    "avg_tile_ratio",
    "iter_time_ms",
    "mask_update_ms",
    "mask_build_ms",
    "full_render_ms",
    "train_render_ms",
    "backward_ms",
    "optimizer_ms",
    "gpu_mem_alloc_mb",
    "gpu_mem_reserved_mb",
    "psnr",
    "l1",
    "ssim_loss",
]


def load_rows(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def to_float(row, key):
    return float(row[key])


def summarize(rows, skip_first):
    filtered = rows[skip_first:]
    if not filtered:
        raise ValueError("No rows left after applying skip_first")

    summary = {"count": len(filtered)}
    for metric in METRICS:
        summary[metric] = sum(to_float(row, metric) for row in filtered) / len(filtered)
    summary["step_total_ms"] = (
        summary["iter_time_ms"] + summary["optimizer_ms"]
    )
    return summary


def print_summary(label, summary):
    print(label)
    print(f"  rows: {summary['count']}")
    print(f"  avg_tile_ratio: {summary['avg_tile_ratio']:.4f}")
    print(f"  iter_time_ms: {summary['iter_time_ms']:.3f}")
    print(f"  optimizer_ms: {summary['optimizer_ms']:.3f}")
    print(f"  step_total_ms: {summary['step_total_ms']:.3f}")
    print(f"  train_render_ms: {summary['train_render_ms']:.3f}")
    print(f"  backward_ms: {summary['backward_ms']:.3f}")
    print(f"  mask_update_ms: {summary['mask_update_ms']:.3f}")
    print(f"  mask_build_ms: {summary['mask_build_ms']:.3f}")
    print(f"  full_render_ms: {summary['full_render_ms']:.3f}")
    print(f"  gpu_mem_alloc_mb: {summary['gpu_mem_alloc_mb']:.1f}")
    print(f"  gpu_mem_reserved_mb: {summary['gpu_mem_reserved_mb']:.1f}")
    print(f"  psnr: {summary['psnr']:.4f}")
    print(f"  l1: {summary['l1']:.6f}")
    print(f"  ssim_loss: {summary['ssim_loss']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Summarize FocusGS timing logs.")
    parser.add_argument("primary", type=Path)
    parser.add_argument("--label", default="primary")
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--compare-label", default="compare")
    parser.add_argument("--skip-first", type=int, default=0)
    args = parser.parse_args()

    primary_summary = summarize(load_rows(args.primary), args.skip_first)
    print_summary(args.label, primary_summary)

    if args.compare is not None:
        compare_summary = summarize(load_rows(args.compare), args.skip_first)
        print()
        print_summary(args.compare_label, compare_summary)
        print()
        speedup = compare_summary["step_total_ms"] / primary_summary["step_total_ms"]
        render_speedup = compare_summary["train_render_ms"] / primary_summary["train_render_ms"]
        print("relative_to_compare")
        print(f"  step_speedup: {speedup:.4f}x")
        print(f"  render_speedup: {render_speedup:.4f}x")
        print(f"  tile_ratio_reduction: {1.0 - primary_summary['avg_tile_ratio']:.4f}")


if __name__ == "__main__":
    main()
