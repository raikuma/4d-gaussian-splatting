import argparse
import csv
import os
import statistics
import subprocess
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_iteration_cuda_ms(train_rows):
    per_iter = defaultdict(float)
    for row in train_rows:
        per_iter[int(row["iteration"])] += float(row["cuda_ms_total"])
    return statistics.mean(per_iter.values())


def mean_section_cuda_ms(train_rows, section_name):
    values = [float(row["cuda_ms_total"]) for row in train_rows if row["section"] == section_name]
    return statistics.mean(values) if values else 0.0


def mean_raster_metric(raster_rows, phase, metric_name):
    values = [float(row[metric_name]) for row in raster_rows if row["phase"] == phase]
    return statistics.mean(values) if values else 0.0


def mean_active_tile_ratio(raster_rows):
    values = [float(row["active_tile_ratio"]) for row in raster_rows if row["phase"] == "forward"]
    return statistics.mean(values) if values else 1.0


def run_case(args, case_name, model_path, profile_dir, tile_ratio=None):
    command = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        "train.py",
        "--config",
        args.config,
        "--model_path",
        model_path,
        "--iterations",
        str(args.iterations),
        "--test_iterations",
        str(args.iterations),
        "--save_iterations",
        str(args.iterations),
        "--profile_training",
        "--profile_rasterizer",
        "--profile_from_iter",
        str(args.profile_from_iter),
        "--profile_warmup_iters",
        str(args.profile_warmup_iters),
        "--profile_iters",
        str(args.profile_iters),
        "--profile_output_dir",
        profile_dir,
        "--quiet",
    ]
    if tile_ratio is not None:
        command.extend(
            [
                "--tile_training",
                "--tile_ratio",
                str(tile_ratio),
                "--tile_selection_mode",
                "random",
            ]
        )

    print(f"[RUN] {case_name}: {' '.join(command)}")
    subprocess.run(command, check=True)


def summarize_case(case_name, tile_ratio, profile_dir):
    train_rows = load_csv_rows(os.path.join(profile_dir, "train_iteration_profile.csv"))
    raster_rows = load_csv_rows(os.path.join(profile_dir, "rasterizer_profile.csv"))
    return {
        "case": case_name,
        "tile_ratio": 1.0 if tile_ratio is None else float(tile_ratio),
        "measured_active_tile_ratio": mean_active_tile_ratio(raster_rows),
        "iter_cuda_ms": mean_iteration_cuda_ms(train_rows),
        "view_render_cuda_ms": mean_section_cuda_ms(train_rows, "view_render"),
        "view_backward_cuda_ms": mean_section_cuda_ms(train_rows, "view_backward"),
        "raster_forward_total_ms": mean_raster_metric(raster_rows, "forward", "total_ms"),
        "raster_forward_render_ms": mean_raster_metric(raster_rows, "forward", "render_ms"),
        "raster_forward_sort_ms": mean_raster_metric(raster_rows, "forward", "sort_ms"),
        "raster_backward_total_ms": mean_raster_metric(raster_rows, "backward", "total_ms"),
        "raster_backward_render_ms": mean_raster_metric(raster_rows, "backward", "backward_render_ms"),
    }


def draw_bar_chart(records, label_key, value_key, title, output_path, formatter):
    width = 1200
    row_height = 44
    left_margin = 320
    right_margin = 130
    top_margin = 80
    bottom_margin = 30
    chart_width = width - left_margin - right_margin
    height = top_margin + bottom_margin + row_height * len(records)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    max_value = max(record[value_key] for record in records) or 1.0

    draw.text((20, 20), title, fill="black", font=font)
    for idx, record in enumerate(records):
        y = top_margin + idx * row_height
        value = record[value_key]
        label = record[label_key]
        bar_width = int((value / max_value) * chart_width)
        draw.text((20, y), str(label), fill="black", font=font)
        draw.rectangle(
            [left_margin, y, left_margin + bar_width, y + 26],
            fill=(52, 119, 235),
            outline=(30, 30, 30),
        )
        draw.text((left_margin + bar_width + 10, y), formatter(value), fill="black", font=font)

    image.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Profile tile ratio sweep for tile-wise training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--conda_env", default="4dgs")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--profile_from_iter", type=int, default=280)
    parser.add_argument("--profile_warmup_iters", type=int, default=20)
    parser.add_argument("--profile_iters", type=int, default=201)
    parser.add_argument("--tile_ratios", nargs="+", type=float, default=[0.75, 0.5, 0.25, 0.125])
    parser.add_argument("--skip_runs", action="store_true")
    args = parser.parse_args()

    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    cases = [("baseline", None)] + [(f"tile_{ratio:g}", ratio) for ratio in args.tile_ratios]
    summaries = []

    for case_name, tile_ratio in cases:
        model_path = os.path.join(output_root, case_name)
        profile_dir = os.path.join(model_path, "profiling")
        os.makedirs(model_path, exist_ok=True)

        if not args.skip_runs:
            run_case(args, case_name, model_path, profile_dir, tile_ratio=tile_ratio)

        summaries.append(summarize_case(case_name, tile_ratio, profile_dir))

    baseline_iter_ms = summaries[0]["iter_cuda_ms"]
    for summary in summaries:
        summary["speedup_vs_baseline"] = baseline_iter_ms / summary["iter_cuda_ms"]

    csv_path = os.path.join(output_root, "tile_ratio_sweep_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    report_lines = ["tile ratio sweep summary", ""]
    for summary in summaries:
        report_lines.append(
            (
                f"{summary['case']}: "
                f"tile_ratio={summary['tile_ratio']:.3f}, "
                f"measured_active_tile_ratio={summary['measured_active_tile_ratio']:.3f}, "
                f"iter_cuda_ms={summary['iter_cuda_ms']:.3f}, "
                f"speedup_vs_baseline={summary['speedup_vs_baseline']:.3f}"
            )
        )

    with open(os.path.join(output_root, "tile_ratio_sweep_report.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")

    ordered = summaries
    draw_bar_chart(
        ordered,
        label_key="case",
        value_key="iter_cuda_ms",
        title="Average Iteration CUDA Time",
        output_path=os.path.join(output_root, "tile_ratio_iter_cuda_ms.png"),
        formatter=lambda value: f"{value:.2f} ms",
    )
    draw_bar_chart(
        ordered,
        label_key="case",
        value_key="speedup_vs_baseline",
        title="Speedup vs Baseline",
        output_path=os.path.join(output_root, "tile_ratio_speedup.png"),
        formatter=lambda value: f"{value:.2f}x",
    )


if __name__ == "__main__":
    main()
