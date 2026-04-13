import argparse
import csv
import os
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont


def mean_by_key(rows, key_name, value_names):
    grouped = defaultdict(lambda: {name: [] for name in value_names})
    for row in rows:
        key = row[key_name]
        for value_name in value_names:
            grouped[key][value_name].append(float(row[value_name]))

    summary = {}
    for key, values in grouped.items():
        summary[key] = {}
        for value_name, items in values.items():
            summary[key][value_name] = sum(items) / max(1, len(items))
    return summary


def write_summary_csv(summary, value_name, output_path):
    sorted_items = sorted(summary.items(), key=lambda item: item[1][value_name], reverse=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", value_name])
        for name, values in sorted_items:
            writer.writerow([name, f"{values[value_name]:.6f}"])


def draw_bar_chart(data, title, output_path):
    if not data:
        return

    sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=True)
    width = 1200
    row_height = 42
    left_margin = 330
    right_margin = 80
    top_margin = 80
    bottom_margin = 40
    bar_height = 24
    chart_width = width - left_margin - right_margin
    height = top_margin + bottom_margin + row_height * len(sorted_items)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    max_value = max(value for _, value in sorted_items) or 1.0

    draw.text((20, 20), title, fill="black", font=font)
    for idx, (label, value) in enumerate(sorted_items):
        y = top_margin + idx * row_height
        bar_width = int((value / max_value) * chart_width)
        draw.text((20, y), label, fill="black", font=font)
        draw.rectangle(
            [left_margin, y, left_margin + bar_width, y + bar_height],
            fill=(66, 133, 244),
            outline=(30, 30, 30),
        )
        draw.text((left_margin + bar_width + 10, y), f"{value:.3f} ms", fill="black", font=font)

    image.save(output_path)


def load_csv_rows(csv_path, iter_start, iter_end):
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        row for row in rows
        if iter_start <= int(row["iteration"]) <= iter_end
    ]


def main():
    parser = argparse.ArgumentParser(description="Summarize profiling outputs.")
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--iter_start", type=int, default=300)
    parser.add_argument("--iter_end", type=int, default=500)
    args = parser.parse_args()

    profile_dir = os.path.abspath(args.profile_dir)
    train_rows = load_csv_rows(os.path.join(profile_dir, "train_iteration_profile.csv"), args.iter_start, args.iter_end)
    raster_rows = load_csv_rows(os.path.join(profile_dir, "rasterizer_profile.csv"), args.iter_start, args.iter_end)

    train_summary = mean_by_key(train_rows, "section", ["cuda_ms_total", "wall_ms_total"])
    write_summary_csv(train_summary, "cuda_ms_total", os.path.join(profile_dir, "train_section_cuda_summary.csv"))
    write_summary_csv(train_summary, "wall_ms_total", os.path.join(profile_dir, "train_section_wall_summary.csv"))

    draw_bar_chart(
        {name: values["cuda_ms_total"] for name, values in train_summary.items()},
        f"Training section avg CUDA time ({args.iter_start}-{args.iter_end})",
        os.path.join(profile_dir, "train_section_cuda_avg.png"),
    )
    draw_bar_chart(
        {name: values["wall_ms_total"] for name, values in train_summary.items()},
        f"Training section avg wall time ({args.iter_start}-{args.iter_end})",
        os.path.join(profile_dir, "train_section_wall_avg.png"),
    )

    forward_rows = [row for row in raster_rows if row["phase"] == "forward"]
    backward_rows = [row for row in raster_rows if row["phase"] == "backward"]
    forward_metrics = [
        "preprocess_ms",
        "scan_ms",
        "copy_rendered_ms",
        "duplicate_ms",
        "sort_ms",
        "zero_ranges_ms",
        "identify_ranges_ms",
        "render_ms",
        "copy_alpha_ms",
        "total_ms",
    ]
    backward_metrics = [
        "backward_render_ms",
        "backward_preprocess_ms",
        "total_ms",
    ]

    forward_summary = {
        metric: sum(float(row[metric]) for row in forward_rows) / max(1, len(forward_rows))
        for metric in forward_metrics
    }
    backward_summary = {
        metric: sum(float(row[metric]) for row in backward_rows) / max(1, len(backward_rows))
        for metric in backward_metrics
    }

    with open(os.path.join(profile_dir, "rasterizer_forward_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "avg_ms"])
        for metric, value in sorted(forward_summary.items(), key=lambda item: item[1], reverse=True):
            writer.writerow([metric, f"{value:.6f}"])

    with open(os.path.join(profile_dir, "rasterizer_backward_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "avg_ms"])
        for metric, value in sorted(backward_summary.items(), key=lambda item: item[1], reverse=True):
            writer.writerow([metric, f"{value:.6f}"])

    draw_bar_chart(
        {k: v for k, v in forward_summary.items() if k != "total_ms"},
        f"Rasterizer forward avg CUDA time ({args.iter_start}-{args.iter_end})",
        os.path.join(profile_dir, "rasterizer_forward_avg.png"),
    )
    draw_bar_chart(
        {k: v for k, v in backward_summary.items() if k != "total_ms"},
        f"Rasterizer backward avg CUDA time ({args.iter_start}-{args.iter_end})",
        os.path.join(profile_dir, "rasterizer_backward_avg.png"),
    )

    lines = [
        f"iteration_range: {args.iter_start}-{args.iter_end}",
        "",
        "train_section_cuda_avg_ms:",
    ]
    for name, values in sorted(train_summary.items(), key=lambda item: item[1]["cuda_ms_total"], reverse=True):
        lines.append(f"- {name}: {values['cuda_ms_total']:.4f}")

    lines.append("")
    lines.append("train_section_wall_avg_ms:")
    for name, values in sorted(train_summary.items(), key=lambda item: item[1]["wall_ms_total"], reverse=True):
        lines.append(f"- {name}: {values['wall_ms_total']:.4f}")

    lines.append("")
    lines.append("rasterizer_forward_avg_ms:")
    for metric, value in sorted(forward_summary.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {metric}: {value:.4f}")

    lines.append("")
    lines.append("rasterizer_backward_avg_ms:")
    for metric, value in sorted(backward_summary.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {metric}: {value:.4f}")

    with open(os.path.join(profile_dir, "profile_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
