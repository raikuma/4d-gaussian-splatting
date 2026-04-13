import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


BG = "#F4EFE7"
CARD = "#FFFDFC"
CARD_ALT = "#F8F3EB"
TEXT = "#1A2335"
MUTED = "#5E6779"
OUTLINE = "#D7CFBF"
GRID = "#E8DFD2"

SEGMENTS = [
    ("view_to_cuda", "View To CUDA", "#9BB4D1"),
    ("view_tile_mask", "Tile Mask", "#6FA8DC"),
    ("raster_forward_render", "Raster Fwd Render", "#2D8C8C"),
    ("raster_forward_sort", "Raster Fwd Sort", "#58B6BA"),
    ("raster_forward_other", "Raster Fwd Other", "#9AD9D3"),
    ("render_non_raster", "Render Non-Raster", "#C2ECE6"),
    ("view_loss", "View Loss", "#E7C35A"),
    ("raster_backward_render", "Raster Bwd Render", "#C95647"),
    ("raster_backward_other", "Raster Bwd Other", "#E89262"),
    ("backward_non_raster", "Bwd Non-Raster", "#F2C88B"),
    ("batch_reduce", "Batch Reduce", "#BDA7D9"),
    ("densify", "Densify", "#86A95E"),
    ("optimizer", "Optimizer", "#556B2F"),
    ("lr_and_sh", "LR / SH", "#C8CDD6"),
    ("residual", "Residual", "#D9DDE4"),
]


def load_font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/bahnschrift.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_text(draw, pos, text, font, fill=TEXT, anchor=None):
    draw.text(pos, text, font=font, fill=fill, anchor=anchor)


def rounded_card(draw, box, fill=CARD):
    draw.rounded_rectangle(box, radius=28, fill=fill, outline=OUTLINE, width=2)


def mean_section_cuda_ms(profile_csv, section):
    values = []
    with open(profile_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["section"] == section:
                values.append(float(row["cuda_ms_total"]))
    return statistics.mean(values) if values else 0.0


def mean_iteration_cuda_ms(profile_csv):
    per_iter = defaultdict(float)
    with open(profile_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            per_iter[int(row["iteration"])] += float(row["cuda_ms_total"])
    return statistics.mean(per_iter.values()) if per_iter else 0.0


def mean_raster_metric(raster_csv, phase, metric):
    values = []
    with open(raster_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["phase"] == phase:
                values.append(float(row[metric]))
    return statistics.mean(values) if values else 0.0


def case_sort_key(case_name):
    if case_name == "baseline":
        return (0, 1.0)
    ratio = float(case_name.split("_", 1)[1])
    return (1, -ratio)


def read_case(case_dir):
    profile_dir = case_dir / "profiling"
    train_csv = profile_dir / "train_iteration_profile.csv"
    raster_csv = profile_dir / "rasterizer_profile.csv"

    iter_total = mean_iteration_cuda_ms(train_csv)
    view_render = mean_section_cuda_ms(train_csv, "view_render")
    view_backward = mean_section_cuda_ms(train_csv, "view_backward")

    raster_forward_render = mean_raster_metric(raster_csv, "forward", "render_ms")
    raster_forward_sort = mean_raster_metric(raster_csv, "forward", "sort_ms")
    raster_forward_total = mean_raster_metric(raster_csv, "forward", "total_ms")
    raster_backward_render = mean_raster_metric(raster_csv, "backward", "backward_render_ms")
    raster_backward_total = mean_raster_metric(raster_csv, "backward", "total_ms")
    active_tile_ratio = mean_raster_metric(raster_csv, "forward", "active_tile_ratio")

    values = {
        "view_to_cuda": mean_section_cuda_ms(train_csv, "view_to_cuda"),
        "view_tile_mask": mean_section_cuda_ms(train_csv, "view_tile_mask"),
        "raster_forward_render": raster_forward_render,
        "raster_forward_sort": raster_forward_sort,
        "raster_forward_other": max(0.0, raster_forward_total - raster_forward_render - raster_forward_sort),
        "render_non_raster": max(0.0, view_render - raster_forward_total),
        "view_loss": mean_section_cuda_ms(train_csv, "view_loss"),
        "raster_backward_render": raster_backward_render,
        "raster_backward_other": max(0.0, raster_backward_total - raster_backward_render),
        "backward_non_raster": max(0.0, view_backward - raster_backward_total),
        "batch_reduce": mean_section_cuda_ms(train_csv, "batch_reduce"),
        "densify": mean_section_cuda_ms(train_csv, "densify"),
        "optimizer": mean_section_cuda_ms(train_csv, "optimizer"),
        "lr_and_sh": mean_section_cuda_ms(train_csv, "lr_and_sh"),
    }
    known_total = sum(values.values())
    values["residual"] = max(0.0, iter_total - known_total)

    return {
        "case": case_dir.name,
        "iter_total": iter_total,
        "active_tile_ratio": active_tile_ratio,
        "values": values,
    }


def format_case_label(case_name, active_tile_ratio):
    if case_name == "baseline":
        return "Baseline\n(1.000)"
    return f"Tile {active_tile_ratio:.3f}"


def draw_summary_cards(draw, box, cases):
    x0, y0, x1, y1 = box
    baseline = next(case for case in cases if case["case"] == "baseline")
    best = min(cases, key=lambda case: case["iter_total"])
    best_speedup = baseline["iter_total"] / best["iter_total"]
    best_tile = best["active_tile_ratio"]
    tile_mask_best = next(case for case in cases if abs(case["active_tile_ratio"] - 0.125) < 1e-6)
    cards = [
        ("Best Speedup", f"{best_speedup:.2f}x", f"best at tile ratio {best_tile:.3f}", "#C95647"),
        ("Baseline Iter", f"{baseline['iter_total']:.1f} ms", "average CUDA time per iter", "#2D8C8C"),
        ("Tile Mask Cost", f"{tile_mask_best['values']['view_tile_mask']:.2f} ms", "at tile ratio 0.125", "#6FA8DC"),
        ("Optimizer Floor", f"{tile_mask_best['values']['optimizer']:.2f} ms", "still present at low tile ratio", "#556B2F"),
    ]
    gap = 18
    card_w = (x1 - x0 - gap * (len(cards) - 1)) // len(cards)
    title_font = load_font(16, bold=True)
    value_font = load_font(34, bold=True)
    subtitle_font = load_font(15)
    for idx, (title, value, subtitle, color) in enumerate(cards):
        cx0 = x0 + idx * (card_w + gap)
        cx1 = cx0 + card_w
        rounded_card(draw, (cx0, y0, cx1, y1), fill=CARD_ALT)
        draw_text(draw, (cx0 + 22, y0 + 18), title, title_font, MUTED)
        draw_text(draw, (cx0 + 22, y0 + 48), value, value_font, color)
        draw_text(draw, (cx0 + 22, y0 + 92), subtitle, subtitle_font, MUTED)


def draw_stacked_bars(draw, box, cases):
    x0, y0, x1, y1 = box
    rounded_card(draw, box)
    title_font = load_font(30, bold=True)
    subtitle_font = load_font(17)
    axis_font = load_font(15)
    label_font = load_font(18, bold=True)
    tiny_font = load_font(14, bold=True)

    draw_text(draw, (x0 + 28, y0 + 20), "Detailed Iteration Breakdown", title_font)
    draw_text(
        draw,
        (x0 + 28, y0 + 58),
        "Former 'Other' is expanded into render non-raster, view loss, optimizer, densify, tile mask, transfer, and tiny residuals.",
        subtitle_font,
        MUTED,
    )

    chart_x0 = x0 + 92
    chart_x1 = x1 - 34
    chart_y0 = y0 + 138
    chart_y1 = y1 - 124
    max_total = max(case["iter_total"] for case in cases)
    ticks = [0, 20, 40, 60, 80]
    for tick in ticks:
        ty = chart_y1 - (tick / max_total) * (chart_y1 - chart_y0)
        draw.line((chart_x0, ty, chart_x1, ty), fill=GRID, width=2)
        draw_text(draw, (chart_x0 - 18, ty), f"{tick}", axis_font, MUTED, anchor="rm")
    draw_text(draw, (chart_x0 - 18, chart_y0 - 24), "ms", axis_font, MUTED, anchor="rm")

    bar_gap = 34
    bar_w = int((chart_x1 - chart_x0 - bar_gap * (len(cases) - 1)) / len(cases))
    chart_h = chart_y1 - chart_y0
    baseline_total = next(case for case in cases if case["case"] == "baseline")["iter_total"]

    for idx, case in enumerate(cases):
        bx0 = chart_x0 + idx * (bar_w + bar_gap)
        bx1 = bx0 + bar_w
        cursor = chart_y1
        for key, _, color in SEGMENTS:
            value = case["values"][key]
            if value <= 0.0:
                continue
            height = chart_h * (value / max_total)
            by0 = cursor - height
            draw.rounded_rectangle((bx0, by0, bx1, cursor), radius=14, fill=color)
            if value >= 3.0:
                draw_text(draw, ((bx0 + bx1) / 2, (by0 + cursor) / 2), f"{value:.1f}", tiny_font, "white", anchor="mm")
            cursor = by0

        speedup = baseline_total / case["iter_total"]
        draw_text(draw, ((bx0 + bx1) / 2, chart_y1 + 24), format_case_label(case["case"], case["active_tile_ratio"]), label_font, TEXT, anchor="ma")
        draw_text(draw, ((bx0 + bx1) / 2, cursor - 14), f"{case['iter_total']:.1f} ms", label_font, TEXT, anchor="ms")
        draw_text(
            draw,
            ((bx0 + bx1) / 2, cursor - 36),
            f"{speedup:.2f}x",
            axis_font,
            "#C95647" if case["case"] != "baseline" else MUTED,
            anchor="ms",
        )


def draw_legend(draw, box):
    x0, y0, x1, y1 = box
    rounded_card(draw, box, fill=CARD_ALT)
    title_font = load_font(24, bold=True)
    label_font = load_font(16, bold=True)
    desc_font = load_font(14)
    draw_text(draw, (x0 + 24, y0 + 18), "Legend", title_font)

    descriptions = {
        "view_to_cuda": "camera/image transfer into CUDA-visible tensors",
        "view_tile_mask": "tile selection and mask construction",
        "raster_forward_render": "forward render kernel body",
        "raster_forward_sort": "duplicate/sort-heavy forward work",
        "raster_forward_other": "scan, zeroing, identify-ranges, copies",
        "render_non_raster": "Python/Torch render wrapper outside rasterizer",
        "view_loss": "loss evaluation on rendered output",
        "raster_backward_render": "backward render kernel body",
        "raster_backward_other": "backward preprocess and remaining raster CUDA",
        "backward_non_raster": "autograd work outside rasterizer",
        "batch_reduce": "batch loss aggregation",
        "densify": "densification / prune bookkeeping",
        "optimizer": "optimizer step and state updates",
        "lr_and_sh": "learning-rate / SH bookkeeping",
        "residual": "small remainder from averaging boundaries",
    }

    columns = 2
    rows_per_col = (len(SEGMENTS) + columns - 1) // columns
    col_w = (x1 - x0 - 48) // columns
    for idx, (key, label, color) in enumerate(SEGMENTS):
        col = idx // rows_per_col
        row = idx % rows_per_col
        base_x = x0 + 24 + col * col_w
        y = y0 + 64 + row * 52
        draw.rounded_rectangle((base_x, y + 5, base_x + 18, y + 23), radius=5, fill=color)
        draw_text(draw, (base_x + 30, y), label, label_font)
        draw_text(draw, (base_x + 30, y + 20), descriptions[key], desc_font, MUTED)


def draw_takeaways(draw, box, cases):
    x0, y0, x1, y1 = box
    rounded_card(draw, box, fill=CARD_ALT)
    title_font = load_font(24, bold=True)
    label_font = load_font(18, bold=True)
    body_font = load_font(17)
    draw_text(draw, (x0 + 24, y0 + 18), "What Replaced 'Other'", title_font)

    baseline = next(case for case in cases if case["case"] == "baseline")
    low = min(cases, key=lambda case: case["iter_total"])

    baseline_other = {
        "optimizer": baseline["values"]["optimizer"],
        "view_loss": baseline["values"]["view_loss"],
        "render_non_raster": baseline["values"]["render_non_raster"],
        "densify": baseline["values"]["densify"],
    }
    low_other = {
        "optimizer": low["values"]["optimizer"],
        "view_loss": low["values"]["view_loss"],
        "view_tile_mask": low["values"]["view_tile_mask"],
        "render_non_raster": low["values"]["render_non_raster"],
    }

    lines = [
        ("Baseline floor", f"Optimizer {baseline_other['optimizer']:.1f} ms and view loss {baseline_other['view_loss']:.1f} ms were already large."),
        ("At low tile ratio", f"Optimizer {low_other['optimizer']:.1f} ms and view loss {low_other['view_loss']:.1f} ms dominate the remaining stack."),
        ("Tile-mask overhead", f"Tile mask reaches {low_other['view_tile_mask']:.2f} ms at 0.125, but is still much smaller than the rasterizer savings."),
        ("Render wrapper", f"Render non-raster stays around {low_other['render_non_raster']:.1f} ms even after raster work shrinks."),
    ]

    y = y0 + 62
    for title, text in lines:
        draw_text(draw, (x0 + 24, y), title, label_font, "#C95647")
        wrapped = textwrap.wrap(text, width=44) or [text]
        for idx, line in enumerate(wrapped):
            draw_text(draw, (x0 + 190, y + idx * 20), line, body_font)
        y += max(42, 20 * len(wrapped) + 12)


def write_csv(output_csv, cases):
    fieldnames = ["case", "active_tile_ratio", "iter_total"] + [key for key, _, _ in SEGMENTS]
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            row = {
                "case": case["case"],
                "active_tile_ratio": f"{case['active_tile_ratio']:.6f}",
                "iter_total": f"{case['iter_total']:.6f}",
            }
            for key, _, _ in SEGMENTS:
                row[key] = f"{case['values'][key]:.6f}"
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Render a detailed tile-ratio profiling breakdown from existing sweep outputs.")
    parser.add_argument("--sweep_root", required=True)
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    case_dirs = [path for path in sweep_root.iterdir() if path.is_dir() and (path / "profiling").exists()]
    cases = [read_case(case_dir) for case_dir in sorted(case_dirs, key=lambda path: case_sort_key(path.name))]

    width, height = 1960, 1380
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(52, bold=True)
    subtitle_font = load_font(20)
    kicker_font = load_font(18, bold=True)
    draw_text(draw, (64, 40), "Tile-Wise Training Profiling", title_font)
    draw_text(draw, (66, 102), "Detailed stacked iteration breakdown from existing sweep logs. The former gray 'Other' bucket is explicitly decomposed.", subtitle_font, MUTED)
    draw_text(draw, (66, 136), "Cook Spinach debug / 300-500 iter average / CUDA-time view / no re-run needed", kicker_font, "#C95647")

    draw_summary_cards(draw, (64, 184, 1896, 322), cases)
    draw_stacked_bars(draw, (64, 356, 1232, 1296), cases)
    draw_legend(draw, (1266, 356, 1896, 860))
    draw_takeaways(draw, (1266, 888, 1896, 1296), cases)

    out_png = sweep_root / "tile_ratio_breakdown_detailed.png"
    out_csv = sweep_root / "tile_ratio_breakdown_detailed.csv"
    canvas.save(out_png)
    write_csv(out_csv, cases)
    print(out_png)
    print(out_csv)


if __name__ == "__main__":
    main()
