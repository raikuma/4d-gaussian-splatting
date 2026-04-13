import argparse
import csv
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


BG = "#F4EFE7"
CARD = "#FFFDFC"
CARD_ALT = "#F9F4EC"
TEXT = "#182033"
MUTED = "#5C6575"
OUTLINE = "#D9D0C2"
GRID = "#E8E0D4"

COLORS = {
    "raster_backward_render": "#C65446",
    "raster_backward_other": "#E28A61",
    "backward_non_raster": "#F0BE5C",
    "raster_forward_render": "#2E8A90",
    "raster_forward_sort": "#4FA7B0",
    "raster_forward_other": "#8BC9CC",
    "other_iter": "#B7C1CF",
}


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


def rounded_card(draw, box, fill=CARD):
    draw.rounded_rectangle(box, radius=28, fill=fill, outline=OUTLINE, width=2)


def draw_text(draw, pos, text, font, fill=TEXT, anchor=None):
    draw.text(pos, text, font=font, fill=fill, anchor=anchor)


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def format_case_label(case_name):
    if case_name == "baseline":
        return "Baseline\n(1.0)"
    ratio = case_name.replace("tile_", "")
    return f"Tile {ratio}"


def prepare_rows(rows):
    prepared = []
    for row in rows:
        iter_total = float(row["iter_cuda_ms"])
        raster_backward_render = float(row["raster_backward_render_ms"])
        raster_backward_total = float(row["raster_backward_total_ms"])
        view_backward = float(row["view_backward_cuda_ms"])
        raster_forward_render = float(row["raster_forward_render_ms"])
        raster_forward_sort = float(row["raster_forward_sort_ms"])
        raster_forward_total = float(row["raster_forward_total_ms"])

        raster_backward_other = max(0.0, raster_backward_total - raster_backward_render)
        backward_non_raster = max(0.0, view_backward - raster_backward_total)
        raster_forward_other = max(0.0, raster_forward_total - raster_forward_render - raster_forward_sort)
        known = (
            raster_backward_render
            + raster_backward_other
            + backward_non_raster
            + raster_forward_render
            + raster_forward_sort
            + raster_forward_other
        )
        other_iter = max(0.0, iter_total - known)

        prepared.append(
            {
                "case": row["case"],
                "label": format_case_label(row["case"]),
                "tile_ratio": float(row["tile_ratio"]),
                "speedup": float(row["speedup_vs_baseline"]),
                "iter_total": iter_total,
                "segments": [
                    ("Raster Bwd Render", raster_backward_render, COLORS["raster_backward_render"]),
                    ("Raster Bwd Other", raster_backward_other, COLORS["raster_backward_other"]),
                    ("Bwd Non-Raster", backward_non_raster, COLORS["backward_non_raster"]),
                    ("Raster Fwd Render", raster_forward_render, COLORS["raster_forward_render"]),
                    ("Raster Fwd Sort", raster_forward_sort, COLORS["raster_forward_sort"]),
                    ("Raster Fwd Other", raster_forward_other, COLORS["raster_forward_other"]),
                    ("Other Iter", other_iter, COLORS["other_iter"]),
                ],
            }
        )
    return prepared


def draw_summary_cards(draw, box, rows):
    x0, y0, x1, y1 = box
    cards = [
        ("Best Speedup", f"{max(row['speedup'] for row in rows):.2f}x", "at tile ratio 0.125", COLORS["raster_backward_render"]),
        ("Baseline Iter", f"{rows[0]['iter_total']:.1f} ms", "average CUDA time per iter", COLORS["raster_forward_render"]),
        ("Fastest Iter", f"{min(row['iter_total'] for row in rows):.1f} ms", "tile ratio 0.125", COLORS["raster_backward_other"]),
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


def draw_stacked_bars(draw, box, rows):
    x0, y0, x1, y1 = box
    rounded_card(draw, box)
    title_font = load_font(30, bold=True)
    subtitle_font = load_font(17)
    axis_font = load_font(15)
    label_font = load_font(18, bold=True)
    value_font = load_font(16, bold=True)

    draw_text(draw, (x0 + 28, y0 + 20), "Tile Ratio Breakdown", title_font)
    draw_text(
        draw,
        (x0 + 28, y0 + 58),
        "Stacked average iteration CUDA time split by rasterizer forward/backward and remaining work.",
        subtitle_font,
        MUTED,
    )

    chart_x0 = x0 + 86
    chart_x1 = x1 - 48
    chart_y0 = y0 + 150
    chart_y1 = y1 - 108
    max_total = max(row["iter_total"] for row in rows)
    tick_values = [0, 20, 40, 60, 80]
    for tick in tick_values:
        ty = chart_y1 - (tick / max_total) * (chart_y1 - chart_y0)
        draw.line((chart_x0, ty, chart_x1, ty), fill=GRID, width=2)
        draw_text(draw, (chart_x0 - 18, ty), f"{tick}", axis_font, MUTED, anchor="rm")
    draw_text(draw, (chart_x0 - 18, chart_y0 - 24), "ms", axis_font, MUTED, anchor="rm")

    bar_gap = 34
    bar_w = int((chart_x1 - chart_x0 - bar_gap * (len(rows) - 1)) / len(rows))
    chart_h = chart_y1 - chart_y0

    for idx, row in enumerate(rows):
        bx0 = chart_x0 + idx * (bar_w + bar_gap)
        bx1 = bx0 + bar_w
        by = chart_y1
        for _, value, color in row["segments"]:
            h = chart_h * (value / max_total)
            by0 = by - h
            draw.rounded_rectangle((bx0, by0, bx1, by), radius=16, fill=color)
            by = by0

        draw_text(draw, ((bx0 + bx1) / 2, chart_y1 + 22), row["label"], label_font, TEXT, anchor="ma")
        draw_text(draw, ((bx0 + bx1) / 2, by - 12), f"{row['iter_total']:.1f} ms", value_font, TEXT, anchor="ms")
        draw_text(
            draw,
            ((bx0 + bx1) / 2, by - 34),
            f"{row['speedup']:.2f}x",
            axis_font,
            MUTED if idx == 0 else COLORS["raster_backward_render"],
            anchor="ms",
        )


def draw_legend(draw, box):
    x0, y0, x1, y1 = box
    rounded_card(draw, box, fill=CARD_ALT)
    title_font = load_font(24, bold=True)
    label_font = load_font(17, bold=True)
    desc_font = load_font(15)
    draw_text(draw, (x0 + 24, y0 + 18), "Legend", title_font)

    entries = [
        ("Raster Bwd Render", COLORS["raster_backward_render"], "backward render kernel body"),
        ("Raster Bwd Other", COLORS["raster_backward_other"], "backward preprocess and remaining raster CUDA"),
        ("Bwd Non-Raster", COLORS["backward_non_raster"], "autograd/loss work outside rasterizer"),
        ("Raster Fwd Render", COLORS["raster_forward_render"], "forward render kernel body"),
        ("Raster Fwd Sort", COLORS["raster_forward_sort"], "duplicate/sort-heavy forward binning cost"),
        ("Raster Fwd Other", COLORS["raster_forward_other"], "scan, zeroing, identify-ranges, copies"),
        ("Other Iter", COLORS["other_iter"], "everything else in the iteration"),
    ]

    row_h = 44
    for idx, (label, color, desc) in enumerate(entries):
        y = y0 + 66 + idx * row_h
        draw.rounded_rectangle((x0 + 24, y + 5, x0 + 46, y + 27), radius=6, fill=color)
        draw_text(draw, (x0 + 60, y), label, label_font)
        draw_text(draw, (x0 + 250, y + 2), desc, desc_font, MUTED)


def draw_insight_panel(draw, box, rows):
    x0, y0, x1, y1 = box
    rounded_card(draw, box, fill=CARD_ALT)
    title_font = load_font(24, bold=True)
    body_font = load_font(17)
    accent_font = load_font(18, bold=True)
    draw_text(draw, (x0 + 24, y0 + 18), "Takeaways", title_font)

    lines = [
        ("High ratios", f"{rows[1]['label'].replace(chr(10), ' ')} is slightly slower than baseline."),
        ("Rasterizer win", "Most of the visible drop comes from backward render and forward render blocks."),
        ("Saturation", "Residual non-raster iteration work becomes dominant as tile ratio gets smaller."),
        ("Practical zone", "0.25-0.5 starts paying off; 0.125 is the fastest in this debug sweep."),
    ]

    y = y0 + 64
    for title, text in lines:
        draw_text(draw, (x0 + 24, y), title, accent_font, COLORS["raster_backward_render"])
        wrapped = textwrap.wrap(text, width=26) or [text]
        for line_idx, line in enumerate(wrapped):
            x = x0 + 170
            draw_text(draw, (x, y + line_idx * 20), line, body_font, TEXT)
        y += max(38, 20 * len(wrapped) + 14)



def main():
    parser = argparse.ArgumentParser(description="Render a polished stacked-bar breakdown for tile ratio profiling.")
    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    rows = prepare_rows(read_rows(summary_csv))
    out_path = Path(args.output) if args.output else summary_csv.with_name("tile_ratio_breakdown_stacked.png")

    width, height = 1760, 1240
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(50, bold=True)
    subtitle_font = load_font(20)
    kicker_font = load_font(18, bold=True)
    draw_text(draw, (64, 42), "Tile-Wise Training Profiling", title_font)
    draw_text(draw, (66, 102), "Cook Spinach debug sweep, 300-500 iter average, stacked iteration breakdown by tile ratio.", subtitle_font, MUTED)
    draw_text(draw, (66, 136), "4DGS / 16x16 tile path / batch size 1 / CUDA-time view", kicker_font, COLORS["raster_backward_render"])

    draw_summary_cards(draw, (64, 184, 1696, 322), rows)
    draw_stacked_bars(draw, (64, 356, 1148, 1148), rows)
    draw_legend(draw, (1182, 356, 1696, 760))
    draw_insight_panel(draw, (1182, 786, 1696, 1148), rows)

    canvas.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
