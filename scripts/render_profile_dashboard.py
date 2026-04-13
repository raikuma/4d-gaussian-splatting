import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BG = "#F5F1E8"
CARD = "#FFFDFC"
CARD_ALT = "#F8F4EE"
TEXT = "#1D2433"
MUTED = "#5D6677"
OUTLINE = "#D8D1C5"
ACCENT_RED = "#D65A4A"
ACCENT_TEAL = "#2D8C8C"
ACCENT_GOLD = "#C68A1E"
ACCENT_BLUE = "#4568DC"
ACCENT_GREEN = "#3E8B57"
ACCENT_SAND = "#C7A975"


def load_font(size, bold=False):
    candidates = [
        ("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        "C:/Windows/Fonts/bahnschrift.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def read_named_csv(path, key_field, value_field):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row[key_field], float(row[value_field])))
    return rows


def rounded_card(draw, box, fill=CARD):
    draw.rounded_rectangle(box, radius=28, fill=fill, outline=OUTLINE, width=2)


def draw_text(draw, pos, text, font, fill=TEXT):
    draw.text(pos, text, font=font, fill=fill)


def draw_bar_list(draw, box, title, subtitle, rows, colors, value_suffix=" ms", max_rows=None):
    x0, y0, x1, y1 = box
    rounded_card(draw, box)
    title_font = load_font(28, bold=True)
    subtitle_font = load_font(17)
    label_font = load_font(19, bold=True)
    meta_font = load_font(16)
    draw_text(draw, (x0 + 28, y0 + 20), title, title_font)
    draw_text(draw, (x0 + 28, y0 + 56), subtitle, subtitle_font, MUTED)

    rows = rows[:max_rows] if max_rows else rows
    if not rows:
        return
    max_value = max(value for _, value in rows) or 1.0
    top = y0 + 96
    inner_left = x0 + 28
    inner_right = x1 - 28
    bar_left = inner_left + 180
    bar_right = inner_right - 80
    row_h = max(44, (y1 - top - 12) // len(rows))

    for idx, (label, value) in enumerate(rows):
        row_y = top + idx * row_h
        color = colors[idx % len(colors)]
        share = value / max_value
        bar_width = max(4, int((bar_right - bar_left) * share))
        draw_text(draw, (inner_left, row_y + 2), label, label_font)
        draw.rounded_rectangle((bar_left, row_y + 8, bar_right, row_y + 24), radius=8, fill="#ECE7DE")
        draw.rounded_rectangle((bar_left, row_y + 8, bar_left + bar_width, row_y + 24), radius=8, fill=color)
        draw_text(draw, (bar_right + 16, row_y), f"{value:.2f}{value_suffix}", meta_font)


def draw_summary_cards(draw, box, cards):
    x0, y0, x1, y1 = box
    gap = 18
    card_w = (x1 - x0 - gap * (len(cards) - 1)) // len(cards)
    title_font = load_font(16, bold=True)
    value_font = load_font(32, bold=True)
    subtitle_font = load_font(15)
    for idx, card in enumerate(cards):
        cx0 = x0 + idx * (card_w + gap)
        cx1 = cx0 + card_w
        rounded_card(draw, (cx0, y0, cx1, y1), fill=CARD_ALT)
        draw_text(draw, (cx0 + 20, y0 + 18), card["title"], title_font, MUTED)
        draw_text(draw, (cx0 + 20, y0 + 50), card["value"], value_font, card["color"])
        draw_text(draw, (cx0 + 20, y0 + 92), card["subtitle"], subtitle_font, MUTED)


def draw_stacked_share(draw, box, title, subtitle, segments):
    x0, y0, x1, y1 = box
    rounded_card(draw, box)
    title_font = load_font(28, bold=True)
    subtitle_font = load_font(17)
    label_font = load_font(17, bold=True)
    meta_font = load_font(16)
    draw_text(draw, (x0 + 28, y0 + 20), title, title_font)
    draw_text(draw, (x0 + 28, y0 + 56), subtitle, subtitle_font, MUTED)

    total = sum(value for _, value, _ in segments) or 1.0
    bar_x0 = x0 + 28
    bar_x1 = x1 - 28
    bar_y0 = y0 + 108
    bar_y1 = bar_y0 + 34
    cursor = bar_x0
    for label, value, color in segments:
        width = int((bar_x1 - bar_x0) * (value / total))
        draw.rounded_rectangle((cursor, bar_y0, cursor + width, bar_y1), radius=10, fill=color)
        cursor += width

    legend_y = bar_y1 + 26
    row_h = 30
    for idx, (label, value, color) in enumerate(segments):
        ly = legend_y + idx * row_h
        share = (value / total) * 100.0
        draw.rounded_rectangle((x0 + 28, ly + 4, x0 + 46, ly + 22), radius=5, fill=color)
        draw_text(draw, (x0 + 58, ly), label, label_font)
        draw_text(draw, (x1 - 190, ly), f"{value:.2f} ms", meta_font)
        draw_text(draw, (x1 - 90, ly), f"{share:.1f}%", meta_font, MUTED)


def main():
    parser = argparse.ArgumentParser(description="Render a polished profiling dashboard from summary files.")
    parser.add_argument("--profile_dir", type=str, required=True)
    parser.add_argument("--backward_summary_dir", type=str, required=True)
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    backward_dir = Path(args.backward_summary_dir)

    train_rows = read_named_csv(profile_dir / "train_section_cuda_summary.csv", "name", "cuda_ms_total")
    raster_fwd_all = read_named_csv(profile_dir / "rasterizer_forward_summary.csv", "metric", "avg_ms")
    raster_bwd_all = read_named_csv(profile_dir / "rasterizer_backward_summary.csv", "metric", "avg_ms")
    backward_summary = json.loads((backward_dir / "backward_op_summary.json").read_text(encoding="utf-8"))

    raster_fwd_rows = [(k.replace("_ms", ""), v) for k, v in raster_fwd_all if k != "total_ms"]
    raster_bwd_rows = [(k.replace("_ms", ""), v) for k, v in raster_bwd_all if k != "total_ms"]

    avg_backward_ms = backward_summary["avg_backward_ms"]
    avg_norm_backward_ms = backward_summary["avg_norm_backward_ms"]
    avg_rasterizer_backward_ms = backward_summary["avg_rasterizer_backward_ms"]
    other_backward_ms = max(0.0, avg_backward_ms - avg_rasterizer_backward_ms - avg_norm_backward_ms)

    top_cuda = backward_summary["top_cuda_ops_ms_total"]
    top_cuda_avg = {name: total / backward_summary["profile_iters"] for name, total in top_cuda}

    cards = [
        {
            "title": "Train Bottleneck",
            "value": f"{train_rows[0][1]:.2f} ms",
            "subtitle": f"{train_rows[0][0]} per iter, 300-500 avg",
            "color": ACCENT_RED,
        },
        {
            "title": "Rasterizer Backward",
            "value": f"{avg_rasterizer_backward_ms:.2f} ms",
            "subtitle": f"{backward_summary['rasterizer_share_of_backward'] * 100:.1f}% of backward-only avg",
            "color": ACCENT_TEAL,
        },
        {
            "title": "Normalize Backward",
            "value": f"{avg_norm_backward_ms:.2f} ms",
            "subtitle": f"{backward_summary['norm_share_of_backward'] * 100:.1f}% of backward-only avg",
            "color": ACCENT_GOLD,
        },
        {
            "title": "Forward Rasterizer",
            "value": f"{dict(raster_fwd_all)['total_ms']:.2f} ms",
            "subtitle": "render + sort dominate forward kernel time",
            "color": ACCENT_BLUE,
        },
    ]

    correction_lines = [
        "Corrected: quaternion normalize is not a major bottleneck.",
        f"Backward-only average puts normalize at {avg_norm_backward_ms:.2f} ms ({backward_summary['norm_share_of_backward'] * 100:.1f}%).",
        f"Rasterizer backward is still dominant at {avg_rasterizer_backward_ms:.2f} ms ({backward_summary['rasterizer_share_of_backward'] * 100:.1f}%).",
        "logging_and_report was disabled during the active profiling window.",
        "Remaining non-rasterizer backward time is spread across cat/copy/conv/zero-fill style ops.",
        f"Examples from 100-step average: aten::cat {top_cuda_avg.get('aten::cat', 0.0):.2f} ms, convolution_backward {top_cuda_avg.get('aten::convolution_backward', 0.0):.2f} ms, aten::zeros {top_cuda_avg.get('aten::zeros', 0.0):.2f} ms.",
    ]

    width, height = 1860, 1420
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(48, bold=True)
    subtitle_font = load_font(20)
    kicker_font = load_font(18, bold=True)
    draw_text(draw, (64, 40), "Cook Spinach Debug Profiling", title_font)
    draw_text(draw, (66, 98), "Updated with corrected backward operator attribution and cleaner 300-500 iter summary.", subtitle_font, MUTED)
    draw_text(draw, (66, 132), "4DGS / batch size 1 / profiling window with reporting disabled", kicker_font, ACCENT_RED)

    draw_summary_cards(draw, (64, 180, 1796, 322), cards)

    draw_bar_list(
        draw,
        (64, 360, 910, 760),
        "Train Loop Breakdown",
        "Average CUDA time per profiled iteration",
        train_rows,
        [ACCENT_RED, ACCENT_TEAL, ACCENT_GOLD, ACCENT_BLUE, ACCENT_GREEN, ACCENT_SAND],
    )
    draw_bar_list(
        draw,
        (950, 360, 1796, 760),
        "Rasterizer Forward Breakdown",
        "Internal CUDA event timing, average per call",
        raster_fwd_rows,
        [ACCENT_BLUE, ACCENT_TEAL, ACCENT_GOLD, ACCENT_GREEN, ACCENT_SAND],
        max_rows=8,
    )

    draw_stacked_share(
        draw,
        (64, 798, 910, 1128),
        "Backward-Only Split",
        "100-step checkpoint replay from chkpnt500.pth",
        [
            ("Rasterizer backward", avg_rasterizer_backward_ms, ACCENT_TEAL),
            ("Other backward ops", other_backward_ms, ACCENT_RED),
            ("Normalize backward", avg_norm_backward_ms, ACCENT_GOLD),
        ],
    )
    draw_bar_list(
        draw,
        (950, 798, 1796, 1128),
        "Rasterizer Backward Breakdown",
        "Average per call from train-time CUDA event profiling",
        raster_bwd_rows,
        [ACCENT_TEAL, ACCENT_RED],
    )

    notes_box = (64, 1148, 1796, 1380)
    rounded_card(draw, notes_box, fill=CARD_ALT)
    notes_title_font = load_font(28, bold=True)
    notes_font = load_font(18)
    draw_text(draw, (92, 1168), "Corrections and Review Notes", notes_title_font)
    y = 1208
    for line in correction_lines:
        draw_text(draw, (92, y), f"- {line}", notes_font, TEXT)
        y += 28

    out_path = profile_dir / "profiling_breakdown_dashboard_v2.png"
    canvas.save(out_path)

    summary_lines = [
        "Updated profiling summary",
        f"- Train bottleneck: {train_rows[0][0]} = {train_rows[0][1]:.4f} ms",
        f"- Rasterizer backward share: {backward_summary['rasterizer_share_of_backward'] * 100:.2f}%",
        f"- Normalize backward share: {backward_summary['norm_share_of_backward'] * 100:.2f}%",
        f"- Other backward share: {(other_backward_ms / avg_backward_ms) * 100:.2f}%",
        "- Corrected finding: normalize backward is not a main bottleneck.",
    ]
    (profile_dir / "corrected_profile_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
