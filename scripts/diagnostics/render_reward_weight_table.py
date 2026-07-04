#!/usr/bin/env python3
"""Render the wide reward-weight CSV as a color-coded PNG table."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_CSV = Path("policies/all_reward_weights_wide_from_v1_to_now_20260626.csv")
DEFAULT_OUTPUT = Path("policies/reward_weights_by_training_run.png")
DEFAULT_PALETTE = "cork"
_CRAMERI_CMAPS = {}
_CRAMERI_IMPORT_FAILED = False
_FALLBACK_PALETTES = {
    "cork": [
        (44, 25, 76),
        (40, 64, 115),
        (61, 107, 152),
        (111, 146, 179),
        (173, 193, 212),
        (230, 237, 236),
        (183, 207, 183),
        (120, 165, 120),
        (67, 129, 66),
        (25, 86, 21),
        (44, 27, 78),
    ],
    "managua": [
    (255, 207, 103),
    (224, 159, 87),
    (193, 116, 73),
    (160, 81, 62),
    (119, 51, 57),
    (87, 41, 73),
    (76, 61, 115),
    (81, 99, 162),
    (95, 137, 195),
    (111, 182, 226),
    (129, 231, 255),
    ],
}


@dataclass
class RunColumn:
    header: str
    timestamp: datetime | None
    time_label: str
    code_label: str


@dataclass
class RewardRow:
    reward: str
    values: list[float | None]
    mean: float | None
    scale: float
    last_change: datetime | None


def _parse_run_column(header: str) -> RunColumn:
    parts = [part.strip() for part in header.split("|")]
    date_text = parts[0] if parts else ""
    code_label = parts[1] if len(parts) > 1 else ""
    timestamp = None
    try:
        timestamp = datetime.strptime(date_text, "%Y-%m-%d %H:%M:%S")
        time_label = timestamp.strftime("%m-%d %H:%M")
    except ValueError:
        time_label = date_text[:16] or header[:16]
    return RunColumn(header=header, timestamp=timestamp, time_label=time_label, code_label=code_label)


def _parse_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _values_differ(left: float | None, right: float | None) -> bool:
    if left is None and right is None:
        return False
    if left is None or right is None:
        return True
    return not math.isclose(left, right, rel_tol=1.0e-9, abs_tol=1.0e-12)


def _last_change(values: list[float | None], columns: list[RunColumn]) -> datetime | None:
    for idx in range(0, len(values) - 1):
        if _values_differ(values[idx], values[idx + 1]):
            return columns[idx].timestamp
    return None


def _fmt_value(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value) < 1.0e-5 and value != 0.0:
        return f"{value:.0e}"
    if abs(value - round(value)) < 1.0e-9:
        return str(int(round(value)))
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    if abs(value) >= 10.0:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.3g}"


def _short_reward_name(name: str, max_chars: int = 36) -> str:
    if len(name) <= max_chars:
        return name
    return name[: max_chars - 1] + "..."


def _sanitize_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def _load_table(csv_path: Path) -> tuple[list[RunColumn], list[RewardRow]]:
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        columns = [_parse_run_column(item) for item in header[1:]]
        rows: list[RewardRow] = []
        for raw in reader:
            if not raw:
                continue
            reward = raw[0].strip()
            values = [_parse_float(item) for item in raw[1:]]
            values.extend([None] * (len(columns) - len(values)))
            values = values[: len(columns)]
            numeric = [value for value in values if value is not None]
            mean = sum(numeric) / len(numeric) if numeric else None
            if mean is None:
                scale = 1.0
            else:
                scale = max([abs(value - mean) for value in numeric] or [1.0])
                scale = scale if scale > 1.0e-12 else 1.0
            rows.append(
                RewardRow(
                    reward=reward,
                    values=values,
                    mean=mean,
                    scale=scale,
                    last_change=_last_change(values, columns),
                )
            )
    rows.sort(key=lambda row: row.last_change or datetime.min, reverse=True)
    return columns, rows


def _blend(base: tuple[int, int, int], target: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(round(base[i] + (target[i] - base[i]) * amount) for i in range(3))


def _fallback_palette(palette: str, t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    colors = _FALLBACK_PALETTES.get(palette, _FALLBACK_PALETTES[DEFAULT_PALETTE])
    scaled = t * (len(colors) - 1)
    left = int(math.floor(scaled))
    right = min(left + 1, len(colors) - 1)
    return _blend(colors[left], colors[right], scaled - left)


def _crameri_color(palette: str, t: float) -> tuple[int, int, int]:
    global _CRAMERI_IMPORT_FAILED

    t = max(0.0, min(1.0, t))
    if palette not in _CRAMERI_CMAPS and not _CRAMERI_IMPORT_FAILED:
        try:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/kbot-rl-loco3-matplotlib")
            from cmcrameri import cm

            _CRAMERI_CMAPS[palette] = getattr(cm, palette)
        except ModuleNotFoundError:
            _CRAMERI_IMPORT_FAILED = True
        except AttributeError:
            pass

    cmap = _CRAMERI_CMAPS.get(palette)
    if cmap is None:
        return _fallback_palette(palette, t)

    rgba = cmap(float(t))
    return tuple(round(float(channel) * 255) for channel in rgba[:3])


def _cell_color(value: float | None, mean: float | None, scale: float, palette: str) -> tuple[int, int, int]:
    if value is None or mean is None:
        return (250, 250, 250)
    delta = (value - mean) / scale
    return _crameri_color(palette, 0.5 + 0.5 * max(-1.0, min(1.0, delta)))


def _text_color(background: tuple[int, int, int]) -> tuple[int, int, int]:
    luminance = 0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2]
    return (245, 245, 245) if luminance < 112 else (20, 20, 20)


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _draw_vertical_text(
    image: Image.Image,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(image)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    label = Image.new("RGBA", (text_w + 4, text_h + 4), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((2 - text_bbox[0], 2 - text_bbox[1]), text, fill=fill + (255,), font=font)
    rotated = label.rotate(90, expand=True)
    x1, y1, x2, y2 = box
    paste_x = x1 + max(0, (x2 - x1 - rotated.width) // 2)
    paste_y = y2 - rotated.height - 4
    image.paste(rotated, (paste_x, paste_y), rotated)


def render_table(csv_path: Path, output_path: Path, palette: str = DEFAULT_PALETTE, render_scale: int = 2) -> None:
    columns, rows = _load_table(csv_path)

    scale_px = max(1, int(render_scale))
    reward_w = 180 * scale_px
    cell_w = 34 * scale_px
    row_h = 18 * scale_px
    header_h = 92 * scale_px
    title_h = 36 * scale_px
    margin = 12 * scale_px

    width = margin * 2 + reward_w + cell_w * len(columns)
    height = margin * 2 + title_h + header_h + row_h * len(rows)

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    small_font = _load_font(7 * scale_px)
    header_font = _load_font(8 * scale_px, bold=True)
    title_font = _load_font(15 * scale_px, bold=True)

    x0 = margin
    y0 = margin
    title = "Reward weights by each training run"
    draw.text((x0, y0), title, fill=(20, 20, 20), font=title_font)

    header_y = y0 + title_h
    data_y = header_y + header_h
    grid = (218, 218, 218)
    header_bg = (235, 235, 235)
    fixed_bg = (246, 246, 246)

    fixed_headers = [("reward", reward_w)]
    x = x0
    for text, width_px in fixed_headers:
        draw.rectangle((x, header_y, x + width_px, data_y), fill=header_bg, outline=grid)
        draw.text((x + 4 * scale_px, header_y + 14 * scale_px), text, fill=(50, 50, 50), font=header_font)
        x += width_px

    for column in columns:
        draw.rectangle((x, header_y, x + cell_w, data_y), fill=header_bg, outline=grid)
        center_x = x + cell_w // 2
        line_w = 10 * scale_px
        gap = 1 * scale_px
        _draw_vertical_text(
            image,
            column.time_label,
            (center_x - line_w - gap, header_y, center_x - gap, data_y),
            small_font,
            (50, 50, 50),
        )
        _draw_vertical_text(
            image,
            column.code_label,
            (center_x + gap, header_y, center_x + line_w + gap, data_y),
            small_font,
            (50, 50, 50),
        )
        x += cell_w

    for row_index, row in enumerate(rows, start=1):
        y = data_y + (row_index - 1) * row_h
        row_bg = fixed_bg if row_index % 2 else (252, 252, 252)
        x = x0
        draw.rectangle((x, y, x + reward_w, y + row_h), fill=row_bg, outline=grid)
        draw.text((x + 4 * scale_px, y + 4 * scale_px), _short_reward_name(row.reward), fill=(25, 25, 25), font=small_font)
        x += reward_w

        for value in row.values:
            fill = _cell_color(value, row.mean, row.scale, palette)
            draw.rectangle((x, y, x + cell_w, y + row_h), fill=fill, outline=(232, 232, 232))
            text = _fmt_value(value)
            if text:
                bbox = draw.textbbox((0, 0), text, font=small_font)
                text_w = bbox[2] - bbox[0]
                draw.text(
                    (x + max(2 * scale_px, (cell_w - text_w) // 2), y + 4 * scale_px),
                    text,
                    fill=_text_color(fill),
                    font=small_font,
                )
            x += cell_w

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"Input wide CSV. Default: {DEFAULT_CSV}")
    parser.add_argument("--output", type=Path, default=None, help=f"Output PNG path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--palette", default=DEFAULT_PALETTE, help=f"Crameri palette name. Default: {DEFAULT_PALETTE}")
    parser.add_argument("--scale", type=int, default=2, help="Integer render scale. Default: 2")
    args = parser.parse_args()

    csv_path = args.csv
    output_path = args.output
    palette = _sanitize_stem(args.palette)
    if output_path is None:
        output_path = DEFAULT_OUTPUT

    render_table(csv_path, output_path, palette=palette, render_scale=args.scale)
    print(output_path)


if __name__ == "__main__":
    main()
