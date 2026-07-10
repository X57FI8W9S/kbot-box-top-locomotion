#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


REWARD_PREFIX = "Episode_Reward/"
OUTPUT_DPI = 180
COMBINED_OUTPUT_DPI = 240
STACK_ALPHA = 0.94
SUBPIXEL_BAND_THRESHOLD_PX = 1.0
FIGSIZE = (16, 9)
LEGEND_COLUMNS = 8
LEGEND_FONT_SIZE = 7.6
LEGEND_ANCHOR = (0.484, 0.4416)
COMBINED_GRID_KWARGS = {
    "height_ratios": [0.81, 0.19],
    "left": 0.045,
    "right": 0.985,
    "top": 0.965,
    "bottom": 0.020,
    "hspace": 0.097,
}


def _vanimo_colormap():
    try:
        from cmcrameri import cm
    except ImportError as exc:
        raise RuntimeError(
            "Could not import cmcrameri. Install it with `.venv/bin/python -m pip install cmcrameri`."
        ) from exc
    return cm.vanimo


def _load_reward_scalars(run_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("Could not import tensorboard. Run from the repo .venv.") from exc

    event_files = sorted(run_dir.glob("events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No events.out.tfevents* files found in: {run_dir}")

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = sorted(tag for tag in accumulator.Tags().get("scalars", []) if tag.startswith(REWARD_PREFIX))
    if not tags:
        raise RuntimeError(f"No {REWARD_PREFIX} scalars found in: {run_dir}")

    steps: list[int] | None = None
    series: dict[str, np.ndarray] = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        tag_steps = [event.step for event in events]
        if steps is None:
            steps = tag_steps
        elif tag_steps != steps:
            raise RuntimeError(f"Scalar steps for {tag} do not match the base reward series")
        name = tag.removeprefix(REWARD_PREFIX)
        series[name] = np.asarray([float(event.value) for event in events], dtype=np.float64)

    if steps is None:
        raise RuntimeError(f"No reward scalar steps found in: {run_dir}")
    return np.asarray(steps, dtype=np.float64), series


def _write_csv(iterations: np.ndarray, series: dict[str, np.ndarray], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["iteration", *series.keys()]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, iteration in enumerate(iterations):
            row = {"iteration": int(iteration)}
            for name, values in series.items():
                row[name] = values[index]
            writer.writerow(row)


def _mean_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(np.abs(finite)))


def _split_terms(series: dict[str, np.ndarray], min_abs_mean: float, top_n: int | None) -> tuple[list[str], list[str]]:
    positive: list[str] = []
    negative: list[str] = []
    for name, values in series.items():
        if _mean_abs(values) < min_abs_mean:
            continue
        mean = float(np.nanmean(values))
        if mean >= 0.0:
            positive.append(name)
        else:
            negative.append(name)
    positive.sort(key=lambda name: _mean_abs(series[name]))
    negative.sort(key=lambda name: _mean_abs(series[name]))
    if top_n is not None and top_n > 0:
        positive = positive[-top_n:]
        negative = negative[-top_n:]
    return positive, negative


def _plot_stack_data(
    series: dict[str, np.ndarray],
    terms: list[str],
    *,
    positive: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not terms:
        return None, None
    if positive:
        values = [np.clip(series[name], 0.0, None) for name in terms]
    else:
        values = [np.clip(series[name], None, 0.0) for name in terms]
    stacked = np.cumsum(np.vstack(values), axis=0)
    return np.vstack(values), stacked


def _stack_total(stacked: np.ndarray | None, iterations: np.ndarray) -> np.ndarray:
    if stacked is None:
        return np.zeros_like(iterations, dtype=np.float64)
    return stacked[-1]


def _style_axes(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Weighted episode reward contribution")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)


def _export_px_per_reward_unit(
    iterations: np.ndarray,
    positive_stack: np.ndarray | None,
    negative_stack: np.ndarray | None,
    total: np.ndarray,
    title: str,
) -> float:
    """Estimate final PNG y pixels per reward unit for the current autoscale."""
    fig = plt.figure(figsize=FIGSIZE, dpi=COMBINED_OUTPUT_DPI, constrained_layout=False)
    grid = fig.add_gridspec(nrows=2, ncols=1, **COMBINED_GRID_KWARGS)
    ax = fig.add_subplot(grid[0])

    if positive_stack is not None:
        base = np.zeros_like(iterations, dtype=np.float64)
        for top in positive_stack:
            ax.fill_between(iterations, base, top)
            base = top

    if negative_stack is not None:
        base = np.zeros_like(iterations, dtype=np.float64)
        for bottom in negative_stack:
            ax.fill_between(iterations, base, bottom)
            base = bottom

    ax.plot(iterations, total, linewidth=3.0)
    ax.axhline(0.0)
    _style_axes(ax, title)
    fig.canvas.draw()
    ylim = ax.get_ylim()
    axes_height_export_px = ax.get_window_extent().height
    plt.close(fig)
    return axes_height_export_px / (ylim[1] - ylim[0])


def _mean_band_px(values: np.ndarray, *, positive: bool, px_per_unit: float) -> float:
    if positive:
        contribution = np.clip(values, 0.0, None)
    else:
        contribution = np.abs(np.clip(values, None, 0.0))
    finite = contribution[np.isfinite(contribution)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite) * px_per_unit)


def _split_vanimo_colors(
    series: dict[str, np.ndarray],
    terms: list[str],
    *,
    positive: bool,
    px_per_unit: float,
    subpixel_threshold_px: float,
) -> dict[str, object]:
    """Use one half of vanimo and flatten subpixel terms to the center color."""
    cmap = _vanimo_colormap()
    center = 0.50
    extreme = 0.98 if positive else 0.02
    center_color = cmap(center)
    visible_terms = [
        name
        for name in terms
        if _mean_band_px(series[name], positive=positive, px_per_unit=px_per_unit) > subpixel_threshold_px
    ]
    colors = {name: center_color for name in terms}
    if len(visible_terms) == 1:
        colors[visible_terms[0]] = cmap((center + extreme) * 0.5)
    elif len(visible_terms) > 1:
        ramp = cmap(np.linspace(center, extreme, len(visible_terms)))
        colors.update(dict(zip(visible_terms, ramp)))
    return colors


def _plot_stack(
    iterations: np.ndarray,
    series: dict[str, np.ndarray],
    positive_terms: list[str],
    negative_terms: list[str],
    output_path: Path,
    legend_path: Path,
    title: str,
    subpixel_threshold_px: float,
) -> None:
    positive_values, positive_stack = _plot_stack_data(series, positive_terms, positive=True)
    negative_values, negative_stack = _plot_stack_data(series, negative_terms, positive=False)
    positive_total = _stack_total(positive_stack, iterations)
    negative_total = _stack_total(negative_stack, iterations)
    total = positive_total + negative_total
    px_per_unit = _export_px_per_reward_unit(iterations, positive_stack, negative_stack, total, title)

    positive_colors = _split_vanimo_colors(
        series,
        positive_terms,
        positive=True,
        px_per_unit=px_per_unit,
        subpixel_threshold_px=subpixel_threshold_px,
    )
    negative_colors = _split_vanimo_colors(
        series,
        negative_terms,
        positive=False,
        px_per_unit=px_per_unit,
        subpixel_threshold_px=subpixel_threshold_px,
    )

    fig = plt.figure(figsize=FIGSIZE, dpi=COMBINED_OUTPUT_DPI, constrained_layout=False)
    grid = fig.add_gridspec(nrows=2, ncols=1, **COMBINED_GRID_KWARGS)
    ax = fig.add_subplot(grid[0])
    legend_ax = fig.add_subplot(grid[1])
    legend_ax.axis("off")
    legend_entries: list[tuple[str, object]] = []

    if positive_stack is not None:
        base = np.zeros_like(iterations, dtype=np.float64)
        for name, top in zip(positive_terms, positive_stack):
            color = positive_colors[name]
            ax.fill_between(iterations, base, top, color=color, alpha=STACK_ALPHA)
            legend_entries.append((name, Patch(facecolor=color, edgecolor="none", alpha=STACK_ALPHA)))
            base = top

    if negative_stack is not None:
        base = np.zeros_like(iterations, dtype=np.float64)
        for name, bottom in zip(negative_terms, negative_stack):
            color = negative_colors[name]
            ax.fill_between(iterations, base, bottom, color=color, alpha=STACK_ALPHA)
            legend_entries.append((name, Patch(facecolor=color, edgecolor="none", alpha=STACK_ALPHA)))
            base = bottom

    ax.plot(iterations, total, color="black", linewidth=3.0)
    legend_entries.append(("total", Line2D([0], [0], color="black", linewidth=3.0)))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    _style_axes(ax, title)
    ax.tick_params(axis="both", labelsize=8)
    ax.title.set_fontsize(11)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    labels = [label for label, _handle in legend_entries]
    handles = [handle for _label, handle in legend_entries]
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=LEGEND_ANCHOR,
        ncol=LEGEND_COLUMNS,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.25,
        handletextpad=0.22,
        columnspacing=0.42,
        borderaxespad=0.0,
        labelspacing=0.55,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    legend_path.parent.mkdir(parents=True, exist_ok=True)
    column_count = min(LEGEND_COLUMNS, max(1, int(np.ceil(len(labels) / 12))))
    legend_height = max(2.0, 0.34 * int(np.ceil(len(labels) / column_count)) + 0.5)
    legend_fig = plt.figure(figsize=(18, legend_height))
    legend_fig.legend(
        handles,
        labels,
        loc="center",
        ncol=column_count,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    legend_fig.savefig(legend_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(legend_fig)


def plot_reward_components(
    run_dir: Path,
    min_abs_mean: float = 0.0,
    top_n: int | None = None,
    subpixel_threshold_px: float = SUBPIXEL_BAND_THRESHOLD_PX,
) -> list[Path]:
    run_dir = run_dir.resolve()
    iterations, series = _load_reward_scalars(run_dir)
    positive_terms, negative_terms = _split_terms(series, min_abs_mean, top_n)
    run_prefix = run_dir.name.split("_v", 1)[0]

    csv_path = run_dir / "metricas" / f"{run_prefix}_reward_components.csv"
    _write_csv(iterations, series, csv_path)

    graph_dir = run_dir / "graficos"
    combined_path = graph_dir / f"{run_prefix}_combined_reward_components.png"
    legend_path = graph_dir / f"{run_prefix}_reward_components_legend.png"

    _plot_stack(
        iterations,
        series,
        positive_terms,
        negative_terms,
        combined_path,
        legend_path,
        "Episode reward components",
        subpixel_threshold_px,
    )
    return [csv_path, combined_path, legend_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot signed KBot reward components from TensorBoard event scalars.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Training run directory containing events.out.tfevents*")
    parser.add_argument(
        "--min-abs-mean",
        type=float,
        default=0.0,
        help="Hide terms whose mean absolute contribution is below this value.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Plot only the N largest positive terms and N largest penalty terms by mean absolute contribution.",
    )
    parser.add_argument(
        "--subpixel-threshold-px",
        type=float,
        default=SUBPIXEL_BAND_THRESHOLD_PX,
        help="Give terms whose average rendered band thickness is at or below this pixel threshold the center color.",
    )
    args = parser.parse_args()

    for output in plot_reward_components(
        args.run_dir,
        min_abs_mean=args.min_abs_mean,
        top_n=args.top_n,
        subpixel_threshold_px=args.subpixel_threshold_px,
    ):
        print(f"[INFO] Wrote: {output}", flush=True)


if __name__ == "__main__":
    main()
