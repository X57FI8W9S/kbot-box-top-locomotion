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


def _plot_stack(
    iterations: np.ndarray,
    series: dict[str, np.ndarray],
    positive_terms: list[str],
    negative_terms: list[str],
    output_path: Path,
    legend_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(18, 9), constrained_layout=True)
    legend_entries: list[tuple[str, object]] = []

    if positive_terms:
        positive_values = [np.clip(series[name], 0.0, None) for name in positive_terms]
        positive_stack = np.cumsum(np.vstack(positive_values), axis=0)
        colors = plt.cm.Greens(np.linspace(0.30, 0.95, len(positive_terms)))
        base = np.zeros_like(iterations, dtype=np.float64)
        for name, top, color in zip(positive_terms, positive_stack, colors):
            ax.fill_between(iterations, base, top, color=color, alpha=0.92)
            legend_entries.append((name, Patch(facecolor=color, edgecolor="none", alpha=0.92)))
            base = top
        positive_total = positive_stack[-1]
    else:
        positive_total = np.zeros_like(iterations, dtype=np.float64)

    if negative_terms:
        negative_values = [np.clip(series[name], None, 0.0) for name in negative_terms]
        negative_stack = np.cumsum(np.vstack(negative_values), axis=0)
        colors = plt.cm.GnBu(np.linspace(0.30, 0.95, len(negative_terms)))
        base = np.zeros_like(iterations, dtype=np.float64)
        for name, bottom, color in zip(negative_terms, negative_stack, colors):
            ax.fill_between(iterations, base, bottom, color=color, alpha=0.92)
            legend_entries.append((name, Patch(facecolor=color, edgecolor="none", alpha=0.92)))
            base = bottom
        negative_total = negative_stack[-1]
    else:
        negative_total = np.zeros_like(iterations, dtype=np.float64)

    total = positive_total + negative_total
    ax.plot(iterations, total, color="black", linewidth=3.0)
    legend_entries.append(("total", Line2D([0], [0], color="black", linewidth=3.0)))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Weighted episode reward contribution")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)
    labels = [label for label, _handle in legend_entries]
    handles = [handle for _label, handle in legend_entries]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    legend_path.parent.mkdir(parents=True, exist_ok=True)
    column_count = min(6, max(1, int(np.ceil(len(labels) / 12))))
    legend_height = max(2.0, 0.34 * int(np.ceil(len(labels) / column_count)) + 0.5)
    legend_fig = plt.figure(figsize=(18, legend_height))
    legend_fig.legend(
        handles,
        labels,
        loc="center",
        ncol=column_count,
        frameon=False,
        fontsize="small",
    )
    legend_fig.savefig(legend_path, dpi=180, bbox_inches="tight")
    plt.close(legend_fig)


def plot_reward_components(run_dir: Path, min_abs_mean: float = 0.0, top_n: int | None = None) -> list[Path]:
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
    args = parser.parse_args()

    for output in plot_reward_components(args.run_dir, min_abs_mean=args.min_abs_mean, top_n=args.top_n):
        print(f"[INFO] Wrote: {output}", flush=True)


if __name__ == "__main__":
    main()
