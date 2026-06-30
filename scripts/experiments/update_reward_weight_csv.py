#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "kbot_forward_flat"
DEFAULT_WIDE_CSV = REPO_ROOT / "policies" / "all_reward_weights_wide_from_v1_to_now_20260626.csv"
DEFAULT_RUNS_CSV = REPO_ROOT / "policies" / "all_reward_weight_runs_from_v1_to_now_20260626.csv"


def _timestamp_from_name(name: str) -> str:
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", name)
    if match is None:
        return ""
    year, month, day, hour, minute, second = match.groups()
    return f"{year}-{month}-{day} {hour}:{minute}:{second}"


def _infer_lineage(run_name: str) -> str:
    lower = run_name.lower()
    if "v4_0" in lower or "v40" in lower:
        return "v4.0"
    if "v4" in lower:
        return "v4"
    if "v3_2" in lower or "v32" in lower:
        return "v3.2"
    if "v3_1" in lower or "v31" in lower:
        return "v3.1"
    if "v3" in lower:
        return "v3"
    if "v2_5" in lower or "v25" in lower:
        return "v2.5"
    if "v2" in lower:
        return "v2"
    if "v1" in lower:
        return "v1"
    return ""


def _load_existing_lineages(runs_csv: Path) -> dict[str, str]:
    if not runs_csv.exists():
        return {}
    with runs_csv.open(newline="") as file:
        return {row["run_dir"]: row.get("lineage", "") for row in csv.DictReader(file)}


def _load_existing_terms(wide_csv: Path) -> list[str]:
    if not wide_csv.exists():
        return []
    with wide_csv.open(newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        return [row[0] for row in reader if row]


def _weight_to_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{float(value):.1f}"
    if isinstance(value, float):
        return str(value)
    return str(value)


def _collect_runs(log_root: Path, runs_csv: Path, wide_csv: Path) -> tuple[list[dict], list[str]]:
    old_lineage = _load_existing_lineages(runs_csv)
    term_order = _load_existing_terms(wide_csv)
    seen_terms = set(term_order)
    runs = []

    for env_yaml in log_root.glob("*/params/env.yaml"):
        run_dir = env_yaml.parents[1]
        rel_run_dir = run_dir.relative_to(REPO_ROOT).as_posix()
        run_name = run_dir.name
        try:
            data = yaml.unsafe_load(env_yaml.read_text())
        except Exception as exc:
            print(f"[WARN] skipped {env_yaml}: {exc}", flush=True)
            continue

        rewards = data.get("rewards", {}) if isinstance(data, dict) else {}
        weights = {}
        if isinstance(rewards, dict):
            for reward_name, spec in rewards.items():
                if isinstance(spec, dict) and "weight" in spec:
                    weights[reward_name] = spec.get("weight")
                    if reward_name not in seen_terms:
                        seen_terms.add(reward_name)
                        term_order.append(reward_name)

        runs.append(
            {
                "timestamp": _timestamp_from_name(run_name),
                "lineage": old_lineage.get(rel_run_dir, _infer_lineage(run_name)),
                "run_dir": rel_run_dir,
                "run_name": run_name,
                "weights": weights,
            }
        )

    runs.sort(key=lambda row: (row["timestamp"], row["run_dir"]), reverse=True)
    return runs, term_order


def update_reward_weight_csv(log_root: Path, wide_csv: Path, runs_csv: Path) -> tuple[int, int]:
    runs, term_order = _collect_runs(log_root, runs_csv, wide_csv)
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    wide_csv.parent.mkdir(parents=True, exist_ok=True)

    with runs_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "timestamp", "lineage", "run_dir", "reward_weight_count"])
        for index, run in enumerate(runs, start=1):
            writer.writerow([f"R{index}", run["timestamp"], run["lineage"], run["run_dir"], len(run["weights"])])

    headers = ["reward"] + [f"{run['timestamp']} | {run['lineage']} | {run['run_name']}" for run in runs]
    with wide_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for term in term_order:
            writer.writerow([term] + [_weight_to_str(run["weights"].get(term)) for run in runs])

    return len(runs), len(term_order)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the KBot reward-weight history CSVs from saved params/env.yaml files.")
    parser.add_argument("--log-root", type=Path, default=LOG_ROOT, help="Run log root. Default: logs/rsl_rl/kbot_forward_flat.")
    parser.add_argument("--wide-csv", type=Path, default=DEFAULT_WIDE_CSV, help="Wide reward-weight CSV path.")
    parser.add_argument("--runs-csv", type=Path, default=DEFAULT_RUNS_CSV, help="Run-index CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs, rewards = update_reward_weight_csv(args.log_root, args.wide_csv, args.runs_csv)
    print(f"[INFO] updated reward weight run index: {args.runs_csv}", flush=True)
    print(f"[INFO] updated reward weight wide CSV: {args.wide_csv}", flush=True)
    print(f"[INFO] reward weight table: {runs} runs, {rewards} reward rows", flush=True)


if __name__ == "__main__":
    main()
