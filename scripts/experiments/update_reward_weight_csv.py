#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "kbot_forward_flat"
DEFAULT_WIDE_CSV = REPO_ROOT / "policies" / "all_reward_weights_wide_from_v1_to_now_20260626.csv"
DEFAULT_RUNS_CSV = REPO_ROOT / "policies" / "all_reward_weight_runs_from_v1_to_now_20260626.csv"
KTR_METADATA_NAME = "ktr_run.yaml"
MAX_DISPLAY_LINEAGE_LEN = 21
VERSION_LETTERS = {
    "v1": "B",
    "v2": "C",
    "v2.1": "D",
    "v2.2": "E",
    "v2.3": "F",
    "v2.4": "G",
    "v2.5": "H",
    "v3": "I",
    "v3.1": "J",
    "v3.2": "K",
    "v4": "L",
}
LETTER_VERSIONS = {letter: version for version, letter in VERSION_LETTERS.items()}


def _timestamp_from_name(name: str) -> str:
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", name)
    if match is None:
        return ""
    year, month, day, hour, minute, second = match.groups()
    return f"{year}-{month}-{day} {hour}:{minute}:{second}"


def _infer_version_from_lineage_code(run_name: str) -> str:
    code = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", run_name)
    letters = re.findall(r"(?:^|\.)([A-Z])\d+", code)
    if not letters:
        return ""
    return LETTER_VERSIONS.get(letters[-1], "")


def _infer_version(run_name: str) -> str:
    lower = run_name.lower()
    if "v4_0" in lower or "v40" in lower:
        return "v4"
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
    if "v2_4" in lower or "v24" in lower:
        return "v2.4"
    if "v2_3" in lower or "v23" in lower:
        return "v2.3"
    if "v2_2" in lower or "v22" in lower:
        return "v2.2"
    if "v2_1" in lower or "v21" in lower:
        return "v2.1"
    if "v2" in lower:
        return "v2"
    if "v1" in lower:
        return "v1"
    code_version = _infer_version_from_lineage_code(run_name)
    if code_version:
        return code_version
    return "unknown"


def _version_letter(version: str) -> str:
    return VERSION_LETTERS.get(version, "X")


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


def _checkpoint_iteration(checkpoint: str | None) -> int | None:
    if not checkpoint:
        return None
    match = re.fullmatch(r"model_(\d+)\.pt", Path(checkpoint).name)
    return int(match.group(1)) if match else None


def _load_agent_yaml(run_dir: Path) -> dict:
    agent_yaml = run_dir / "params" / "agent.yaml"
    if not agent_yaml.exists():
        return {}
    try:
        data = yaml.unsafe_load(agent_yaml.read_text())
    except Exception as exc:
        print(f"[WARN] skipped agent metadata {agent_yaml}: {exc}", flush=True)
        return {}
    return data if isinstance(data, dict) else {}


def _load_ktr_metadata(run_dir: Path) -> dict:
    metadata_yaml = run_dir / "params" / KTR_METADATA_NAME
    if not metadata_yaml.exists():
        return {}
    try:
        data = yaml.safe_load(metadata_yaml.read_text())
    except Exception as exc:
        print(f"[WARN] skipped KTR metadata {metadata_yaml}: {exc}", flush=True)
        return {}
    return data if isinstance(data, dict) else {}


def _format_iteration(iteration: int | None) -> str:
    if iteration is None:
        return "?"
    if iteration >= 1000 and iteration % 1000 == 0:
        return f"{iteration // 1000}k"
    return str(iteration)


def _format_segment(from_iter: int | None, to_iter: int | None) -> str:
    if from_iter is None and to_iter is None:
        return ""
    return f"{_format_iteration(from_iter)}-{_format_iteration(to_iter)}"


def _compact_lineage(lineage: str) -> str:
    def _format_repeated(pattern: list[str], count: int) -> str:
        return f"({'.'.join(pattern)}){count}"

    parts = lineage.split(".")
    if len(parts) < 3:
        return lineage

    tail = parts[1:]
    best: list[str] = [""] * (len(tail) + 1)
    best[len(tail)] = ""

    for index in range(len(tail) - 1, -1, -1):
        best_here = tail[index]
        if best[index + 1]:
            best_here = f"{best_here}.{best[index + 1]}"

        remaining = len(tail) - index
        for pattern_len in range(1, remaining // 2 + 1):
            pattern = tail[index : index + pattern_len]
            repeats = 1
            cursor = index + pattern_len
            while cursor + pattern_len <= len(tail) and tail[cursor : cursor + pattern_len] == pattern:
                repeats += 1
                cursor += pattern_len

            for count in range(2, repeats + 1):
                literal = ".".join(tail[index : index + pattern_len * count])
                compact = _format_repeated(pattern, count)
                if len(compact) >= len(literal):
                    continue
                candidate = compact
                next_index = index + pattern_len * count
                if best[next_index]:
                    candidate = f"{candidate}.{best[next_index]}"
                if len(candidate) < len(best_here):
                    best_here = candidate

        best[index] = best_here

    return f"{parts[0]}.{best[0]}" if best[0] else parts[0]


def _compact_lineage_suffix(parent_lineage: str, child_lineage: str) -> str:
    if child_lineage == parent_lineage:
        return ""
    prefix = f"{parent_lineage}."
    if not child_lineage.startswith(prefix):
        return _compact_lineage(child_lineage)
    suffix = child_lineage[len(prefix) :]
    compact = _compact_lineage(f"R.{suffix}")
    return compact[2:] if compact.startswith("R.") else compact


def _alias_label(index: int) -> str:
    labels = "zyxwvutsrqponmlkjihgfedcba"
    if index < len(labels):
        return labels[index]
    return f"z{index - len(labels) + 1}"


def _assign_display_lineages(runs: list[dict]) -> None:
    lineages: set[str] = set()
    parent_by_lineage: dict[str, str] = {}
    compact_by_lineage: dict[str, str] = {}
    first_seen: dict[str, tuple[str, str]] = {}

    for run in runs:
        lineage = run["lineage"]
        lineages.add(lineage)
        parent_by_lineage.setdefault(lineage, run.get("lineage_parent", ""))
        compact_by_lineage[lineage] = run["compact_lineage"]
        first_seen.setdefault(lineage, (run["timestamp"], run["run_dir"]))

    children: defaultdict[str, list[str]] = defaultdict(list)
    for lineage, parent in parent_by_lineage.items():
        if parent and parent != lineage and parent in lineages:
            children[parent].append(lineage)

    for child_list in children.values():
        child_list.sort(key=lambda lineage: first_seen.get(lineage, ("", "")))

    roots = [
        lineage
        for lineage in lineages
        if not parent_by_lineage.get(lineage) or parent_by_lineage[lineage] not in lineages
    ]
    roots.sort(key=lambda lineage: first_seen.get(lineage, ("", "")))

    display_by_lineage: dict[str, str] = {}
    alias_by_lineage: dict[str, str] = {}

    def display_from_base(lineage: str, base_lineage: str, base_display: str) -> str:
        if not base_lineage:
            return compact_by_lineage[lineage]
        suffix = _compact_lineage_suffix(base_lineage, lineage)
        return base_display if not suffix else f"{base_display}.{suffix}"

    def walk(lineage: str, base_lineage: str = "", base_display: str = "") -> None:
        candidate = display_from_base(lineage, base_lineage, base_display)
        child_candidates = [
            display_from_base(child, base_lineage, base_display)
            for child in children.get(lineage, [])
        ]
        needs_alias = len(candidate) > MAX_DISPLAY_LINEAGE_LEN or any(
            len(child) > MAX_DISPLAY_LINEAGE_LEN for child in child_candidates
        )
        if needs_alias:
            alias = _alias_label(len(alias_by_lineage))
            alias_by_lineage[lineage] = alias
            candidate = alias
            base_lineage = lineage
            base_display = alias

        display_by_lineage[lineage] = candidate
        for child in children.get(lineage, []):
            walk(child, base_lineage, base_display)

    for root in roots:
        walk(root)

    for run in runs:
        run["display_lineage"] = display_by_lineage.get(run["lineage"], run["compact_lineage"])
        run["alias"] = alias_by_lineage.get(run["lineage"], "")
        run["display_label"] = run["display_lineage"]


def _infer_iteration_range(agent_data: dict) -> tuple[int | None, int | None]:
    resume = bool(agent_data.get("resume", False))
    max_iterations = agent_data.get("max_iterations")
    from_iter = _checkpoint_iteration(agent_data.get("load_checkpoint")) if resume else 0
    to_iter = None
    if isinstance(max_iterations, int) and from_iter is not None:
        to_iter = from_iter + max_iterations - 1
    return from_iter, to_iter


def _assign_lineages(runs: list[dict]) -> None:
    runs_by_name = {run["run_name"]: run for run in runs}
    processed_by_name: dict[str, dict] = {}
    root_counts: defaultdict[str, int] = defaultdict(int)
    child_counts: defaultdict[str, int] = defaultdict(int)

    for run in sorted(runs, key=lambda row: (row["timestamp"], row["run_dir"])):
        letter = run["version_letter"]
        parent = None
        if run["resume"]:
            parent = processed_by_name.get(run["load_run"])

        if parent is None:
            root_counts[letter] += 1
            lineage = f"{letter}{root_counts[letter]}"
            lineage_parent = ""
        else:
            parent_lineage = parent["lineage"]
            child_counts[parent_lineage] += 1
            child_index = child_counts[parent_lineage]
            same_version = run["version_letter"] == parent["version_letter"]
            suffix = str(child_index) if same_version else f"{letter}{child_index}"
            lineage = f"{parent_lineage}.{suffix}"
            lineage_parent = parent_lineage

        run["lineage"] = lineage
        run["lineage_parent"] = lineage_parent
        run["compact_lineage"] = _compact_lineage(lineage)
        run["segment"] = _format_segment(run["from_iter"], run["to_iter"])
        processed_by_name[run["run_name"]] = run

    _assign_display_lineages(runs)


def _collect_runs(log_root: Path, runs_csv: Path, wide_csv: Path, extra_runs: list[dict] | None = None) -> tuple[list[dict], list[str]]:
    term_order = _load_existing_terms(wide_csv)
    seen_terms = set(term_order)
    runs = []

    for env_yaml in log_root.glob("*/params/env.yaml"):
        run_dir = env_yaml.parents[1]
        rel_run_dir = run_dir.relative_to(REPO_ROOT).as_posix()
        run_name = run_dir.name
        agent_data = _load_agent_yaml(run_dir)
        ktr_metadata = _load_ktr_metadata(run_dir)
        from_iter, to_iter = _infer_iteration_range(agent_data)
        if ktr_metadata:
            from_iter = ktr_metadata.get("from_iter", from_iter)
            to_iter = ktr_metadata.get("to_iter", to_iter)
        version = str(ktr_metadata.get("version") or _infer_version(run_name))
        load_run = str(ktr_metadata.get("load_run") or agent_data.get("load_run", ""))
        load_checkpoint = str(ktr_metadata.get("load_checkpoint") or agent_data.get("load_checkpoint", ""))
        resume = bool(ktr_metadata.get("load_run") or agent_data.get("resume", False))
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
                "version": version,
                "version_letter": _version_letter(version),
                "resume": resume,
                "load_run": load_run,
                "load_checkpoint": load_checkpoint,
                "from_iter": from_iter,
                "to_iter": to_iter,
                "run_dir": rel_run_dir,
                "run_name": run_name,
                "weights": weights,
            }
        )

    if extra_runs:
        runs.extend(extra_runs)

    _assign_lineages(runs)
    runs.sort(key=lambda row: (row["timestamp"], row["run_dir"]), reverse=True)
    return runs, term_order


def predict_next_display_lineage(
    *,
    load_run: str,
    load_checkpoint: str,
    from_iter: int,
    to_iter: int,
    version: str,
    log_root: Path = LOG_ROOT,
    wide_csv: Path = DEFAULT_WIDE_CSV,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    provisional_run_name = f"<pending_{timestamp}>"
    provisional_run_dir = f"logs/rsl_rl/kbot_forward_flat/{provisional_run_name}"
    provisional = {
        "timestamp": timestamp,
        "version": version,
        "version_letter": _version_letter(version),
        "resume": True,
        "load_run": load_run,
        "load_checkpoint": load_checkpoint,
        "from_iter": from_iter,
        "to_iter": to_iter,
        "run_dir": provisional_run_dir,
        "run_name": provisional_run_name,
        "weights": {},
    }
    runs, _term_order = _collect_runs(log_root, DEFAULT_RUNS_CSV, wide_csv, extra_runs=[provisional])
    for run in runs:
        if run["run_dir"] == provisional_run_dir:
            return run["display_lineage"]
    raise RuntimeError("Could not predict display lineage for provisional KTR run")


def display_lineage_for_run_dir(
    run_dir: Path,
    *,
    log_root: Path = LOG_ROOT,
    wide_csv: Path = DEFAULT_WIDE_CSV,
) -> str:
    rel_run_dir = run_dir.resolve().relative_to(REPO_ROOT).as_posix()
    runs, _term_order = _collect_runs(log_root, DEFAULT_RUNS_CSV, wide_csv)
    for run in runs:
        if run["run_dir"] == rel_run_dir:
            return run["display_lineage"]
    raise RuntimeError(f"Could not find display lineage for run directory: {run_dir}")


def update_reward_weight_csv(log_root: Path, wide_csv: Path, runs_csv: Path) -> tuple[int, int]:
    runs, term_order = _collect_runs(log_root, runs_csv, wide_csv)
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    wide_csv.parent.mkdir(parents=True, exist_ok=True)

    with runs_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "label",
                "timestamp",
                "lineage",
                "compact_lineage",
                "display_lineage",
                "alias",
                "version",
                "segment",
                "from_iter",
                "to_iter",
                "parent_run",
                "parent_checkpoint",
                "run_dir",
                "reward_weight_count",
            ]
        )
        for index, run in enumerate(runs, start=1):
            writer.writerow(
                [
                    f"R{index}",
                    run["timestamp"],
                    run["lineage"],
                    run["compact_lineage"],
                    run["display_lineage"],
                    run["alias"],
                    run["version"],
                    run["segment"],
                    run["from_iter"] if run["from_iter"] is not None else "",
                    run["to_iter"] if run["to_iter"] is not None else "",
                    run["load_run"],
                    run["load_checkpoint"],
                    run["run_dir"],
                    len(run["weights"]),
                ]
            )

    headers = ["reward"] + [f"{run['timestamp']} | {run['display_label']} | {run['run_name']}" for run in runs]
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
