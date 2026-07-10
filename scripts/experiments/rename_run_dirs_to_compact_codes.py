#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re

import yaml

import update_reward_weight_csv as reward_index


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "kbot_forward_flat"
MAP_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "rename_maps"

TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".yaml", ".yml"}


def _timestamp_prefix(run_name: str, timestamp: str) -> str:
    match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", run_name)
    if match:
        return match.group(1)
    if timestamp:
        return timestamp.replace(" ", "_").replace(":", "-")
    raise ValueError(f"Could not infer timestamp prefix for {run_name!r}")


def _rewrite_text_file(path: Path, replacements: list[tuple[str, str]]) -> bool:
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        return False

    new_text = text
    for old, new in replacements:
        new_text = new_text.replace(old, new)
        new_text = new_text.replace(re.escape(old), re.escape(new))

    if new_text == text:
        return False
    path.write_text(new_text)
    return True


def _iter_rewrite_files() -> list[Path]:
    roots = [
        REPO_ROOT / "logs" / "rsl_rl" / "curricula",
        LOG_ROOT,
        REPO_ROOT / "policies",
    ]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            if "git" in path.parts:
                continue
            files.append(path)
    return files


def _build_rows() -> list[dict[str, str]]:
    runs, _term_order = reward_index._collect_runs(
        LOG_ROOT,
        reward_index.DEFAULT_RUNS_CSV,
        reward_index.DEFAULT_WIDE_CSV,
    )
    rows: list[dict[str, str]] = []
    for run in sorted(runs, key=lambda row: (row["timestamp"], row["run_dir"])):
        old_name = run["run_name"]
        new_name = f"{_timestamp_prefix(old_name, run['timestamp'])}_{run['display_lineage']}"
        rows.append(
            {
                "timestamp": run["timestamp"],
                "old_name": old_name,
                "new_name": new_name,
                "old_path": run["run_dir"],
                "new_path": f"logs/rsl_rl/kbot_forward_flat/{new_name}",
                "display_lineage": run["display_lineage"],
                "lineage": run["lineage"],
                "compact_lineage": run["compact_lineage"],
                "alias": run["alias"],
                "version": run["version"],
                "segment": run["segment"],
                "from_iter": "" if run["from_iter"] is None else str(run["from_iter"]),
                "to_iter": "" if run["to_iter"] is None else str(run["to_iter"]),
                "resume": "1" if run["resume"] else "0",
                "parent_old_name": run["load_run"] if run["resume"] else "",
                "parent_checkpoint": run["load_checkpoint"] if run["resume"] else "",
                "action": "unchanged" if old_name == new_name else "rename",
            }
        )
    return rows


def _check_collisions(rows: list[dict[str, str]]) -> None:
    by_new: dict[str, list[str]] = {}
    for row in rows:
        by_new.setdefault(row["new_name"], []).append(row["old_name"])

    duplicates = {new: old for new, old in by_new.items() if len(old) > 1}
    if duplicates:
        details = "\n".join(f"{new}: {old}" for new, old in sorted(duplicates.items()))
        raise RuntimeError(f"Duplicate target run names:\n{details}")

    for row in rows:
        source = LOG_ROOT / row["old_name"]
        target = LOG_ROOT / row["new_name"]
        if source == target:
            continue
        if target.exists():
            raise RuntimeError(f"Target already exists for {row['old_name']}: {target}")


def _write_map(rows: list[dict[str, str]], map_path: Path) -> None:
    map_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "action",
        "old_name",
        "new_name",
        "old_path",
        "new_path",
        "display_lineage",
        "lineage",
        "compact_lineage",
        "alias",
        "version",
        "segment",
        "from_iter",
        "to_iter",
        "resume",
        "parent_old_name",
        "parent_checkpoint",
    ]
    with map_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rename_dirs(rows: list[dict[str, str]]) -> int:
    renamed = 0
    for row in rows:
        if row["action"] != "rename":
            continue
        source = LOG_ROOT / row["old_name"]
        target = LOG_ROOT / row["new_name"]
        source.rename(target)
        renamed += 1
    return renamed


def _write_ktr_metadata(rows: list[dict[str, str]]) -> int:
    old_to_new = {row["old_name"]: row["new_name"] for row in rows}
    changed = 0
    for row in rows:
        run_dir = LOG_ROOT / row["new_name"]
        params_dir = run_dir / "params"
        if not params_dir.exists():
            continue
        metadata_path = params_dir / reward_index.KTR_METADATA_NAME
        if metadata_path.exists():
            try:
                data = yaml.safe_load(metadata_path.read_text())
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
        else:
            data = {}

        parent = row["parent_old_name"] if row["resume"] == "1" else ""
        parent = old_to_new.get(parent, parent)
        next_data = dict(data)
        next_data["version"] = row["version"]
        next_data["run_name"] = row["new_name"].split("_", 2)[-1]
        next_data["load_run"] = parent
        next_data["load_checkpoint"] = row["parent_checkpoint"] if row["resume"] == "1" else ""
        if row["from_iter"]:
            next_data["from_iter"] = int(row["from_iter"])
        if row["to_iter"]:
            next_data["to_iter"] = int(row["to_iter"])

        if next_data != data:
            metadata_path.write_text(yaml.safe_dump(next_data, sort_keys=False))
            changed += 1
    return changed


def _rewrite_references(rows: list[dict[str, str]]) -> int:
    replacements = [
        (row["old_name"], row["new_name"])
        for row in rows
        if row["old_name"] != row["new_name"]
    ]
    if not replacements:
        return 0

    changed = 0
    for path in _iter_rewrite_files():
        if _rewrite_text_file(path, replacements):
            changed += 1
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename KBot run directories to timestamp_compactcode names.")
    parser.add_argument("--dry-run", action="store_true", help="Write the map and validate collisions, but do not rename.")
    parser.add_argument("--map", type=Path, default=None, help="Old-to-new CSV map path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    map_path = args.map or (MAP_ROOT / f"{stamp}_kbot_forward_flat_run_dir_renames.csv")

    rows = _build_rows()
    _check_collisions(rows)
    _write_map(rows, map_path)

    rename_count = sum(1 for row in rows if row["action"] == "rename")
    if args.dry_run:
        print(f"[DRY-RUN] map: {map_path}", flush=True)
        print(f"[DRY-RUN] indexed runs: {len(rows)}", flush=True)
        print(f"[DRY-RUN] directories to rename: {rename_count}", flush=True)
        return

    renamed = _rename_dirs(rows)
    metadata = _write_ktr_metadata(rows)
    rewritten = _rewrite_references(rows)
    print(f"[INFO] map: {map_path}", flush=True)
    print(f"[INFO] indexed runs: {len(rows)}", flush=True)
    print(f"[INFO] directories renamed: {renamed}", flush=True)
    print(f"[INFO] KTR metadata files updated: {metadata}", flush=True)
    print(f"[INFO] text files updated: {rewritten}", flush=True)


if __name__ == "__main__":
    main()
