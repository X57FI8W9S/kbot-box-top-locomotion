#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "kbot_forward_flat"
CURRICULA_DIR = REPO_ROOT / "logs" / "rsl_rl" / "curricula"
ENV_CFG = REPO_ROOT / "source" / "kbot_loco" / "kbot_loco" / "tasks" / "locomotion" / "env_cfg.py"
DEFAULT_PLAN = SCRIPT_DIR / "training_plan.csv"
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
KTR = SCRIPT_DIR / "kbot_train_run.py"
MILESTONES = (300, 500, 800, 1200, 2500, 5000, 10000)
V4_CLASS = "KBotForwardFlatV4Top4StarterEnvCfg"


@dataclass(frozen=True)
class Plan:
    headers: list[str]
    rows: dict[str, dict[str, str]]


@dataclass
class RunResult:
    milestone: int
    run_dir: str = ""
    checkpoint: str = ""
    graph_path: str = ""
    legend_path: str = ""
    reward_csv_path: str = ""
    video_path: str = ""
    metrics_path: str = ""
    final_x_distance_m: str = ""
    final_y_distance_m: str = ""
    final_hud_approved_step_fraction: str = ""
    final_hud_joules_per_meter: str = ""
    status: str = "complete"
    notes: str = ""


def _read_plan(path: Path) -> Plan:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not reader.fieldnames:
            raise ValueError(f"Empty training plan: {path}")
        if reader.fieldnames[0] != "reward":
            raise ValueError(f"First training plan column must be 'reward', got {reader.fieldnames[0]!r}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            reward = (row.get("reward") or "").strip()
            if not reward:
                continue
            rows[reward] = {key: (value or "").strip() for key, value in row.items()}
    return Plan(headers=list(reader.fieldnames), rows=rows)


def _available_milestones(plan: Plan) -> list[int]:
    milestones = []
    for header in plan.headers[1:]:
        if header.isdigit():
            milestones.append(int(header))
    return milestones


def _seed_iteration(seed: str, from_iter: int | None) -> int:
    if from_iter is not None:
        return from_iter
    match = re.search(r"model_(\d+)\.pt$", seed)
    if match is not None:
        return int(match.group(1))
    if seed in {"v3.2-m200", "seed", "k1"}:
        return 200
    if seed in {"latest", "current"}:
        return _latest_checkpoint_iteration()
    path = Path(seed)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.is_dir():
        checkpoints = sorted(
            path.glob("model_*.pt"),
            key=lambda candidate: _checkpoint_iteration(candidate.name) or -1,
        )
        if checkpoints:
            iteration = _checkpoint_iteration(checkpoints[-1].name)
            if iteration is not None:
                return iteration
    raise ValueError(f"Could not infer start iteration from seed {seed!r}; pass --from-iter.")


def _checkpoint_iteration(name: str) -> int | None:
    match = re.fullmatch(r"model_(\d+)\.pt", Path(name).name)
    return int(match.group(1)) if match else None


def _latest_checkpoint_iteration() -> int:
    candidates: list[tuple[float, int]] = []
    if LOG_ROOT.exists():
        for run_dir in LOG_ROOT.iterdir():
            if not run_dir.is_dir():
                continue
            for checkpoint in run_dir.glob("model_*.pt"):
                iteration = _checkpoint_iteration(checkpoint.name)
                if iteration is not None:
                    candidates.append((checkpoint.stat().st_mtime, iteration))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {LOG_ROOT}")
    return max(candidates)[1]


def _milestone_sequence(from_iter: int, to_iter: int | None, plan: Plan) -> list[int]:
    plan_milestones = _available_milestones(plan)
    if not plan_milestones:
        raise ValueError("Training plan has no numeric milestone columns.")
    target = to_iter if to_iter is not None else max(m for m in plan_milestones if m > from_iter)
    sequence = [m for m in MILESTONES if from_iter < m <= target]
    missing = [str(m) for m in sequence if str(m) not in plan.headers]
    if missing:
        raise ValueError(f"Training plan is missing milestone columns: {', '.join(missing)}")
    return sequence


def _class_weight_block(text: str) -> tuple[int, int]:
    class_match = re.search(rf"(?m)^class {re.escape(V4_CLASS)}\b", text)
    if class_match is None:
        raise ValueError(f"Could not find class {V4_CLASS} in {ENV_CFG}")
    start_match = re.search(r"(?m)^ {8}reward_weight_groups = \(", text[class_match.start() :])
    if start_match is None:
        raise ValueError(f"Could not find reward_weight_groups in {V4_CLASS}")
    start = class_match.start() + start_match.start()
    end_match = re.search(r"(?m)^ {8}reward_params = \{", text[start:])
    if end_match is None:
        raise ValueError(f"Could not find reward_params after reward_weight_groups in {V4_CLASS}")
    end = start + end_match.start()
    return start, end


def _weight_pattern(reward: str) -> re.Pattern[str]:
    return re.compile(
        rf'(?m)^(?P<prefix>\s*"{re.escape(reward)}":\s*)'
        r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"(?P<suffix>\s*,.*)$"
    )


def _same_numeric_value(left: str, right: str) -> bool:
    try:
        return float(left) == float(right)
    except ValueError:
        return left == right


def _planned_changes(plan: Plan, milestone: int, env_text: str) -> list[tuple[str, str, str]]:
    start, end = _class_weight_block(env_text)
    block = env_text[start:end]
    column = str(milestone)
    changes: list[tuple[str, str, str]] = []
    for reward, row in plan.rows.items():
        target = (row.get(column) or "").strip()
        if not target:
            continue
        match = _weight_pattern(reward).search(block)
        if match is None:
            print(f"[WARN] plan term not found in V4 weight block, skipping: {reward}", flush=True)
            continue
        old = match.group("value")
        if not _same_numeric_value(old, target):
            changes.append((reward, old, target))
    return changes


def _patch_weights(changes: Iterable[tuple[str, str, str]]) -> None:
    text = ENV_CFG.read_text()
    start, end = _class_weight_block(text)
    block = text[start:end]
    for reward, old, new in changes:
        pattern = _weight_pattern(reward)
        match = pattern.search(block)
        if match is None:
            raise ValueError(f"Could not restore/patch missing V4 reward weight: {reward}")
        current = match.group("value")
        if current != old:
            raise RuntimeError(f"Refusing to patch {reward}: expected current value {old}, found {current}")
        replacement = f"{match.group('prefix')}{new}{match.group('suffix')}"
        block = block[: match.start()] + replacement + block[match.end() :]
    ENV_CFG.write_text(text[:start] + block + text[end:])


def _print_plan(seed: str, from_iter: int, sequence: list[int], plan_path: Path, plan: Plan, curriculum_csv: Path) -> None:
    env_text = ENV_CFG.read_text()
    print(f"[INFO] seed: {seed}", flush=True)
    print(f"[INFO] range: {from_iter} -> {sequence[-1]}", flush=True)
    print(f"[INFO] plan: {plan_path}", flush=True)
    print(f"[INFO] curriculum_csv: {curriculum_csv}", flush=True)
    for milestone in sequence:
        changes = _planned_changes(plan, milestone, env_text)
        print(f"[INFO] milestone {milestone}: {len(changes)} weight changes", flush=True)
        for reward, old, new in changes:
            print(f"  - {reward}: {old} -> {new}", flush=True)


def _confirm_or_cancel(yes: bool, dry_run: bool) -> None:
    if yes or dry_run:
        return
    try:
        response = input("Press Enter to run this curriculum, or type anything else to cancel: ")
    except EOFError as exc:
        raise RuntimeError("No confirmation input received; rerun with --yes to proceed non-interactively.") from exc
    if response.strip():
        raise RuntimeError("Cancelled before starting.")


def _run_ktr(command: list[str], *, dry_run: bool) -> str:
    print("+ " + shlex.join(command), flush=True)
    if dry_run:
        return ""
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        print(line, end="", flush=True)
    return_code = process.wait()
    output = "".join(output_lines)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, output=output)
    return output


def _done_value(output: str, key: str) -> str:
    pattern = re.compile(rf"^\[DONE\] {re.escape(key)}: (.+)$", re.MULTILINE)
    match = pattern.search(output)
    return match.group(1).strip() if match else ""


def _load_metrics(result: RunResult) -> None:
    if not result.metrics_path:
        return
    path = Path(result.metrics_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
    except Exception:
        return
    for key in (
        "final_x_distance_m",
        "final_y_distance_m",
        "final_hud_approved_step_fraction",
        "final_hud_joules_per_meter",
    ):
        value = data.get(key)
        if value is not None:
            setattr(result, key, str(value))


def _append_curriculum_rows(
    path: Path,
    *,
    curriculum_timestamp: str,
    from_iter: int,
    to_iter: int,
    result: RunResult,
    changes: list[tuple[str, str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "curriculum_timestamp",
        "milestone",
        "run_dir",
        "from_iter",
        "to_iter",
        "checkpoint",
        "graph_path",
        "legend_path",
        "reward_csv_path",
        "video_path",
        "metrics_path",
        "final_x_distance_m",
        "final_y_distance_m",
        "final_hud_approved_step_fraction",
        "final_hud_joules_per_meter",
        "reward",
        "old_weight",
        "new_weight",
        "delta",
        "status",
        "notes",
    ]
    write_header = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "row_type": "run",
                "curriculum_timestamp": curriculum_timestamp,
                "milestone": result.milestone,
                "run_dir": result.run_dir,
                "from_iter": from_iter,
                "to_iter": to_iter,
                "checkpoint": result.checkpoint,
                "graph_path": result.graph_path,
                "legend_path": result.legend_path,
                "reward_csv_path": result.reward_csv_path,
                "video_path": result.video_path,
                "metrics_path": result.metrics_path,
                "final_x_distance_m": result.final_x_distance_m,
                "final_y_distance_m": result.final_y_distance_m,
                "final_hud_approved_step_fraction": result.final_hud_approved_step_fraction,
                "final_hud_joules_per_meter": result.final_hud_joules_per_meter,
                "status": result.status,
                "notes": result.notes,
            }
        )
        for reward, old, new in changes:
            try:
                delta = str(float(new) - float(old))
            except ValueError:
                delta = ""
            writer.writerow(
                {
                    "row_type": "weight_delta",
                    "curriculum_timestamp": curriculum_timestamp,
                    "milestone": result.milestone,
                    "from_iter": from_iter,
                    "to_iter": to_iter,
                    "reward": reward,
                    "old_weight": old,
                    "new_weight": new,
                    "delta": delta,
                    "status": result.status,
                    "notes": result.notes,
                }
            )


def _ktr_command(args: argparse.Namespace, seed: str, to_iter: int) -> list[str]:
    ktr_seed = "v3.2-m200" if seed in {"seed", "k1"} else seed
    command = [
        str(KTR),
        "--seed",
        ktr_seed,
        "--to-iter",
        str(to_iter),
        "--save-interval",
        str(args.save_interval),
        "--num-envs",
        str(args.num_envs),
        "--python",
        str(args.python),
        "--yes",
    ]
    if args.no_graphs:
        command.append("--no-graphs")
    if args.no_video:
        command.append("--no-video")
    if args.no_weight_csv:
        command.append("--no-weight-csv")
    if args.policy_only_resume:
        command.append("--policy-only-resume")
    if not args.headless:
        command.append("--no-headless")
    if args.video_length is not None:
        command.extend(["--video-length", str(args.video_length)])
    if args.dry_run:
        command.append("--dry-run")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KBot curriculum milestones from scripts/experiments/training_plan.csv.")
    parser.add_argument("--seed", default="v3.2-m200", help="Seed alias, checkpoint path, run directory, latest, or current. Default: v3.2-m200.")
    parser.add_argument("--from-iter", type=int, default=None, help="Start iteration. Default: infer from seed.")
    parser.add_argument("--to-iter", type=int, default=None, help="Final curriculum milestone. Default: highest milestone in the plan.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="Training plan CSV. Default: scripts/experiments/training_plan.csv.")
    parser.add_argument("--save-interval", type=int, default=25, help="Checkpoint save interval passed to KTR. Default: 25.")
    parser.add_argument("--num-envs", type=int, default=4096, help="Parallel training envs passed to KTR. Default: 4096.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON, help="Python executable passed to KTR. Default: repo .venv.")
    parser.add_argument("--video-length", type=int, default=3000, help="Trailing HUD rollout length in env steps. Default: 3000.")
    parser.add_argument("--no-graphs", action="store_true", help="Pass --no-graphs to each KTR run.")
    parser.add_argument("--no-video", action="store_true", help="Pass --no-video to each KTR run.")
    parser.add_argument("--no-weight-csv", action="store_true", help="Pass --no-weight-csv to each KTR run.")
    parser.add_argument("--policy-only-resume", action="store_true", help="Pass --policy-only-resume to each KTR run.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True, help="Run KTR headless. Default: true.")
    parser.add_argument("--curriculum-csv", type=Path, default=None, help="Append results to this curriculum CSV instead of creating a new one.")
    parser.add_argument("--yes", action="store_true", help="Run without the outer interactive confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Print the curriculum plan and KTR commands without patching or training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = _read_plan(args.plan)
    from_iter = _seed_iteration(args.seed, args.from_iter)
    sequence = _milestone_sequence(from_iter, args.to_iter, plan)
    curriculum_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    curriculum_csv = args.curriculum_csv or CURRICULA_DIR / f"{curriculum_timestamp}_ktc.csv"

    _print_plan(args.seed, from_iter, sequence, args.plan, plan, curriculum_csv)
    _confirm_or_cancel(args.yes, args.dry_run)
    if args.dry_run:
        seed = args.seed
        for milestone in sequence:
            print("+ " + shlex.join(_ktr_command(args, seed, milestone)), flush=True)
            seed = f"<previous_run_dir>/model_{milestone}.pt"
        return

    seed = args.seed
    current_iter = from_iter
    for milestone in sequence:
        env_text = ENV_CFG.read_text()
        changes = _planned_changes(plan, milestone, env_text)
        reverse_changes = [(reward, new, old) for reward, old, new in reversed(changes)]
        result = RunResult(milestone=milestone)
        try:
            _patch_weights(changes)
            output = _run_ktr(_ktr_command(args, seed, milestone), dry_run=False)
            result.run_dir = _done_value(output, "run_dir")
            result.checkpoint = _done_value(output, "checkpoint")
            result.graph_path = _done_value(output, "graph")
            result.legend_path = _done_value(output, "legend")
            result.reward_csv_path = _done_value(output, "reward_csv")
            result.video_path = _done_value(output, "video")
            result.metrics_path = _done_value(output, "metrics_json")
            _load_metrics(result)
            if not result.checkpoint:
                raise RuntimeError(f"KTR completed milestone {milestone} but did not report a checkpoint.")
        except Exception as exc:
            result.status = "failed"
            result.notes = str(exc)
            _append_curriculum_rows(
                curriculum_csv,
                curriculum_timestamp=curriculum_timestamp,
                from_iter=current_iter,
                to_iter=milestone,
                result=result,
                changes=changes,
            )
            raise
        finally:
            try:
                _patch_weights(reverse_changes)
            except Exception as restore_exc:
                print(f"[ERROR] failed to restore temporary weights after milestone {milestone}: {restore_exc}", file=sys.stderr)
                raise

        _append_curriculum_rows(
            curriculum_csv,
            curriculum_timestamp=curriculum_timestamp,
            from_iter=current_iter,
            to_iter=milestone,
            result=result,
            changes=changes,
        )
        seed = result.checkpoint
        current_iter = milestone

    print("[DONE] KTC curriculum complete.", flush=True)
    print(f"[DONE] curriculum_csv: {curriculum_csv}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
