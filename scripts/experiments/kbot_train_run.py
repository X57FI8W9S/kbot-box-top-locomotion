#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import update_reward_weight_csv as reward_weight_history


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "kbot_forward_flat"
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
MILESTONES = (300, 500, 800, 1200, 2500, 5000, 10000)
REWARD_WEIGHTS_WIDE_CSV = REPO_ROOT / "policies" / "all_reward_weights_wide_from_v1_to_now_20260626.csv"
REWARD_WEIGHTS_RUNS_CSV = REPO_ROOT / "policies" / "all_reward_weight_runs_from_v1_to_now_20260626.csv"
REWARD_WEIGHTS_TABLE_PNG = REPO_ROOT / "policies" / "reward_weights_by_training_run.png"


TASK_ALIASES = {
    "v4": "Isaac-KBot-Forward-Flat-V4-Top4Starter-v0",
}
TASK_VERSIONS = {
    "v4": "v4",
    "Isaac-KBot-Forward-Flat-V4-Top4Starter-v0": "v4",
}


@dataclass(frozen=True)
class SeedSpec:
    load_run: str
    checkpoint: str
    iteration: int
    slug: str


SEED_ALIASES = {
    "v3.2-m200": SeedSpec(
        load_run="2026-06-23_15-42-59_v3_2_may31_top4_4096envs_zero_to_200_save25_20260623",
        checkpoint="model_200.pt",
        iteration=200,
        slug="v32m200",
    ),
}


def _checkpoint_iteration(checkpoint: str) -> int | None:
    match = re.fullmatch(r"model_(\d+)\.pt", Path(checkpoint).name)
    if match is None:
        return None
    return int(match.group(1))


def _highest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(
        run_dir.glob("model_*.pt"),
        key=lambda path: _checkpoint_iteration(path.name) if _checkpoint_iteration(path.name) is not None else -1,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in: {run_dir}")
    return checkpoints[-1]


def _latest_run_seed() -> SeedSpec:
    run_dirs = [path for path in LOG_ROOT.iterdir() if path.is_dir()] if LOG_ROOT.exists() else []
    candidates: list[tuple[float, Path, Path, int]] = []
    for run_dir in run_dirs:
        try:
            checkpoint = _highest_checkpoint(run_dir)
        except FileNotFoundError:
            continue
        iteration = _checkpoint_iteration(checkpoint.name)
        if iteration is None:
            continue
        candidates.append((checkpoint.stat().st_mtime, run_dir, checkpoint, iteration))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint candidates found in: {LOG_ROOT}")
    _mtime, run_dir, checkpoint, iteration = max(candidates, key=lambda item: item[0])
    return SeedSpec(run_dir.name, checkpoint.name, iteration, f"{run_dir.name}-m{iteration}")


def _seed_from_path(path_text: str) -> SeedSpec:
    path = Path(path_text)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if path.is_dir():
        checkpoint = _highest_checkpoint(path)
        run_dir = path
    else:
        checkpoint = path
        run_dir = checkpoint.parent
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    iteration = _checkpoint_iteration(checkpoint.name)
    if iteration is None:
        raise ValueError(f"Could not infer iteration from checkpoint name: {checkpoint.name}")
    return SeedSpec(run_dir.name, checkpoint.name, iteration, f"{run_dir.name}-m{iteration}")


def _resolve_seed(args: argparse.Namespace) -> SeedSpec:
    if args.load_run or args.checkpoint:
        if not args.load_run or not args.checkpoint:
            raise ValueError("--load-run and --checkpoint must be provided together")
        iteration = _checkpoint_iteration(args.checkpoint)
        if iteration is None and args.from_iter is None:
            raise ValueError("--from-iter is required when checkpoint name is not model_<iter>.pt")
        return SeedSpec(args.load_run, args.checkpoint, iteration or args.from_iter, f"{args.load_run}-{args.checkpoint}")
    if args.seed in SEED_ALIASES:
        return SEED_ALIASES[args.seed]
    if args.seed in {"latest", "current"}:
        return _latest_run_seed()
    return _seed_from_path(args.seed)


def _next_milestone(from_iter: int) -> int:
    for milestone in MILESTONES:
        if milestone > from_iter:
            return milestone
    raise ValueError(f"No default milestone after {from_iter}; pass --to-iter explicitly")


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return re.sub(r"_+", "_", slug)[:80] or "run"


def _task_version(task_alias: str, task: str) -> str:
    if task_alias in TASK_VERSIONS:
        return TASK_VERSIONS[task_alias]
    if task in TASK_VERSIONS:
        return TASK_VERSIONS[task]
    inferred = reward_weight_history._infer_version(task_alias)
    if inferred != "unknown":
        return inferred
    return reward_weight_history._infer_version(task)


def _predict_run_code(task_alias: str, task: str, seed: SeedSpec, from_iter: int, to_iter: int) -> str:
    return reward_weight_history.predict_next_display_lineage(
        load_run=seed.load_run,
        load_checkpoint=seed.checkpoint,
        from_iter=from_iter,
        to_iter=to_iter,
        version=_task_version(task_alias, task),
    )


def _temporary_run_name(
    task_alias: str,
    planned_code: str,
    from_iter: int,
    to_iter: int,
    save_interval: int,
) -> str:
    date = datetime.now().strftime("%Y%m%d")
    return (
        f"{_slug(task_alias)}_ktr_pending_{_slug(planned_code)}_"
        f"m{from_iter}_to_{to_iter}_save{save_interval}_{date}"
    )


def _run(command: list[str], *, dry_run: bool) -> None:
    print("+ " + shlex.join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _command_arg(command: list[str], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    value_index = index + 1
    if value_index >= len(command):
        return None
    return command[value_index]


def _run_timestamp(run_dir: Path) -> str | None:
    match = re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", run_dir.name)
    return match.group(0) if match else None


def _notify(title: str, message: str, *, enabled: bool) -> None:
    if not enabled:
        return
    print("\a", end="", flush=True)
    notify_send = shutil.which("notify-send")
    if notify_send is None:
        return
    env = os.environ.copy()
    try:
        subprocess.run(
            [notify_send, title, message],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _print_done_summary(
    *,
    run_dir: Path,
    to_iter: int,
    graphs: bool,
    video: bool,
    weight_csv: bool,
    video_command: list[str] | None,
    notify: bool,
) -> None:
    checkpoint = run_dir / f"model_{to_iter}.pt"
    timestamp = _run_timestamp(run_dir)
    graph_path = run_dir / "graficos" / f"{timestamp}_combined_reward_components.png" if timestamp else None
    legend_path = run_dir / "graficos" / f"{timestamp}_reward_components_legend.png" if timestamp else None
    csv_path = run_dir / "metricas" / f"{timestamp}_reward_components.csv" if timestamp else None
    video_path = Path(_command_arg(video_command, "--output")) if video_command is not None else None
    metrics_path = Path(_command_arg(video_command, "--metrics_output")) if video_command is not None else None

    print("[DONE] KTR workflow complete.", flush=True)
    print(f"[DONE] run_dir: {run_dir}", flush=True)
    print(f"[DONE] checkpoint: {checkpoint}", flush=True)
    if graphs and graph_path is not None:
        print(f"[DONE] graph: {graph_path}", flush=True)
        print(f"[DONE] legend: {legend_path}", flush=True)
        print(f"[DONE] reward_csv: {csv_path}", flush=True)
    if video and video_path is not None:
        print(f"[DONE] video: {video_path}", flush=True)
        print(f"[DONE] metrics_json: {metrics_path}", flush=True)
    if weight_csv:
        print(f"[DONE] reward_weights_csv: {REWARD_WEIGHTS_WIDE_CSV}", flush=True)
        print(f"[DONE] reward_weights_runs_csv: {REWARD_WEIGHTS_RUNS_CSV}", flush=True)
        print(f"[DONE] reward_weights_table_png: {REWARD_WEIGHTS_TABLE_PNG}", flush=True)

    message = f"model_{to_iter}.pt complete"
    if video_path is not None:
        message = f"{message}\n{video_path.name}"
    _notify("KTR complete", message, enabled=notify)


def _confirm_or_cancel(commands: list[list[str]], *, yes: bool, dry_run: bool) -> None:
    print("[INFO] Planned commands:", flush=True)
    for command in commands:
        print("+ " + shlex.join(command), flush=True)
    if dry_run or yes:
        return
    try:
        response = input("Press Enter to proceed, or type anything else to cancel: ")
    except EOFError as exc:
        raise RuntimeError("No confirmation input received; rerun with --yes to proceed non-interactively.") from exc
    if response.strip():
        raise RuntimeError("Cancelled before starting.")


def _find_run_dir(run_name: str) -> Path:
    matches = sorted(LOG_ROOT.glob(f"*_{run_name}"), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"Could not find run directory for run_name={run_name!r} in {LOG_ROOT}")
    return matches[-1]


def _build_train_command(
    python: Path,
    task: str,
    seed: SeedSpec,
    from_iter: int,
    to_iter: int,
    save_interval: int,
    run_name: str,
    num_envs: int,
    headless: bool,
    policy_only_resume: bool,
) -> list[str]:
    max_iterations = to_iter - from_iter + 1
    command = [
        str(python),
        "scripts/rsl_rl/train.py",
        "--task",
        task,
        "--num_envs",
        str(num_envs),
        "--resume",
        "--load_run",
        seed.load_run,
        "--checkpoint",
        seed.checkpoint,
        "--max_iterations",
        str(max_iterations),
        "--save_interval",
        str(save_interval),
        "--run_name",
        run_name,
    ]
    if headless:
        command.append("--headless")
    if policy_only_resume:
        command.append("--policy_only_resume")
    return command


def _build_graph_command(python: Path, run_dir: Path) -> list[str]:
    return [str(python), "scripts/diagnostics/plot_reward_components.py", "--run-dir", str(run_dir)]


def _build_weight_csv_command(python: Path) -> list[str]:
    return [str(python), "scripts/experiments/update_reward_weight_csv.py"]


def _build_weight_table_command(python: Path) -> list[str]:
    return [
        str(python),
        "scripts/diagnostics/render_reward_weight_table.py",
        "--csv",
        str(REWARD_WEIGHTS_WIDE_CSV),
        "--output",
        str(REWARD_WEIGHTS_TABLE_PNG),
        "--palette",
        "cork",
        "--scale",
        "2",
    ]


def _write_ktr_metadata(
    *,
    run_dir: Path,
    task_alias: str,
    task: str,
    run_name: str,
    seed: SeedSpec,
    from_iter: int,
    to_iter: int,
    save_interval: int,
) -> None:
    metadata_path = run_dir / "params" / reward_weight_history.KTR_METADATA_NAME
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task_alias": task_alias,
        "task": task,
        "version": _task_version(task_alias, task),
        "run_name": run_name,
        "load_run": seed.load_run,
        "load_checkpoint": seed.checkpoint,
        "from_iter": from_iter,
        "to_iter": to_iter,
        "save_interval": save_interval,
    }
    import yaml

    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))


def _replace_text(path: Path, replacements: dict[str, str]) -> None:
    if not path.exists():
        return
    text = path.read_text()
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text)


def _rename_run_dir_to_code(run_dir: Path, run_code: str, old_run_name: str) -> Path:
    timestamp = _run_timestamp(run_dir)
    if timestamp is None:
        raise RuntimeError(f"Could not infer timestamp from run directory: {run_dir}")
    final_run_dir = run_dir.parent / f"{timestamp}_{run_code}"
    if final_run_dir == run_dir:
        return run_dir
    if final_run_dir.exists():
        raise FileExistsError(f"Compact-code run directory already exists: {final_run_dir}")

    old_run_dir_text = str(run_dir)
    run_dir.rename(final_run_dir)
    replacements = {
        old_run_dir_text: str(final_run_dir),
        f"run_name: {old_run_name}": f"run_name: {run_code}",
    }
    _replace_text(final_run_dir / "params" / "agent.yaml", replacements)
    _replace_text(final_run_dir / "params" / "env.yaml", replacements)
    return final_run_dir


def _build_video_command(
    python: Path,
    task: str,
    run_dir: Path,
    to_iter: int,
    video_length: int,
    run_name: str,
    headless: bool,
) -> list[str]:
    checkpoint = run_dir / f"model_{to_iter}.pt"
    video_seconds = int(round(video_length * 0.01))
    output = run_dir / "videos" / "play" / f"trailing-hud-model_{to_iter}-{_slug(run_name)}-{video_seconds}s.mp4"
    metrics = output.with_suffix(".json")
    command = [
        str(python),
        "scripts/rsl_rl/play_trailing.py",
        "--task",
        task,
        "--checkpoint",
        str(checkpoint),
        "--video_length",
        str(video_length),
        "--exact_reset",
        "--prime_default_targets",
        "--fall_reset_height",
        "-1000",
        "--output",
        str(output),
        "--metrics_output",
        str(metrics),
    ]
    if headless:
        command.append("--headless")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standard KBot train -> reward graph -> trailing HUD video workflow. "
            "With no iteration arguments, train the default seed to the next milestone."
        )
    )
    parser.add_argument(
        "--seed",
        default="latest",
        help="Seed alias, checkpoint path, run directory, latest, or current. Default: latest.",
    )
    parser.add_argument("--load-run", default=None, help="Explicit RSL-RL load_run directory name.")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint name for --load-run.")
    parser.add_argument("--task", default="v4", help="Task alias or full task id. Default: v4.")
    parser.add_argument("--from-iter", type=int, default=None, help="Source checkpoint iteration. Default: infer from seed.")
    parser.add_argument("--to-iter", type=int, default=None, help="Target iteration. Default: next milestone after --from-iter.")
    parser.add_argument("--save-interval", type=int, default=25, help="Checkpoint save interval. Default: 25.")
    parser.add_argument("--num-envs", type=int, default=4096, help="Parallel training envs. Default: 4096.")
    parser.add_argument("--run-name", default=None, help="Override run_name passed to train.py.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON, help="Python executable. Default: repo .venv.")
    parser.add_argument("--video-length", type=int, default=3000, help="Trailing HUD rollout length in env steps. Default: 3000.")
    parser.add_argument("--no-graphs", dest="graphs", action="store_false", help="Skip reward graph generation.")
    parser.add_argument("--no-video", dest="video", action="store_false", help="Skip trailing HUD video generation.")
    parser.add_argument("--no-weight-csv", dest="weight_csv", action="store_false", help="Skip reward-weight history CSV update.")
    parser.add_argument("--skip-train", action="store_true", help="Reuse an existing run directory and only make requested artifacts.")
    parser.add_argument("--policy-only-resume", action="store_true", help="Resume policy/critic and iteration without optimizer state.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True, help="Run train/play headless. Default: true.")
    parser.add_argument("--yes", action="store_true", help="Run without the interactive confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--notify", action=argparse.BooleanOptionalAction, default=True, help="Send a terminal bell and best-effort desktop notification when done. Default: true.")
    parser.set_defaults(graphs=True, video=True, weight_csv=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = _resolve_seed(args)
    from_iter = args.from_iter if args.from_iter is not None else seed.iteration
    if from_iter != seed.iteration:
        raise ValueError(
            f"--from-iter ({from_iter}) does not match seed checkpoint iteration ({seed.iteration}). "
            "Pass a matching seed checkpoint or omit --from-iter."
        )
    to_iter = args.to_iter if args.to_iter is not None else _next_milestone(from_iter)
    if to_iter <= from_iter:
        raise ValueError(f"--to-iter must be greater than --from-iter ({from_iter}); got {to_iter}")
    if args.save_interval <= 0:
        raise ValueError("--save-interval must be positive")

    task = TASK_ALIASES.get(args.task, args.task)
    planned_run_code = args.run_name or _predict_run_code(args.task, task, seed, from_iter, to_iter)
    auto_compact_run_name = args.run_name is None and not args.skip_train
    run_name = (
        _temporary_run_name(args.task, planned_run_code, from_iter, to_iter, args.save_interval)
        if auto_compact_run_name
        else planned_run_code
    )

    print(f"[INFO] seed: {seed.load_run}/{seed.checkpoint}", flush=True)
    print(f"[INFO] task: {task}", flush=True)
    print(f"[INFO] range: {from_iter} -> {to_iter}", flush=True)
    print(f"[INFO] run_name: {run_name}", flush=True)
    if auto_compact_run_name:
        print(f"[INFO] final_run_name after training: {planned_run_code}", flush=True)

    if args.dry_run:
        run_dir = LOG_ROOT / f"<timestamp>_{planned_run_code}"
    else:
        run_dir = _find_run_dir(run_name) if args.skip_train else LOG_ROOT / f"<timestamp>_{planned_run_code}"

    commands: list[list[str]] = []
    if not args.skip_train:
        train_command = _build_train_command(
            args.python,
            task,
            seed,
            from_iter,
            to_iter,
            args.save_interval,
            run_name,
            args.num_envs,
            args.headless,
            args.policy_only_resume,
        )
        commands.append(train_command)

    if args.graphs:
        commands.append(_build_graph_command(args.python, run_dir))

    video_command: list[str] | None = None
    if args.video:
        checkpoint = run_dir / f"model_{to_iter}.pt"
        if args.skip_train and not args.dry_run and not checkpoint.exists():
            raise FileNotFoundError(f"Expected video checkpoint not found: {checkpoint}")
        artifact_run_name = planned_run_code if auto_compact_run_name else run_name
        video_command = _build_video_command(args.python, task, run_dir, to_iter, args.video_length, artifact_run_name, args.headless)
        commands.append(video_command)
    if args.weight_csv:
        commands.append(_build_weight_csv_command(args.python))
        commands.append(_build_weight_table_command(args.python))

    print(f"[INFO] run_dir: {run_dir}", flush=True)
    _confirm_or_cancel(commands, yes=args.yes, dry_run=args.dry_run)
    if args.dry_run:
        return

    if not args.skip_train:
        _run(commands[0], dry_run=args.dry_run)
        if not args.dry_run:
            run_dir = _find_run_dir(run_name)
            if auto_compact_run_name:
                actual_run_code = reward_weight_history.display_lineage_for_run_dir(run_dir)
                run_dir = _rename_run_dir_to_code(run_dir, actual_run_code, run_name)
                run_name = actual_run_code
                print(f"[INFO] renamed run_dir to compact code: {run_dir}", flush=True)
            _write_ktr_metadata(
                run_dir=run_dir,
                task_alias=args.task,
                task=task,
                run_name=run_name,
                seed=seed,
                from_iter=from_iter,
                to_iter=to_iter,
                save_interval=args.save_interval,
            )
            print(f"[INFO] resolved run_dir: {run_dir}", flush=True)

    post_train_commands: list[list[str]] = []
    if args.graphs:
        post_train_commands.append(_build_graph_command(args.python, run_dir))
    if args.video:
        checkpoint = run_dir / f"model_{to_iter}.pt"
        if not args.dry_run and not checkpoint.exists():
            raise FileNotFoundError(f"Expected video checkpoint not found: {checkpoint}")
        video_command = _build_video_command(args.python, task, run_dir, to_iter, args.video_length, run_name, args.headless)
        post_train_commands.append(video_command)
    if args.weight_csv:
        post_train_commands.append(_build_weight_csv_command(args.python))
        post_train_commands.append(_build_weight_table_command(args.python))

    for command in post_train_commands:
        _run(command, dry_run=args.dry_run)

    _print_done_summary(
        run_dir=run_dir,
        to_iter=to_iter,
        graphs=args.graphs,
        video=args.video,
        weight_csv=args.weight_csv,
        video_command=video_command,
        notify=args.notify,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        _notify("KTR failed", str(exc), enabled=True)
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
