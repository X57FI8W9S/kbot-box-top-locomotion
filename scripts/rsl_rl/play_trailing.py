#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import json
import os
import sys
import time
import importlib.metadata as metadata
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = REPO_ROOT / "isaac_lab" / "IsaacLab"
ISAAC_RSL_RL_DIR = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"

for path in (
    REPO_ROOT / "source" / "kbot_loco",
    ISAACLAB_ROOT / "source" / "isaaclab",
    ISAACLAB_ROOT / "source" / "isaaclab_assets",
    ISAACLAB_ROOT / "source" / "isaaclab_rl",
    ISAACLAB_ROOT / "source" / "isaaclab_tasks",
    ISAAC_RSL_RL_DIR,
):
    sys.path.insert(0, str(path))

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402


OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
BOOTSTRAP_APPROVED_STEP_ADVANCE_M = 0.008
BOOTSTRAP_APPROVED_STEP_DURATION_S = 0.07

parser = argparse.ArgumentParser(description="Play an RSL-RL checkpoint with a trailing camera and speed HUD.")
parser.add_argument("--video_length", type=int, default=1500, help="Length of the recorded video in steps.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--camera_distance", type=float, default=2.18, help="Trailing camera distance behind robot in meters.")
parser.add_argument("--camera_height", type=float, default=-0.32, help="Camera height relative to root in meters.")
parser.add_argument("--target_distance", type=float, default=0.35, help="Look-at distance ahead of root in meters.")
parser.add_argument("--target_height", type=float, default=-0.32, help="Look-at height relative to root in meters.")
parser.add_argument("--camera_window_s", type=float, default=3.0, help="Rolling-average camera direction window in seconds.")
parser.add_argument("--camera_cycle_window", type=int, default=5, help="Rolling-average camera direction window in full gait cycles.")
parser.add_argument(
    "--camera_window_mode",
    choices=("cycles", "time"),
    default="cycles",
    help="Use last full gait cycles or a fixed seconds window for camera direction smoothing.",
)
parser.add_argument("--hud_window_s", type=float, default=3.0, help="Fallback HUD rolling-average window in seconds until full gait cycles are available.")
parser.add_argument("--width", type=int, default=OUTPUT_WIDTH, help="Output video width. HUD layout is authored for 1280.")
parser.add_argument("--height", type=int, default=OUTPUT_HEIGHT, help="Output video height. HUD layout is authored for 720.")
parser.add_argument("--output", type=str, default=None, help="Output mp4 path.")
parser.add_argument(
    "--side_output",
    type=str,
    default=None,
    help="Deprecated. The side view is now composed into the main output video.",
)
parser.add_argument("--metrics_output", type=str, default=None, help="Optional JSON path for rollout metric summary.")
parser.add_argument(
    "--fall_reset_height",
    type=float,
    default=0.0,
    help="Reset the rollout only if root height falls to or below this value. Set below -100 to disable.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import kbot_loco  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.sensors import ContactSensor  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from rsl_rl_compat import rsl_rl_train_cfg  # noqa: E402

installed_version = version.parse(metadata.version("rsl-rl-lib"))


def _draw_fixed_value(
    frame: np.ndarray,
    label: str,
    value: float,
    unit: str,
    y: int,
    *,
    label_x: int = 42,
    sign_x: int = 156,
    value_x: int = 178,
    unit_x: int = 264,
    scale: float = 0.58,
    thickness: int = 1,
    color: tuple[int, int, int] = (210, 230, 255),
) -> None:
    sign = "-" if value < 0.0 else ""
    value_abs = abs(value)
    cv2.putText(frame, label, (label_x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    cv2.putText(frame, sign, (sign_x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    cv2.putText(frame, f"{value_abs:0.2f}", (value_x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    cv2.putText(frame, unit, (unit_x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _put_fixed(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    width: int,
    scale: float = 0.38,
    color: tuple[int, int, int] = (210, 230, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(frame, f"{text:<{width}}"[:width], origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _format_float(value: float, width: int, precision: int) -> str:
    if not np.isfinite(value):
        return f"{'--':>{width}}"
    return f"{value:{width}.{precision}f}"[-width:]


def _nanmean_window(values: deque[float]) -> float:
    if not values:
        return 0.0
    array = np.asarray(tuple(values), dtype=np.float32)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else 0.0


def _short_joint_name(name: str) -> str:
    side = "L" if name.startswith("left_") else "R" if name.startswith("right_") else ""
    if "hip_pitch" in name:
        return f"{side} hipP"
    if "hip_roll" in name:
        return f"{side} hipR"
    if "hip_yaw" in name:
        return f"{side} hipY"
    if "knee" in name:
        return f"{side} knee"
    if "ankle" in name:
        return f"{side} ankle"
    return name[:8]


def _joint_pair_key(name: str) -> tuple[int, str]:
    if "hip_pitch" in name:
        return (0, "hipP")
    if "hip_roll" in name:
        return (1, "hipR")
    if "hip_yaw" in name:
        return (2, "hipY")
    if "knee" in name:
        return (3, "knee")
    if "ankle" in name:
        return (4, "ankle")
    return (99, name.replace("left_", "").replace("right_", ""))


def _paired_joint_rows(joint_names: list[str]) -> list[tuple[int | None, int | None]]:
    pairs: dict[str, dict[str, int]] = {}
    order: dict[str, int] = {}
    for index, name in enumerate(joint_names):
        sort_index, label = _joint_pair_key(name)
        side = "left" if name.startswith("left_") else "right" if name.startswith("right_") else "other"
        pairs.setdefault(label, {})[side] = index
        order[label] = min(order.get(label, sort_index), sort_index)
    return [
        (pairs[label].get("left"), pairs[label].get("right", pairs[label].get("other")))
        for label in sorted(pairs, key=lambda item: (order[item], item))
    ]


def _draw_joint_table(
    frame: np.ndarray,
    joint_names: list[str],
    joint_pos: np.ndarray,
    torso_mean: float,
    torso_rms: float,
    hip_rms: float,
    window_s: float,
) -> None:
    height, width = frame.shape[:2]
    panel_x0 = max(420, width - 390)
    panel_y0 = 22
    panel_x1 = width - 22
    panel_y1 = 314

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x0, panel_y0), (panel_x1, panel_y1), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.48, frame, 0.52, 0)

    text_x = panel_x0 + 18
    cv2.putText(frame, "joint pos avg", (text_x, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
    cv2.putText(frame, f"{window_s:0.1f}s", (panel_x1 - 62, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 210, 235), 1)
    cv2.putText(frame, "torso rms", (text_x, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 230, 255), 1)
    cv2.putText(frame, f"{torso_rms: 0.3f}", (text_x + 108, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 230, 255), 1)
    cv2.putText(frame, "hip ry rms", (text_x + 190, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 230, 255), 1)
    cv2.putText(frame, f"{hip_rms: 0.3f}", (text_x + 300, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 230, 255), 1)
    cv2.putText(frame, "torso avg", (text_x, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 230, 255), 1)
    cv2.putText(frame, f"{torso_mean: 0.3f}", (text_x + 108, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    col_a_x = text_x
    col_a_val_x = text_x + 82
    col_b_x = text_x + 184
    col_b_val_x = text_x + 278
    row_y = 140
    row_step = 28
    for left_i, right_i in _paired_joint_rows(joint_names):
        if left_i is not None:
            cv2.putText(frame, _short_joint_name(joint_names[left_i]), (col_a_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 230, 255), 1)
            cv2.putText(frame, f"{joint_pos[left_i]: 0.2f}", (col_a_val_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
        if right_i is not None:
            cv2.putText(frame, _short_joint_name(joint_names[right_i]), (col_b_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 230, 255), 1)
            cv2.putText(frame, f"{joint_pos[right_i]: 0.2f}", (col_b_val_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
        row_y += row_step


def _draw_hud(
    frame: np.ndarray,
    speed: float,
    command_speed: float,
    yaw_rate: float,
    root_height: float,
    x_distance: float,
    y_distance: float,
    joint_names: list[str],
    joint_pos: np.ndarray,
    torso_mean: float,
    torso_rms: float,
    hip_rms: float,
    window_label: str,
    step_stats: dict[str, dict[str, float]],
    support_stats: dict[str, float],
    fsep_m: float,
    ksep_m: float,
    joules_per_meter: float,
    approved_step_fraction: float,
    left_swing_clearance_m: float,
    right_swing_clearance_m: float,
) -> np.ndarray:
    frame = np.ascontiguousarray(frame)
    height, width = frame.shape[:2]
    overlay = frame.copy()
    panel_x0 = 18
    panel_y0 = 18
    panel_x1 = width - 18
    panel_y1 = 122
    cv2.rectangle(overlay, (panel_x0, panel_y0), (panel_x1, panel_y1), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.48, frame, 0.52, 0)

    x = panel_x0 + 18
    top_y = 50
    cv2.putText(frame, "speed", (x, top_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 230, 255), 1)
    cv2.putText(frame, f"{speed:5.2f} m/s", (x + 72, top_y), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
    col0 = x
    col1 = x + 114
    col2 = x + 238
    col3 = x + 354
    col4 = x + 469
    col5 = x + 574
    _put_fixed(frame, f"hgt {_format_float(root_height, 5, 2)}", (col2, top_y), width=10, scale=0.36)
    _put_fixed(frame, f"x {_format_float(x_distance, 6, 2)}", (col3, top_y), width=9, scale=0.36)
    _put_fixed(frame, f"y {_format_float(y_distance, 6, 2)}", (col4, top_y), width=9, scale=0.36)
    _put_fixed(frame, f"apv {_format_float(100.0 * approved_step_fraction, 5, 1)}%", (col5, top_y), width=10, scale=0.34)
    _put_fixed(frame, f"cmd {_format_float(command_speed, 5, 2)}", (col0, 74), width=10, scale=0.40)
    _put_fixed(frame, f"yaw {_format_float(yaw_rate, 6, 2)}", (col1, 74), width=11, scale=0.40)
    _put_fixed(frame, window_label, (col5, 74), width=10, scale=0.36, color=(190, 210, 235))
    _put_fixed(frame, f"torR {_format_float(torso_rms, 6, 3)}", (col0, 96), width=13, scale=0.36)
    _put_fixed(frame, f"torA {_format_float(torso_mean, 6, 3)}", (col1, 96), width=13, scale=0.36)
    _put_fixed(frame, f"hipR {_format_float(hip_rms, 6, 3)}", (col2, 96), width=13, scale=0.36)
    _put_fixed(frame, f"fsep {_format_float(fsep_m, 5, 2)}", (col3, 96), width=11, scale=0.36)
    _put_fixed(frame, f"ksep {_format_float(ksep_m, 5, 2)}", (col4, 96), width=11, scale=0.36)
    _put_fixed(frame, f"J/m {_format_float(joules_per_meter, 6, 1)}", (col5, 96), width=11, scale=0.34)

    # Fixed 1280x720 HUD slots. Keep numbers fixed-width so signs and new digits do not
    # move neighboring indicators; columns are placed by decimal-cell starts, not text length.
    left_x = 735
    left_val_x = 797
    right_x = 855
    right_val_x = 927
    support_x = 985
    support_l_x = 1036
    support_r_x = 1088
    gait_label_x = 1112
    gait_t_x = 1150
    gait_m_x = 1188
    gait_hz_x = 1227
    cv2.putText(frame, "gait", (gait_label_x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 210, 235), 1)
    cv2.putText(frame, "t", (gait_t_x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 210, 235), 1)
    cv2.putText(frame, "m", (gait_m_x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 210, 235), 1)
    cv2.putText(frame, "Hz", (gait_hz_x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 210, 235), 1)
    for label, row_y in (("L", 58), ("R", 74), ("C", 90)):
        row = step_stats[label]
        color = (230, 240, 255)
        cv2.putText(frame, label, (gait_label_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1)
        _put_fixed(frame, _format_float(row["time_s"], 4, 2), (gait_t_x, row_y), width=4, scale=0.34, color=color)
        _put_fixed(frame, _format_float(row["length_m"], 5, 3), (gait_m_x - 6, row_y), width=5, scale=0.32, color=color)
        _put_fixed(frame, _format_float(row["rate_hz"], 5, 2), (gait_hz_x, row_y), width=5, scale=0.34, color=color)

    cv2.putText(frame, "spt%", (support_x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (190, 210, 235), 1)
    cv2.putText(frame, "L", (support_l_x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (190, 210, 235), 1)
    cv2.putText(frame, "R", (support_r_x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (190, 210, 235), 1)
    for label, key_l, key_r, row_y in (
        ("gnd", "left_stance", "right_stance", 58),
        ("sgl", "left_single", "right_single", 74),
        ("dbl", "double", "airborne", 90),
    ):
        cv2.putText(frame, label, (support_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (230, 240, 255), 1)
        _put_fixed(frame, f"{100.0 * support_stats[key_l]:3.0f}", (support_l_x - 10, row_y), width=3, scale=0.32)
        _put_fixed(frame, f"{100.0 * support_stats[key_r]:3.0f}", (support_r_x - 10, row_y), width=3, scale=0.32)
    cv2.putText(frame, "air", (support_r_x - 42, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (230, 240, 255), 1)
    cv2.putText(frame, "clr", (support_x, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (230, 240, 255), 1)
    _put_fixed(frame, f"{1000.0 * left_swing_clearance_m:3.0f}", (support_l_x - 10, 106), width=3, scale=0.32)
    _put_fixed(frame, f"{1000.0 * right_swing_clearance_m:3.0f}", (support_r_x - 10, 106), width=3, scale=0.32)
    cv2.putText(frame, "mm", (support_r_x - 42, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (230, 240, 255), 1)
    row_y = 42
    row_step = 15
    for left_i, right_i in _paired_joint_rows(joint_names):
        if left_i is not None:
            cv2.putText(frame, _short_joint_name(joint_names[left_i]), (left_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 240, 255), 1)
            _put_fixed(frame, _format_float(float(joint_pos[left_i]), 5, 2), (left_val_x, row_y), width=5, scale=0.36)
        if right_i is not None:
            cv2.putText(frame, _short_joint_name(joint_names[right_i]), (right_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 240, 255), 1)
            _put_fixed(frame, _format_float(float(joint_pos[right_i]), 5, 2), (right_val_x, row_y), width=5, scale=0.36)
        row_y += row_step
    return frame


def _root_forward_xy(robot) -> torch.Tensor:
    root_pos = robot.data.root_pos_w[0]
    root_quat = robot.data.root_quat_w[0:1]
    forward_b = torch.tensor([[1.0, 0.0, 0.0]], device=root_pos.device)
    forward_w = quat_apply(root_quat, forward_b)[0]
    return forward_w[:2] / torch.clamp(torch.linalg.norm(forward_w[:2]), min=1.0e-6)


def _smooth_forward_xy(forward_window: deque[torch.Tensor], forward_xy: torch.Tensor, maxlen: int) -> torch.Tensor:
    forward_window.append(forward_xy.detach().clone())
    _trim_deque(forward_window, maxlen)
    forward_stack = torch.stack(tuple(forward_window), dim=0)
    smoothed = torch.mean(forward_stack, dim=0)
    return smoothed / torch.clamp(torch.linalg.norm(smoothed), min=1.0e-6)


def _trim_deque(values: deque, maxlen: int) -> None:
    while len(values) > maxlen:
        values.popleft()


def _contact_body_ids(contact_sensor: ContactSensor, names: tuple[str, ...]) -> list[int]:
    body_names = getattr(contact_sensor, "body_names", None)
    if body_names is None:
        body_names = getattr(contact_sensor.data, "body_names", None)
    if body_names is not None:
        return [list(body_names).index(name) for name in names]
    return list(range(len(names)))


def _camera_cycle_window_steps(
    left_touchdown_frames: deque[int],
    right_touchdown_frames: deque[int],
    current_frame: int,
    cycle_window: int,
    fallback_steps: int,
) -> int:
    start_frames = []
    if len(left_touchdown_frames) >= cycle_window + 1:
        start_frames.append(left_touchdown_frames[-(cycle_window + 1)])
    if len(right_touchdown_frames) >= cycle_window + 1:
        start_frames.append(right_touchdown_frames[-(cycle_window + 1)])
    if not start_frames:
        return fallback_steps
    return max(1, current_frame - min(start_frames) + 1)


def _cycle_window_steps(
    left_touchdown_frames: deque[int],
    right_touchdown_frames: deque[int],
    current_frame: int,
    cycle_window: int,
    fallback_steps: int,
) -> tuple[int, str]:
    steps = _camera_cycle_window_steps(left_touchdown_frames, right_touchdown_frames, current_frame, cycle_window, fallback_steps)
    if len(left_touchdown_frames) >= cycle_window + 1 or len(right_touchdown_frames) >= cycle_window + 1:
        return steps, f"{cycle_window:d}cy avg"
    return steps, "warm avg"


def _mean_stats(rows: list[tuple[float, float]]) -> dict[str, float]:
    if not rows:
        return {"time_s": float("nan"), "length_m": float("nan"), "rate_hz": float("nan")}
    times = np.asarray([row[0] for row in rows], dtype=np.float64)
    lengths = np.asarray([row[1] for row in rows], dtype=np.float64)
    mean_time = float(np.mean(times))
    return {
        "time_s": mean_time,
        "length_m": float(np.mean(lengths)),
        "rate_hz": 1.0 / mean_time if mean_time > 1.0e-6 else float("nan"),
    }


def _recent_step_stats(events: deque[tuple[int, str, float]], cycle_window: int, dt: float) -> dict[str, dict[str, float]]:
    left_steps: list[tuple[float, float]] = []
    right_steps: list[tuple[float, float]] = []
    full_cycles: list[tuple[float, float]] = []
    approved_steps: list[bool] = []
    previous_by_side: dict[str, tuple[int, float]] = {}
    previous_event: tuple[int, str, float] | None = None
    for frame, side, root_x in events:
        if previous_event is not None and previous_event[1] != side:
            step = ((frame - previous_event[0]) * dt, root_x - previous_event[2])
            approved_steps.append(
                step[0] >= BOOTSTRAP_APPROVED_STEP_DURATION_S
                and step[1] >= BOOTSTRAP_APPROVED_STEP_ADVANCE_M
            )
            if side == "L":
                left_steps.append(step)
            else:
                right_steps.append(step)
        if side in previous_by_side:
            previous_frame, previous_x = previous_by_side[side]
            full_cycles.append(((frame - previous_frame) * dt, root_x - previous_x))
        previous_by_side[side] = (frame, root_x)
        previous_event = (frame, side, root_x)
    recent_approved_steps = approved_steps[-2 * cycle_window :]
    approved_fraction = float(np.mean(recent_approved_steps)) if recent_approved_steps else 0.0
    return {
        "L": _mean_stats(left_steps[-cycle_window:]),
        "R": _mean_stats(right_steps[-cycle_window:]),
        "C": _mean_stats(full_cycles[-cycle_window:]),
        "approved_fraction": approved_fraction,
    }


def _set_trailing_camera(
    base_env,
    robot,
    forward_xy: torch.Tensor,
    distance: float,
    height: float,
    target_distance: float,
    target_height: float,
) -> None:
    root_pos = robot.data.root_pos_w[0]
    eye = root_pos.clone()
    eye[0] -= forward_xy[0] * distance
    eye[1] -= forward_xy[1] * distance
    eye[2] += height

    target = root_pos.clone()
    target[0] += forward_xy[0] * target_distance
    target[1] += forward_xy[1] * target_distance
    target[2] += target_height
    base_env.sim.set_camera_view(tuple(float(v) for v in eye), tuple(float(v) for v in target))


def _set_side_camera(
    base_env,
    robot,
    forward_xy: torch.Tensor,
    distance: float,
    height: float,
    target_distance: float,
    target_height: float,
) -> None:
    root_pos = robot.data.root_pos_w[0]
    side_xy = torch.stack((forward_xy[1], -forward_xy[0]))

    eye = root_pos.clone()
    eye[0] += side_xy[0] * distance
    eye[1] += side_xy[1] * distance
    eye[2] += height

    target = root_pos.clone()
    target[2] += target_height
    base_env.sim.set_camera_view(tuple(float(v) for v in eye), tuple(float(v) for v in target))


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0, "final": 0.0}
    return {
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "final": float(array[-1]),
    }


def _ensure_parent_dir(path: str) -> None:
    parent = Path(path).parent
    if str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)


def _render_view_frame(env, width: int, height: int) -> np.ndarray | None:
    frame = env.unwrapped.render()
    if frame is None:
        return None
    frame_height, frame_width = frame.shape[:2]
    target_aspect = width / height
    frame_aspect = frame_width / frame_height
    if frame_aspect > target_aspect:
        crop_width = max(1, int(round(frame_height * target_aspect)))
        x0 = max(0, (frame_width - crop_width) // 2)
        frame = frame[:, x0 : x0 + crop_width]
    elif frame_aspect < target_aspect:
        crop_height = max(1, int(round(frame_width / target_aspect)))
        y0 = max(0, (frame_height - crop_height) // 2)
        frame = frame[y0 : y0 + crop_height, :]
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required for trailing-camera playback.")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.viewer.eye = (0.0, -1.0, 1.2)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.7)
    step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), args_cli.video_length * step_dt + 1.0)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, rsl_rl_train_cfg(agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, rsl_rl_train_cfg(agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    output_path = args_cli.output
    if output_path is None:
        output_path = os.path.join(log_dir, "videos", "play", "trailing-hud.mp4")
    _ensure_parent_dir(output_path)
    output_width = OUTPUT_WIDTH
    output_height = OUTPUT_HEIGHT
    view_width = output_width // 2
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        int(1.0 / env.unwrapped.step_dt),
        (output_width, output_height),
    )

    obs = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    contact_sensor: ContactSensor = env.unwrapped.scene.sensors["contact_forces"]
    contact_body_ids = _contact_body_ids(contact_sensor, ("foot1", "foot3"))
    command_manager = getattr(env.unwrapped, "command_manager", None)
    root_start_xy = robot.data.root_pos_w[0, :2].detach().clone()
    dt = env.unwrapped.step_dt
    fallback_hud_window_steps = max(1, int(round(args_cli.hud_window_s / dt)))
    fallback_camera_window_steps = max(1, int(round(args_cli.camera_window_s / dt)))
    speed_window: deque[float] = deque()
    command_window: deque[float] = deque()
    yaw_window: deque[float] = deque()
    root_height_window: deque[float] = deque()
    joint_pos_window: deque[np.ndarray] = deque()
    torso_tilt_window: deque[float] = deque()
    hip_roll_yaw_window: deque[np.ndarray] = deque()
    fsep_window: deque[float] = deque()
    ksep_window: deque[float] = deque()
    left_swing_clearance_window: deque[float] = deque()
    right_swing_clearance_window: deque[float] = deque()
    left_contact_window: deque[bool] = deque()
    right_contact_window: deque[bool] = deque()
    positive_work_window: deque[float] = deque()
    root_x_window: deque[float] = deque()
    forward_window: deque[torch.Tensor] = deque()
    left_touchdown_frames: deque[int] = deque(maxlen=max(args_cli.camera_cycle_window + 1, 2))
    right_touchdown_frames: deque[int] = deque(maxlen=max(args_cli.camera_cycle_window + 1, 2))
    touchdown_events: deque[tuple[int, str, float]] = deque(maxlen=max(4 * args_cli.camera_cycle_window + 8, 32))
    previous_contact = torch.zeros(2, dtype=torch.bool, device=robot.data.root_pos_w.device)
    joint_names = list(robot.data.joint_names)
    hip_roll_yaw_ids = [i for i, name in enumerate(joint_names) if "hip_roll" in name or "hip_yaw" in name]
    body_names = list(robot.body_names)
    foot_body_ids = [body_names.index("foot1"), body_names.index("foot3")]
    knee_proxy_body_ids = [body_names.index("leg2_shell"), body_names.index("leg2_shell_2")]
    sole_center_offsets = torch.tensor(
        [(0.03, -0.036528655, -0.0194786795), (0.03, -0.036528755, -0.0234786545)],
        dtype=robot.data.body_pos_w.dtype,
        device=robot.data.body_pos_w.device,
    )
    fall_reset_count = 0
    rollout_metrics: dict[str, list[float]] = {
        "speed": [],
        "command_speed": [],
        "yaw_rate": [],
        "root_height": [],
        "root_height_window_mean": [],
        "torso_tilt": [],
        "torso_tilt_window_mean": [],
        "torso_tilt_window_rms": [],
        "hip_roll_yaw_window_mean_abs": [],
        "hip_roll_yaw_window_rms": [],
        "fsep_m": [],
        "ksep_m": [],
        "left_swing_clearance_m": [],
        "right_swing_clearance_m": [],
        "x_distance_m": [],
        "y_distance_m": [],
        "positive_joint_work_j": [],
        "hud_joules_per_meter": [],
    }

    for frame_index in range(args_cli.video_length):
        start_time = time.time()
        if frame_index > 0 and float(robot.data.root_pos_w[0, 2].item()) <= args_cli.fall_reset_height:
            obs, _ = env.reset()
            if installed_version >= version.parse("4.0.0"):
                policy.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.unwrapped.device))
            fall_reset_count += 1
            previous_contact.zero_()
            speed_window.clear()
            command_window.clear()
            yaw_window.clear()
            root_height_window.clear()
            joint_pos_window.clear()
            torso_tilt_window.clear()
            hip_roll_yaw_window.clear()
            fsep_window.clear()
            ksep_window.clear()
            left_swing_clearance_window.clear()
            right_swing_clearance_window.clear()
            left_contact_window.clear()
            right_contact_window.clear()
            positive_work_window.clear()
            root_x_window.clear()
            forward_window.clear()
            left_touchdown_frames.clear()
            right_touchdown_frames.clear()
            root_start_xy = robot.data.root_pos_w[0, :2].detach().clone()

        foot_contact = contact_sensor.data.current_contact_time[0, contact_body_ids] > 0.0
        touchdown = foot_contact & ~previous_contact
        if bool(touchdown[0].item()):
            left_touchdown_frames.append(frame_index)
            touchdown_events.append((frame_index, "L", float(robot.data.root_pos_w[0, 0].item())))
        if bool(touchdown[1].item()):
            right_touchdown_frames.append(frame_index)
            touchdown_events.append((frame_index, "R", float(robot.data.root_pos_w[0, 0].item())))
        previous_contact = foot_contact.detach().clone()
        left_contact_window.append(bool(foot_contact[0].item()))
        right_contact_window.append(bool(foot_contact[1].item()))

        camera_window_steps = (
            _camera_cycle_window_steps(
                left_touchdown_frames,
                right_touchdown_frames,
                frame_index,
                args_cli.camera_cycle_window,
                fallback_camera_window_steps,
            )
            if args_cli.camera_window_mode == "cycles"
            else fallback_camera_window_steps
        )
        hud_window_steps, hud_window_label = _cycle_window_steps(
            left_touchdown_frames,
            right_touchdown_frames,
            frame_index,
            args_cli.camera_cycle_window,
            fallback_hud_window_steps,
        )
        forward_xy = _smooth_forward_xy(forward_window, _root_forward_xy(robot), camera_window_steps)
        speed = float(robot.data.root_lin_vel_b[0, 0].item())
        yaw_rate = float(robot.data.root_ang_vel_b[0, 2].item())
        root_height = float(robot.data.root_pos_w[0, 2].item())
        root_x = float(robot.data.root_pos_w[0, 0].item())
        root_delta_xy = robot.data.root_pos_w[0, :2] - root_start_xy
        x_distance = float(root_delta_xy[0].item())
        y_distance = float(root_delta_xy[1].item())
        command_speed = 0.0
        if command_manager is not None:
            try:
                command_speed = float(command_manager.get_command("base_velocity")[0, 0].item())
            except Exception:
                command_speed = 0.0
        speed_window.append(speed)
        command_window.append(command_speed)
        yaw_window.append(yaw_rate)
        root_height_window.append(root_height)
        joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
        joint_vel = robot.data.joint_vel[0]
        joint_torque = getattr(robot.data, "applied_torque", None)
        if joint_torque is None:
            joint_torque = getattr(robot.data, "computed_torque", None)
        if joint_torque is None:
            positive_work_step = 0.0
        else:
            joint_power = joint_torque[0] * joint_vel
            positive_work_step = float(torch.sum(torch.clamp(joint_power, min=0.0)).item() * dt)
        torso_tilt = float(robot.data.projected_gravity_b[0, 1].item())
        joint_pos_window.append(joint_pos)
        positive_work_window.append(positive_work_step)
        root_x_window.append(root_x)
        torso_tilt_window.append(torso_tilt)
        if hip_roll_yaw_ids:
            hip_roll_yaw_window.append(joint_pos[hip_roll_yaw_ids])
        else:
            hip_roll_yaw_window.append(np.zeros(1, dtype=np.float32))
        foot_pos_w = robot.data.body_pos_w[0, foot_body_ids]
        foot_quat_w = robot.data.body_quat_w[0, foot_body_ids]
        sole_pos_w = foot_pos_w + quat_apply(foot_quat_w, sole_center_offsets)
        sole_clearance = torch.clamp(sole_pos_w[:, 2], min=0.0).detach().cpu().numpy()
        root_pos_w = robot.data.root_pos_w[0:1]
        root_quat_w = robot.data.root_quat_w[0:1].expand(2, -1)
        sole_pos_b = quat_apply_inverse(root_quat_w, sole_pos_w - root_pos_w)
        knee_pos_w = robot.data.body_pos_w[0, knee_proxy_body_ids]
        knee_pos_b = quat_apply_inverse(root_quat_w, knee_pos_w - root_pos_w)
        foot_sep_y = abs(float(sole_pos_b[0, 1].item() - sole_pos_b[1, 1].item()))
        knee_sep_y = abs(float(knee_pos_b[0, 1].item() - knee_pos_b[1, 1].item()))
        fsep_window.append(foot_sep_y)
        ksep_window.append(knee_sep_y)
        left_swing_clearance_window.append(float(sole_clearance[0]) if not bool(foot_contact[0].item()) else float("nan"))
        right_swing_clearance_window.append(float(sole_clearance[1]) if not bool(foot_contact[1].item()) else float("nan"))
        for values in (
            speed_window,
            command_window,
            yaw_window,
            root_height_window,
            joint_pos_window,
            torso_tilt_window,
            hip_roll_yaw_window,
            fsep_window,
            ksep_window,
            left_swing_clearance_window,
            right_swing_clearance_window,
            left_contact_window,
            right_contact_window,
            positive_work_window,
            root_x_window,
        ):
            _trim_deque(values, hud_window_steps)
        torso_tilt_window_array = np.asarray(tuple(torso_tilt_window), dtype=np.float32)
        hip_roll_yaw_window_array = np.concatenate(tuple(hip_roll_yaw_window))
        root_height_window_mean = float(np.mean(root_height_window))
        torso_tilt_window_mean = float(np.mean(torso_tilt_window_array))
        torso_tilt_window_rms = float(np.sqrt(np.mean(np.square(torso_tilt_window_array))))
        hip_roll_yaw_window_mean_abs = float(np.mean(np.abs(hip_roll_yaw_window_array)))
        hip_roll_yaw_window_rms = float(np.sqrt(np.mean(np.square(hip_roll_yaw_window_array))))
        rollout_metrics["speed"].append(speed)
        rollout_metrics["command_speed"].append(command_speed)
        rollout_metrics["yaw_rate"].append(yaw_rate)
        rollout_metrics["root_height"].append(root_height)
        rollout_metrics["root_height_window_mean"].append(root_height_window_mean)
        rollout_metrics["torso_tilt"].append(torso_tilt)
        rollout_metrics["torso_tilt_window_mean"].append(torso_tilt_window_mean)
        rollout_metrics["torso_tilt_window_rms"].append(torso_tilt_window_rms)
        rollout_metrics["hip_roll_yaw_window_mean_abs"].append(hip_roll_yaw_window_mean_abs)
        rollout_metrics["hip_roll_yaw_window_rms"].append(hip_roll_yaw_window_rms)
        rollout_metrics["fsep_m"].append(foot_sep_y)
        rollout_metrics["ksep_m"].append(knee_sep_y)
        rollout_metrics["left_swing_clearance_m"].append(
            float(sole_clearance[0]) if not bool(foot_contact[0].item()) else float("nan")
        )
        rollout_metrics["right_swing_clearance_m"].append(
            float(sole_clearance[1]) if not bool(foot_contact[1].item()) else float("nan")
        )
        rollout_metrics["x_distance_m"].append(x_distance)
        rollout_metrics["y_distance_m"].append(y_distance)
        rollout_metrics["positive_joint_work_j"].append(positive_work_step)

        hud_joint_pos = np.mean(np.stack(tuple(joint_pos_window), axis=0), axis=0)
        hud_speed = float(np.mean(speed_window))
        hud_command_speed = float(np.mean(command_window))
        hud_yaw_rate = float(np.mean(yaw_window))
        hud_root_height = root_height_window_mean
        hud_fsep_m = float(np.mean(fsep_window))
        hud_ksep_m = float(np.mean(ksep_window))
        hud_left_swing_clearance_m = _nanmean_window(left_swing_clearance_window)
        hud_right_swing_clearance_m = _nanmean_window(right_swing_clearance_window)
        hud_step_stats = _recent_step_stats(touchdown_events, args_cli.camera_cycle_window, dt)
        hud_approved_step_fraction = hud_step_stats["approved_fraction"]
        root_x_array = np.asarray(tuple(root_x_window), dtype=np.float32)
        if root_x_array.size >= 2:
            hud_window_distance = max(float(root_x_array[-1] - root_x_array[0]), 1.0e-3)
        else:
            hud_window_distance = 1.0e-3
        hud_positive_work_j = float(np.sum(tuple(positive_work_window))) if positive_work_window else 0.0
        hud_joules_per_meter = hud_positive_work_j / hud_window_distance
        rollout_metrics["hud_joules_per_meter"].append(hud_joules_per_meter)
        left_contact_array = np.asarray(tuple(left_contact_window), dtype=bool)
        right_contact_array = np.asarray(tuple(right_contact_window), dtype=bool)
        if left_contact_array.size:
            left_single = left_contact_array & ~right_contact_array
            right_single = right_contact_array & ~left_contact_array
            double_support = left_contact_array & right_contact_array
            airborne = ~left_contact_array & ~right_contact_array
            hud_support_stats = {
                "left_stance": float(np.mean(left_contact_array)),
                "right_stance": float(np.mean(right_contact_array)),
                "left_single": float(np.mean(left_single)),
                "right_single": float(np.mean(right_single)),
                "double": float(np.mean(double_support)),
                "airborne": float(np.mean(airborne)),
            }
        else:
            hud_support_stats = {
                "left_stance": 0.0,
                "right_stance": 0.0,
                "left_single": 0.0,
                "right_single": 0.0,
                "double": 0.0,
                "airborne": 0.0,
            }

        _set_trailing_camera(
            env.unwrapped,
            robot,
            forward_xy,
            args_cli.camera_distance,
            args_cli.camera_height,
            args_cli.target_distance,
            args_cli.target_height,
        )
        trailing_frame = _render_view_frame(env, view_width, output_height)

        _set_side_camera(
            env.unwrapped,
            robot,
            forward_xy,
            args_cli.camera_distance,
            args_cli.camera_height,
            args_cli.target_distance,
            args_cli.target_height,
        )
        side_frame = _render_view_frame(env, view_width, output_height)

        if trailing_frame is not None and side_frame is not None:
            frame = np.concatenate((trailing_frame, side_frame), axis=1)
            frame = _draw_hud(
                frame,
                hud_speed,
                hud_command_speed,
                hud_yaw_rate,
                hud_root_height,
                x_distance,
                y_distance,
                joint_names,
                hud_joint_pos,
                torso_tilt_window_mean,
                torso_tilt_window_rms,
                hip_roll_yaw_window_rms,
                hud_window_label,
                hud_step_stats,
                hud_support_stats,
                hud_fsep_m,
                hud_ksep_m,
                hud_joules_per_meter,
                hud_approved_step_fraction,
                hud_left_swing_clearance_m,
                hud_right_swing_clearance_m,
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    writer.release()
    print(f"[INFO] Wrote side-by-side trailing/side HUD video to: {output_path}")
    if args_cli.metrics_output is not None:
        metrics_path = Path(args_cli.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "checkpoint": resume_path,
            "video": output_path,
            "video_layout": "16:9 trailing_left_8:9_side_right_8:9_single_hud",
            "output_width": output_width,
            "output_height": output_height,
            "video_length_steps": args_cli.video_length,
            "dt": dt,
            "window_s": args_cli.hud_window_s,
            "hud_window_mode": "last_full_gait_cycles_with_time_fallback",
            "camera_window_mode": args_cli.camera_window_mode,
            "camera_window_s_fallback": args_cli.camera_window_s,
            "camera_cycle_window": args_cli.camera_cycle_window,
            "final_hud_window_label": hud_window_label,
            "final_hud_fsep_m": hud_fsep_m,
            "final_hud_ksep_m": hud_ksep_m,
            "final_hud_sep_m": hud_fsep_m,
            "final_hud_left_swing_clearance_m": hud_left_swing_clearance_m,
            "final_hud_right_swing_clearance_m": hud_right_swing_clearance_m,
            "final_x_distance_m": x_distance,
            "final_y_distance_m": y_distance,
            "final_hud_positive_joint_work_j": hud_positive_work_j,
            "final_hud_joules_per_meter": hud_joules_per_meter,
            "final_hud_approved_step_fraction": hud_approved_step_fraction,
            "bootstrap_approved_step_advance_m": BOOTSTRAP_APPROVED_STEP_ADVANCE_M,
            "bootstrap_approved_step_duration_s": BOOTSTRAP_APPROVED_STEP_DURATION_S,
            "final_hud_step_stats": hud_step_stats,
            "fall_reset_height": args_cli.fall_reset_height,
            "fall_reset_count": fall_reset_count,
            "policy_reset_mode": "fall_reset_only",
            "joint_names": joint_names,
            "hip_roll_yaw_joint_names": [joint_names[index] for index in hip_roll_yaw_ids],
            "metrics": {
                name: _summary(values)
                for name, values in rollout_metrics.items()
            },
        }
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote rollout metrics to: {metrics_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
