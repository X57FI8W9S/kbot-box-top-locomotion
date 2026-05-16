#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import json
import math
import os
import sys
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


parser = argparse.ArgumentParser(description="Evaluate a KBot checkpoint with gait diagnostics and hard gates.")
parser.add_argument("--baseline_metrics", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--video_length", type=int, default=1500)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--cycle_window", type=int, default=5)
parser.add_argument("--hip_roll_mean_limit", type=float, default=0.03)
parser.add_argument("--torso_roll_mean_limit", type=float, default=0.025)
parser.add_argument("--yaw_drift_per_meter_limit", type=float, default=0.35)
parser.add_argument("--lateral_drift_per_meter_limit", type=float, default=0.10)
parser.add_argument("--min_timeout_fraction", type=float, default=1.0)
parser.add_argument("--min_speed_tracking_ratio", type=float, default=0.80)
parser.add_argument("--min_fsep_mean", type=float, default=0.28)
parser.add_argument("--min_fsep_p05", type=float, default=0.24)
parser.add_argument("--min_ksep_mean", type=float, default=0.26)
parser.add_argument("--max_fsep_target_error_mean", type=float, default=0.06)
parser.add_argument("--max_cycle_cadence_hz", type=float, default=2.50)
parser.add_argument("--min_cmd_vx_for_step_gates", type=float, default=0.05)
parser.add_argument("--min_step_root_advance_m", type=float, default=0.02)
parser.add_argument("--min_cycle_root_advance_m", type=float, default=0.04)
parser.add_argument("--approved_step_advance_m", type=float, default=0.008)
parser.add_argument("--approved_step_duration_s", type=float, default=0.07)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = False

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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

sys.path.insert(0, str(REPO_ROOT / "scripts" / "rsl_rl"))
from rsl_rl_compat import rsl_rl_train_cfg  # noqa: E402


installed_version = version.parse(metadata.version("rsl-rl-lib"))


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "p05": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0, "final": 0.0}
    return {
        "mean": float(np.mean(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "min": float(np.min(array)),
        "final": float(array[-1]),
    }


def _finite_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": 0.0, "p05": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0, "final": 0.0}
    return _summary(array.tolist())


def _rms_centered(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    centered = values - np.mean(values)
    return float(np.sqrt(np.mean(np.square(centered))))


def _joint_ids(joint_names: list[str], pattern: str) -> list[int]:
    return [i for i, name in enumerate(joint_names) if pattern in name]


def _contact_body_ids(contact_sensor: ContactSensor, names: tuple[str, ...]) -> list[int]:
    body_names = getattr(contact_sensor, "body_names", None)
    if body_names is None:
        body_names = getattr(contact_sensor.data, "body_names", None)
    if body_names is not None:
        return [list(body_names).index(name) for name in names]
    return list(range(len(names)))


def _paired_joint_symmetry(joint_names: list[str], samples: np.ndarray) -> dict[str, float]:
    sign_map = {"hip_pitch": 1.0, "hip_roll": -1.0, "hip_yaw": -1.0, "knee": -1.0, "ankle": -1.0}
    out: dict[str, float] = {}
    for key, sign in sign_map.items():
        left = [i for i, name in enumerate(joint_names) if name.startswith("left_") and key in name]
        right = [i for i, name in enumerate(joint_names) if name.startswith("right_") and key in name]
        if not left or not right:
            continue
        left_mean = float(np.mean(samples[:, left[0]]))
        right_norm_mean = float(sign * np.mean(samples[:, right[0]]))
        out[f"{key}_mean_error"] = left_mean - right_norm_mean
        out[f"{key}_mean_abs_error"] = abs(left_mean - right_norm_mean)
    return out


def _mean(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([row[key] for row in rows]))


def _last_mean(rows: list[dict], key: str, count: int) -> float:
    if not rows:
        return 0.0
    return _mean(rows[-count:], key)


def _std(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.std([row[key] for row in rows]))


def _cadence_hz(rows: list[dict]) -> float:
    duration = _mean(rows, "duration_s")
    return 1.0 / duration if duration > 1.0e-6 else 0.0


def _fraction(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([bool(row[key]) for row in rows]))


def _baseline_scorecard(path: str | None) -> dict[str, float]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "scorecard" in data:
        return data["scorecard"]
    if "metrics" in data and "scorecard" in data["metrics"]:
        return data["metrics"]["scorecard"]
    return data


def _percent_change(current: float, baseline: float) -> float:
    if abs(baseline) < 1.0e-9:
        return 0.0 if abs(current) < 1.0e-9 else math.inf
    return 100.0 * (current - baseline) / abs(baseline)


def _baseline_comparison(scorecard: dict[str, float], baseline: dict[str, float]) -> dict[str, dict[str, float]]:
    keys = (
        "cycle_cadence_hz",
        "step_root_advance_mean_m",
        "cycle_root_advance_mean_m",
        "approved_step_fraction",
        "left_approved_step_fraction",
        "right_approved_step_fraction",
        "root_height_p05_m",
        "fsep_mean_m",
        "ksep_mean_m",
        "speed_tracking_ratio",
        "yaw_drift_rad_per_m",
        "lateral_drift_m_per_m",
        "swing_sole_clearance_mean_left_m",
        "swing_sole_clearance_mean_right_m",
        "swing_sole_clearance_p05_left_m",
        "swing_sole_clearance_p05_right_m",
    )
    comparison: dict[str, dict[str, float]] = {}
    for key in keys:
        if key not in scorecard or key not in baseline:
            continue
        current = float(scorecard[key])
        base = float(baseline[key])
        comparison[key] = {
            "baseline": base,
            "current": current,
            "delta": current - base,
            "percent_change": _percent_change(current, base),
        }
    return comparison


def _events(
    time: np.ndarray,
    contact: np.ndarray,
    foot_positions: np.ndarray,
    root_x: np.ndarray,
) -> tuple[list[dict], list[dict], list[dict]]:
    events: list[dict] = []
    previous = contact[0]
    for i in range(1, len(time)):
        for foot_i, foot_name in enumerate(("L", "R")):
            if contact[i, foot_i] and not previous[foot_i]:
                events.append(
                    {
                        "type": "touchdown",
                        "foot": foot_name,
                        "time": float(time[i]),
                        "frame": int(i),
                        "x": float(foot_positions[i, foot_i, 0]),
                        "y": float(foot_positions[i, foot_i, 1]),
                    }
                )
            if previous[foot_i] and not contact[i, foot_i]:
                events.append(
                    {
                        "type": "toe_off",
                        "foot": foot_name,
                        "time": float(time[i]),
                        "frame": int(i),
                        "x": float(foot_positions[i, foot_i, 0]),
                        "y": float(foot_positions[i, foot_i, 1]),
                    }
                )
        previous = contact[i]

    touchdowns = [event for event in events if event["type"] == "touchdown"]
    steps: list[dict] = []
    for prev, current in zip(touchdowns, touchdowns[1:]):
        if prev["foot"] == current["foot"]:
            continue
        steps.append(
            {
                "step_foot": prev["foot"],
                "start_foot": prev["foot"],
                "end_foot": current["foot"],
                "start_time": prev["time"],
                "end_time": current["time"],
                "duration_s": current["time"] - prev["time"],
                "step_length_m": current["x"] - prev["x"],
                "root_advance_m": float(root_x[current["frame"]] - root_x[prev["frame"]]),
                "step_width_m": abs(current["y"] - prev["y"]),
                "start_frame": prev["frame"],
                "end_frame": current["frame"],
            }
        )

    cycles: list[dict] = []
    for foot in ("L", "R"):
        foot_touchdowns = [event for event in touchdowns if event["foot"] == foot]
        for prev, current in zip(foot_touchdowns, foot_touchdowns[1:]):
            cycles.append(
                {
                    "cycle_foot": foot,
                    "start_time": prev["time"],
                    "end_time": current["time"],
                    "duration_s": current["time"] - prev["time"],
                    "cycle_length_m": current["x"] - prev["x"],
                    "root_advance_m": float(root_x[current["frame"]] - root_x[prev["frame"]]),
                    "start_frame": prev["frame"],
                    "end_frame": current["frame"],
                }
            )
    cycles.sort(key=lambda row: row["start_time"])
    return events, steps, cycles


def _add_support_metrics(
    rows: list[dict],
    contact: np.ndarray,
    full_support: np.ndarray,
    events: list[dict],
    dt: float,
    stance_key: str,
) -> None:
    toe_offs = [event for event in events if event["type"] == "toe_off"]
    foot_index = {"L": 0, "R": 1}
    for row in rows:
        start_frame = row["start_frame"]
        end_frame = row["end_frame"]
        interval_contact = contact[start_frame:end_frame]
        interval_full_support = full_support[start_frame:end_frame]
        duration = max(row["duration_s"], 1.0e-6)

        if interval_contact.size == 0:
            double_support_duration = 0.0
            single_support_l_duration = 0.0
            single_support_r_duration = 0.0
            airborne_duration = 0.0
            full_support_duration = 0.0
        else:
            left_contact = interval_contact[:, 0]
            right_contact = interval_contact[:, 1]
            double_support_duration = float(np.sum(left_contact & right_contact) * dt)
            single_support_l_duration = float(np.sum(left_contact & ~right_contact) * dt)
            single_support_r_duration = float(np.sum(right_contact & ~left_contact) * dt)
            airborne_duration = float(np.sum(~left_contact & ~right_contact) * dt)
            stance_foot = row[stance_key]
            stance_contact = interval_contact[:, foot_index[stance_foot]]
            stance_duration = float(np.sum(stance_contact) * dt)
            full_support_duration = float(np.sum(interval_full_support[:, foot_index[stance_foot]]) * dt)

        if interval_contact.size == 0:
            stance_duration = 0.0
        swing_duration = max(duration - stance_duration, 0.0)
        row["double_support_duration_s"] = double_support_duration
        row["single_support_l_duration_s"] = single_support_l_duration
        row["single_support_r_duration_s"] = single_support_r_duration
        row["airborne_duration_s"] = airborne_duration
        row["stance_duration_s"] = stance_duration
        row["swing_duration_s"] = swing_duration
        row["full_support_duration_s"] = full_support_duration
        row["double_support_ratio"] = double_support_duration / duration
        row["single_support_l_ratio"] = single_support_l_duration / duration
        row["single_support_r_ratio"] = single_support_r_duration / duration
        row["airborne_ratio"] = airborne_duration / duration
        row["duty_factor"] = stance_duration / duration
        row["swing_ratio"] = swing_duration / duration
        row["full_support_ratio"] = full_support_duration / duration

        opposite_foot = "R" if row[stance_key] == "L" else "L"
        opposite_toe_off = next(
            (
                event
                for event in toe_offs
                if event["foot"] == opposite_foot
                and row["start_frame"] <= event["frame"] <= row["end_frame"]
            ),
            None,
        )
        row["opposite_toe_off_time"] = opposite_toe_off["time"] if opposite_toe_off is not None else ""
        row["landing_to_opposite_toe_off_s"] = (
            opposite_toe_off["time"] - row["start_time"] if opposite_toe_off is not None else 0.0
        )


def _cycle_window_start(events: list[dict], cycle_count: int, dt: float) -> float:
    left_touchdowns = [event["time"] for event in events if event["type"] == "touchdown" and event["foot"] == "L"]
    if len(left_touchdowns) >= cycle_count + 1:
        return float(left_touchdowns[-(cycle_count + 1)])
    right_touchdowns = [event["time"] for event in events if event["type"] == "touchdown" and event["foot"] == "R"]
    if len(right_touchdowns) >= cycle_count + 1:
        return float(right_touchdowns[-(cycle_count + 1)])
    return 0.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _html(summary: dict) -> str:
    gate_rows = "\n".join(
        f"<tr><td>{gate}</td><td>{'PASS' if ok else 'FAIL'}</td></tr>"
        for gate, ok in summary["gates"].items()
    )
    metric_rows = "\n".join(
        f"<tr><td>{name}</td><td>{value}</td></tr>"
        for name, value in summary["scorecard"].items()
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>KBot Diagnostics</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; margin: 16px 0; min-width: 520px; }}
    td, th {{ border: 1px solid #c8d0d8; padding: 6px 9px; text-align: left; }}
    th {{ background: #e8edf2; }}
    .decision {{ font-size: 24px; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>KBot Diagnostics</h1>
  <div class="decision">{summary["decision"]}</div>
  <h2>Gates</h2>
  <table><tr><th>Gate</th><th>Status</th></tr>{gate_rows}</table>
  <h2>Scorecard</h2>
  <table><tr><th>Metric</th><th>Value</th></tr>{metric_rows}</table>
</body>
</html>
"""


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), args_cli.video_length * step_dt + 1.0)

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else Path(os.path.dirname(checkpoint_path)) / "diagnostics" / Path(checkpoint_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, rsl_rl_train_cfg(agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, rsl_rl_train_cfg(agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    contact_sensor: ContactSensor = env.unwrapped.scene.sensors["contact_forces"]
    contact_body_names = list(getattr(contact_sensor, "body_names", None) or getattr(contact_sensor.data, "body_names", None) or [])
    pad_names = ("left_heel_pad", "left_toe_pad", "right_heel_pad", "right_toe_pad")
    use_pad_contacts = all(name in contact_body_names for name in pad_names)
    if use_pad_contacts:
        pad_body_ids = _contact_body_ids(contact_sensor, pad_names)
        contact_body_ids = pad_body_ids
    else:
        pad_body_ids = []
        contact_body_ids = _contact_body_ids(contact_sensor, ("foot1", "foot3"))
    command_manager = getattr(env.unwrapped, "command_manager", None)
    joint_names = list(robot.data.joint_names)
    hip_roll_ids = _joint_ids(joint_names, "hip_roll")
    hip_yaw_ids = _joint_ids(joint_names, "hip_yaw")
    knee_ids = _joint_ids(joint_names, "knee")
    body_names = list(robot.body_names)
    foot_body_ids = [body_names.index("foot1"), body_names.index("foot3")]
    knee_proxy_body_ids = [body_names.index("leg2_shell"), body_names.index("leg2_shell_2")]
    sole_center_offsets = torch.tensor(
        [(0.03, -0.036528655, -0.0194786795), (0.03, -0.036528755, -0.0234786545)],
        dtype=robot.data.body_pos_w.dtype,
        device=robot.data.body_pos_w.device,
    )

    series: dict[str, list] = {
        "time": [],
        "vx": [],
        "cmd_vx": [],
        "vy": [],
        "yaw_rate": [],
        "root_x": [],
        "root_y": [],
        "root_z": [],
        "root_roll_proxy": [],
        "root_pitch_proxy": [],
        "joint_pos": [],
        "contact": [],
        "pad_contact": [],
        "foot_pos": [],
        "foot_up_z": [],
        "foot_forward_z": [],
        "foot_lateral_z": [],
        "sole_clearance": [],
        "fsep": [],
        "ksep": [],
        "stance_slip": [],
    }

    for step in range(args_cli.video_length):
        t = step * step_dt
        command_speed = 0.0
        if command_manager is not None:
            try:
                command_speed = float(command_manager.get_command("base_velocity")[0, 0].item())
            except Exception:
                command_speed = 0.0
        contact_time = contact_sensor.data.current_contact_time[0, contact_body_ids].detach().cpu().numpy()
        raw_contact = contact_time > 0.0
        if use_pad_contacts:
            pad_contact = raw_contact.astype(np.float32)
            contact = np.asarray([raw_contact[0] or raw_contact[1], raw_contact[2] or raw_contact[3]], dtype=bool)
        else:
            pad_contact = np.zeros(4, dtype=np.float32)
            contact = raw_contact
        foot_pos = robot.data.body_pos_w[0, foot_body_ids].detach().cpu().numpy()
        foot_quat = robot.data.body_quat_w[0, foot_body_ids]
        up_b = torch.zeros(2, 3, device=foot_quat.device)
        up_b[:, 2] = 1.0
        forward_b = torch.zeros(2, 3, device=foot_quat.device)
        forward_b[:, 0] = 1.0
        lateral_b = torch.zeros(2, 3, device=foot_quat.device)
        lateral_b[:, 1] = 1.0
        up_z = quat_apply(foot_quat, up_b)[:, 2].detach().cpu().numpy()
        forward_z = quat_apply(foot_quat, forward_b)[:, 2].detach().cpu().numpy()
        lateral_z = quat_apply(foot_quat, lateral_b)[:, 2].detach().cpu().numpy()
        sole_pos_w = robot.data.body_pos_w[0, foot_body_ids] + quat_apply(foot_quat, sole_center_offsets)
        sole_clearance = torch.clamp(sole_pos_w[:, 2], min=0.0).detach().cpu().numpy()
        root_pos_w = robot.data.root_pos_w[0:1]
        root_quat_w = robot.data.root_quat_w[0:1].expand(2, -1)
        sole_pos_b = quat_apply_inverse(root_quat_w, sole_pos_w - root_pos_w)
        knee_pos_w = robot.data.body_pos_w[0, knee_proxy_body_ids]
        knee_pos_b = quat_apply_inverse(root_quat_w, knee_pos_w - root_pos_w)
        fsep = abs(float(sole_pos_b[0, 1].item() - sole_pos_b[1, 1].item()))
        ksep = abs(float(knee_pos_b[0, 1].item() - knee_pos_b[1, 1].item()))
        foot_vel = robot.data.body_lin_vel_w[0, foot_body_ids, :2].detach().cpu().numpy()
        stance_slip = np.linalg.norm(foot_vel, axis=1) * contact

        series["time"].append(t)
        series["vx"].append(float(robot.data.root_lin_vel_b[0, 0].item()))
        series["cmd_vx"].append(command_speed)
        series["vy"].append(float(robot.data.root_lin_vel_b[0, 1].item()))
        series["yaw_rate"].append(float(robot.data.root_ang_vel_b[0, 2].item()))
        series["root_x"].append(float(robot.data.root_pos_w[0, 0].item()))
        series["root_y"].append(float(robot.data.root_pos_w[0, 1].item()))
        series["root_z"].append(float(robot.data.root_pos_w[0, 2].item()))
        series["root_roll_proxy"].append(float(robot.data.projected_gravity_b[0, 1].item()))
        series["root_pitch_proxy"].append(float(robot.data.projected_gravity_b[0, 0].item()))
        series["joint_pos"].append(robot.data.joint_pos[0].detach().cpu().numpy())
        series["contact"].append(contact.astype(np.float32))
        series["pad_contact"].append(pad_contact)
        series["foot_pos"].append(foot_pos)
        series["foot_up_z"].append(up_z)
        series["foot_forward_z"].append(forward_z)
        series["foot_lateral_z"].append(lateral_z)
        series["sole_clearance"].append(sole_clearance)
        series["fsep"].append(fsep)
        series["ksep"].append(ksep)
        series["stance_slip"].append(stance_slip)

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if installed_version >= version.parse("4.0.0"):
                policy.reset(dones)

    arrays = {key: np.asarray(value) for key, value in series.items()}
    events, steps, cycles = _events(
        arrays["time"],
        arrays["contact"].astype(bool),
        arrays["foot_pos"],
        arrays["root_x"],
    )
    window_start = _cycle_window_start(events, args_cli.cycle_window, step_dt)
    window_mask = arrays["time"] >= window_start
    joint_window = arrays["joint_pos"][window_mask]
    root_roll_window = arrays["root_roll_proxy"][window_mask]
    hip_roll_window = joint_window[:, hip_roll_ids] if hip_roll_ids else np.zeros((0, 1))
    hip_yaw_window = joint_window[:, hip_yaw_ids] if hip_yaw_ids else np.zeros((0, 1))
    knee_window = np.abs(joint_window[:, knee_ids]) if knee_ids else np.zeros((0, 1))
    contact = arrays["contact"].astype(bool)
    pad_contact = arrays["pad_contact"].astype(bool)
    double_support = contact[:, 0] & contact[:, 1]
    airborne = ~contact[:, 0] & ~contact[:, 1]
    swing_clearance = np.where(~contact, arrays["sole_clearance"], np.nan)
    left_swing_clearance = swing_clearance[:, 0]
    right_swing_clearance = swing_clearance[:, 1]
    if use_pad_contacts:
        full_support = np.column_stack((pad_contact[:, 0] & pad_contact[:, 1], pad_contact[:, 2] & pad_contact[:, 3]))
        toe_only = np.column_stack((pad_contact[:, 1] & ~pad_contact[:, 0], pad_contact[:, 3] & ~pad_contact[:, 2]))
        heel_only = np.column_stack((pad_contact[:, 0] & ~pad_contact[:, 1], pad_contact[:, 2] & ~pad_contact[:, 3]))
    else:
        full_support = contact & (arrays["foot_up_z"] > 0.96)
        not_flat = contact & (arrays["foot_up_z"] <= 0.96)
        toe_only = not_flat & (arrays["foot_forward_z"] < -0.12)
        heel_only = not_flat & (arrays["foot_forward_z"] > 0.12)
    edge_walk = contact & (np.abs(arrays["foot_lateral_z"]) > 0.12)
    inner_edge = np.column_stack(
        (
            edge_walk[:, 0] & (arrays["foot_lateral_z"][:, 0] < 0.0),
            edge_walk[:, 1] & (arrays["foot_lateral_z"][:, 1] > 0.0),
        )
    )
    outer_edge = np.column_stack(
        (
            edge_walk[:, 0] & (arrays["foot_lateral_z"][:, 0] > 0.0),
            edge_walk[:, 1] & (arrays["foot_lateral_z"][:, 1] < 0.0),
        )
    )
    _add_support_metrics(steps, contact, full_support, events, step_dt, "step_foot")
    _add_support_metrics(cycles, contact, full_support, events, step_dt, "cycle_foot")
    x_distance = float(arrays["root_x"][-1] - arrays["root_x"][0])
    distance = max(x_distance, 1.0e-6)
    yaw_drift = float(np.trapz(arrays["yaw_rate"], arrays["time"]))
    lateral_drift = float(arrays["root_y"][-1] - arrays["root_y"][0])
    speed_ratio = float(np.mean(arrays["vx"]) / max(np.mean(arrays["cmd_vx"]), 1.0e-6))

    left_steps = [row for row in steps if row["step_foot"] == "L"]
    right_steps = [row for row in steps if row["step_foot"] == "R"]
    left_cycles = [row for row in cycles if row["cycle_foot"] == "L"]
    right_cycles = [row for row in cycles if row["cycle_foot"] == "R"]
    for row in steps:
        row["approved_step"] = (
            row["root_advance_m"] >= args_cli.approved_step_advance_m
            and row["duration_s"] >= args_cli.approved_step_duration_s
        )
    approved_steps = [row for row in steps if row["approved_step"]]
    left_approved_steps = [row for row in left_steps if row["approved_step"]]
    right_approved_steps = [row for row in right_steps if row["approved_step"]]
    step_lengths = [row["step_length_m"] for row in steps]
    step_advances = [row["root_advance_m"] for row in steps]
    step_durations = [row["duration_s"] for row in steps]
    cycle_lengths = [row["cycle_length_m"] for row in cycles]
    cycle_advances = [row["root_advance_m"] for row in cycles]
    cycle_durations = [row["duration_s"] for row in cycles]
    left_cycle_duration_last5 = _last_mean(left_cycles, "duration_s", 5)
    right_cycle_duration_last5 = _last_mean(right_cycles, "duration_s", 5)
    left_step_duration_last5 = _last_mean(left_steps, "duration_s", 5)
    right_step_duration_last5 = _last_mean(right_steps, "duration_s", 5)
    left_cycle_cadence_hz = _cadence_hz(left_cycles)
    right_cycle_cadence_hz = _cadence_hz(right_cycles)
    cycle_cadences = [hz for hz in (left_cycle_cadence_hz, right_cycle_cadence_hz) if hz > 0.0]
    cycle_cadence_hz = float(np.mean(cycle_cadences)) if cycle_cadences else 0.0
    moving_command = float(np.mean(arrays["cmd_vx"])) > args_cli.min_cmd_vx_for_step_gates
    symmetry = _paired_joint_symmetry(joint_names, joint_window) if joint_window.size else {}
    scorecard = {
        "distance_m": distance,
        "x_distance_m": x_distance,
        "y_distance_m": lateral_drift,
        "speed_mean_mps": float(np.mean(arrays["vx"])),
        "command_speed_mean_mps": float(np.mean(arrays["cmd_vx"])),
        "speed_tracking_ratio": speed_ratio,
        "yaw_drift_rad_per_m": yaw_drift / distance,
        "lateral_drift_m_per_m": lateral_drift / distance,
        "root_roll_mean_5cycle": float(np.mean(root_roll_window)) if root_roll_window.size else 0.0,
        "root_roll_rms_centered_5cycle": _rms_centered(root_roll_window),
        "hip_roll_mean_abs_5cycle_rad": float(np.mean(np.abs(hip_roll_window))) if hip_roll_window.size else 0.0,
        "hip_roll_rms_centered_5cycle_rad": _rms_centered(hip_roll_window.reshape(-1)),
        "hip_yaw_mean_abs_5cycle_rad": float(np.mean(np.abs(hip_yaw_window))) if hip_yaw_window.size else 0.0,
        "knee_abs_mean_5cycle_rad": float(np.mean(knee_window)) if knee_window.size else 0.0,
        "root_height_mean_m": float(np.mean(arrays["root_z"])),
        "root_height_p05_m": float(np.percentile(arrays["root_z"], 5)),
        "fsep_mean_m": float(np.mean(arrays["fsep"])),
        "fsep_p05_m": float(np.percentile(arrays["fsep"], 5)),
        "fsep_final_m": float(arrays["fsep"][-1]),
        "fsep_target_error_mean_m": float(np.mean(np.abs(arrays["fsep"] - 0.3164))),
        "ksep_mean_m": float(np.mean(arrays["ksep"])),
        "ksep_p05_m": float(np.percentile(arrays["ksep"], 5)),
        "ksep_final_m": float(arrays["ksep"][-1]),
        "double_support_fraction": float(np.mean(double_support)),
        "airborne_fraction": float(np.mean(airborne)),
        "full_support_fraction_left": float(np.mean(full_support[:, 0])),
        "full_support_fraction_right": float(np.mean(full_support[:, 1])),
        "toe_only_fraction_left": float(np.mean(toe_only[:, 0])),
        "toe_only_fraction_right": float(np.mean(toe_only[:, 1])),
        "heel_only_fraction_left": float(np.mean(heel_only[:, 0])),
        "heel_only_fraction_right": float(np.mean(heel_only[:, 1])),
        "sole_normal_z_mean_left": float(np.mean(arrays["foot_up_z"][:, 0])),
        "sole_normal_z_mean_right": float(np.mean(arrays["foot_up_z"][:, 1])),
        "swing_sole_clearance_mean_left_m": _finite_summary(left_swing_clearance)["mean"],
        "swing_sole_clearance_mean_right_m": _finite_summary(right_swing_clearance)["mean"],
        "swing_sole_clearance_p05_left_m": _finite_summary(left_swing_clearance)["p05"],
        "swing_sole_clearance_p05_right_m": _finite_summary(right_swing_clearance)["p05"],
        "swing_sole_clearance_max_left_m": _finite_summary(left_swing_clearance)["max"],
        "swing_sole_clearance_max_right_m": _finite_summary(right_swing_clearance)["max"],
        "stance_sole_tilt_l2_mean": float(np.mean((1.0 - np.square(arrays["foot_up_z"])) * contact)),
        "toe_down_proxy_fraction_left": float(np.mean(toe_only[:, 0])),
        "toe_down_proxy_fraction_right": float(np.mean(toe_only[:, 1])),
        "heel_down_proxy_fraction_left": float(np.mean(heel_only[:, 0])),
        "heel_down_proxy_fraction_right": float(np.mean(heel_only[:, 1])),
        "edge_walk_proxy_fraction_left": float(np.mean(edge_walk[:, 0])),
        "edge_walk_proxy_fraction_right": float(np.mean(edge_walk[:, 1])),
        "inner_edge_proxy_fraction_left": float(np.mean(inner_edge[:, 0])),
        "inner_edge_proxy_fraction_right": float(np.mean(inner_edge[:, 1])),
        "outer_edge_proxy_fraction_left": float(np.mean(outer_edge[:, 0])),
        "outer_edge_proxy_fraction_right": float(np.mean(outer_edge[:, 1])),
        "stance_slip_mean_mps": float(np.mean(arrays["stance_slip"])),
        "step_count": len(steps),
        "left_step_count": len(left_steps),
        "right_step_count": len(right_steps),
        "approved_step_count": len(approved_steps),
        "left_approved_step_count": len(left_approved_steps),
        "right_approved_step_count": len(right_approved_steps),
        "approved_step_fraction": _fraction(steps, "approved_step"),
        "left_approved_step_fraction": _fraction(left_steps, "approved_step"),
        "right_approved_step_fraction": _fraction(right_steps, "approved_step"),
        "approved_step_advance_threshold_m": args_cli.approved_step_advance_m,
        "approved_step_duration_threshold_s": args_cli.approved_step_duration_s,
        "step_length_mean_m": float(np.mean(step_lengths)) if step_lengths else 0.0,
        "step_root_advance_mean_m": float(np.mean(step_advances)) if step_advances else 0.0,
        "step_duration_mean_s": float(np.mean(step_durations)) if step_durations else 0.0,
        "left_step_length_mean_m": _mean(left_steps, "step_length_m"),
        "right_step_length_mean_m": _mean(right_steps, "step_length_m"),
        "left_step_root_advance_mean_m": _mean(left_steps, "root_advance_m"),
        "right_step_root_advance_mean_m": _mean(right_steps, "root_advance_m"),
        "left_step_duration_mean_s": _mean(left_steps, "duration_s"),
        "right_step_duration_mean_s": _mean(right_steps, "duration_s"),
        "left_step_length_last5_mean_m": _last_mean(left_steps, "step_length_m", 5),
        "right_step_length_last5_mean_m": _last_mean(right_steps, "step_length_m", 5),
        "left_step_root_advance_last5_mean_m": _last_mean(left_steps, "root_advance_m", 5),
        "right_step_root_advance_last5_mean_m": _last_mean(right_steps, "root_advance_m", 5),
        "left_step_duration_last5_mean_s": _last_mean(left_steps, "duration_s", 5),
        "right_step_duration_last5_mean_s": _last_mean(right_steps, "duration_s", 5),
        "step_duration_std_s": _std(steps, "duration_s"),
        "left_right_step_duration_error_mean_s": _mean(left_steps, "duration_s") - _mean(right_steps, "duration_s"),
        "left_right_step_duration_error_last5_s": left_step_duration_last5 - right_step_duration_last5,
        "step_double_support_ratio_mean": _mean(steps, "double_support_ratio"),
        "left_step_double_support_ratio_mean": _mean(left_steps, "double_support_ratio"),
        "right_step_double_support_ratio_mean": _mean(right_steps, "double_support_ratio"),
        "step_full_support_ratio_mean": _mean(steps, "full_support_ratio"),
        "left_step_full_support_ratio_mean": _mean(left_steps, "full_support_ratio"),
        "right_step_full_support_ratio_mean": _mean(right_steps, "full_support_ratio"),
        "left_step_full_support_ratio_last5_mean": _last_mean(left_steps, "full_support_ratio", 5),
        "right_step_full_support_ratio_last5_mean": _last_mean(right_steps, "full_support_ratio", 5),
        "left_landing_to_opposite_toe_off_last5_mean_s": _last_mean(
            left_steps, "landing_to_opposite_toe_off_s", 5
        ),
        "right_landing_to_opposite_toe_off_last5_mean_s": _last_mean(
            right_steps, "landing_to_opposite_toe_off_s", 5
        ),
        "cycle_count": len(cycles),
        "left_cycle_count": len(left_cycles),
        "right_cycle_count": len(right_cycles),
        "cycle_length_mean_m": float(np.mean(cycle_lengths)) if cycle_lengths else 0.0,
        "cycle_root_advance_mean_m": float(np.mean(cycle_advances)) if cycle_advances else 0.0,
        "cycle_duration_mean_s": float(np.mean(cycle_durations)) if cycle_durations else 0.0,
        "cycle_duration_std_s": _std(cycles, "duration_s"),
        "cycle_cadence_hz": cycle_cadence_hz,
        "left_cycle_cadence_hz": left_cycle_cadence_hz,
        "right_cycle_cadence_hz": right_cycle_cadence_hz,
        "max_cycle_cadence_hz": max(left_cycle_cadence_hz, right_cycle_cadence_hz),
        "left_cycle_length_mean_m": _mean(left_cycles, "cycle_length_m"),
        "right_cycle_length_mean_m": _mean(right_cycles, "cycle_length_m"),
        "left_cycle_root_advance_mean_m": _mean(left_cycles, "root_advance_m"),
        "right_cycle_root_advance_mean_m": _mean(right_cycles, "root_advance_m"),
        "left_cycle_duration_mean_s": _mean(left_cycles, "duration_s"),
        "right_cycle_duration_mean_s": _mean(right_cycles, "duration_s"),
        "left_cycle_length_last5_mean_m": _last_mean(left_cycles, "cycle_length_m", 5),
        "right_cycle_length_last5_mean_m": _last_mean(right_cycles, "cycle_length_m", 5),
        "left_cycle_root_advance_last5_mean_m": _last_mean(left_cycles, "root_advance_m", 5),
        "right_cycle_root_advance_last5_mean_m": _last_mean(right_cycles, "root_advance_m", 5),
        "left_cycle_duration_last5_mean_s": _last_mean(left_cycles, "duration_s", 5),
        "right_cycle_duration_last5_mean_s": _last_mean(right_cycles, "duration_s", 5),
        "left_right_cycle_duration_error_mean_s": _mean(left_cycles, "duration_s") - _mean(right_cycles, "duration_s"),
        "left_right_cycle_duration_error_last5_s": left_cycle_duration_last5 - right_cycle_duration_last5,
        "cycle_double_support_ratio_mean": _mean(cycles, "double_support_ratio"),
        "cycle_full_support_ratio_mean": _mean(cycles, "full_support_ratio"),
        "left_stance_duration_mean_s": _mean(left_cycles, "stance_duration_s"),
        "right_stance_duration_mean_s": _mean(right_cycles, "stance_duration_s"),
        "left_swing_duration_mean_s": _mean(left_cycles, "swing_duration_s"),
        "right_swing_duration_mean_s": _mean(right_cycles, "swing_duration_s"),
        "left_duty_factor_mean": _mean(left_cycles, "duty_factor"),
        "right_duty_factor_mean": _mean(right_cycles, "duty_factor"),
        "left_swing_ratio_mean": _mean(left_cycles, "swing_ratio"),
        "right_swing_ratio_mean": _mean(right_cycles, "swing_ratio"),
        "left_stance_duration_last5_mean_s": _last_mean(left_cycles, "stance_duration_s", 5),
        "right_stance_duration_last5_mean_s": _last_mean(right_cycles, "stance_duration_s", 5),
        "left_swing_duration_last5_mean_s": _last_mean(left_cycles, "swing_duration_s", 5),
        "right_swing_duration_last5_mean_s": _last_mean(right_cycles, "swing_duration_s", 5),
        "left_duty_factor_last5_mean": _last_mean(left_cycles, "duty_factor", 5),
        "right_duty_factor_last5_mean": _last_mean(right_cycles, "duty_factor", 5),
        "left_cycle_full_support_ratio_last5_mean": _last_mean(left_cycles, "full_support_ratio", 5),
        "right_cycle_full_support_ratio_last5_mean": _last_mean(right_cycles, "full_support_ratio", 5),
        **symmetry,
    }
    gates = {
        "speed_tracking": scorecard["speed_tracking_ratio"] >= args_cli.min_speed_tracking_ratio,
        "yaw_drift": abs(scorecard["yaw_drift_rad_per_m"]) <= args_cli.yaw_drift_per_meter_limit,
        "lateral_drift": abs(scorecard["lateral_drift_m_per_m"]) <= args_cli.lateral_drift_per_meter_limit,
        "root_roll_mean": abs(scorecard["root_roll_mean_5cycle"]) <= args_cli.torso_roll_mean_limit,
        "hip_roll_mean": scorecard["hip_roll_mean_abs_5cycle_rad"] <= args_cli.hip_roll_mean_limit,
        "alternating_steps": len(steps) >= 2 * args_cli.cycle_window,
        "airborne": scorecard["airborne_fraction"] <= 0.02,
        "root_height": scorecard["root_height_p05_m"] >= 0.50,
        "fsep_mean": scorecard["fsep_mean_m"] >= args_cli.min_fsep_mean,
        "fsep_p05": scorecard["fsep_p05_m"] >= args_cli.min_fsep_p05,
        "fsep_target_error": scorecard["fsep_target_error_mean_m"] <= args_cli.max_fsep_target_error_mean,
        "ksep_mean": scorecard["ksep_mean_m"] >= args_cli.min_ksep_mean,
        "cycle_cadence": scorecard["max_cycle_cadence_hz"] <= args_cli.max_cycle_cadence_hz,
        "step_root_advance": (not moving_command)
        or scorecard["step_root_advance_mean_m"] >= args_cli.min_step_root_advance_m,
        "cycle_root_advance": (not moving_command)
        or scorecard["cycle_root_advance_mean_m"] >= args_cli.min_cycle_root_advance_m,
    }
    decision = "APPROVE" if all(gates.values()) else "REJECT"
    if not all(gates.values()) and gates["alternating_steps"] and gates["yaw_drift"] and gates["lateral_drift"]:
        decision = "REVIEW_VIDEO"

    baseline_scorecard = _baseline_scorecard(args_cli.baseline_metrics)
    baseline_comparison = _baseline_comparison(scorecard, baseline_scorecard)

    summary = {
        "checkpoint": checkpoint_path,
        "task": args_cli.task,
        "dt": step_dt,
        "duration_s": args_cli.video_length * step_dt,
        "cycle_window": args_cli.cycle_window,
        "cycle_window_start_s": window_start,
        "joint_names": joint_names,
        "contact_quality_note": (
            "Using true heel/toe pad contacts."
            if use_pad_contacts
            else "No heel/toe/edge contact bodies are available; full_support metrics are orientation/contact proxies."
        ),
        "contact_body_names": contact_body_names,
        "uses_heel_toe_pads": use_pad_contacts,
        "metrics": {
            "vx": _summary(arrays["vx"].tolist()),
            "cmd_vx": _summary(arrays["cmd_vx"].tolist()),
            "vy": _summary(arrays["vy"].tolist()),
            "yaw_rate": _summary(arrays["yaw_rate"].tolist()),
            "root_height": _summary(arrays["root_z"].tolist()),
            "root_roll_proxy": _summary(arrays["root_roll_proxy"].tolist()),
            "left_swing_sole_clearance": _finite_summary(left_swing_clearance),
            "right_swing_sole_clearance": _finite_summary(right_swing_clearance),
            "fsep": _summary(arrays["fsep"].tolist()),
            "ksep": _summary(arrays["ksep"].tolist()),
        },
        "scorecard": scorecard,
        "baseline_metrics": args_cli.baseline_metrics,
        "baseline_comparison": baseline_comparison,
        "gates": gates,
        "decision": decision,
    }

    _write_csv(output_dir / "step_events.csv", events)
    _write_csv(output_dir / "gait_cycles.csv", cycles)
    _write_csv(output_dir / "steps.csv", steps)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# KBot Diagnostic Summary", "", f"Decision: {decision}", "", "## Scorecard"]
    lines.extend(f"- {key}: {value}" for key, value in scorecard.items())
    if baseline_comparison:
        lines.extend(["", "## Baseline Comparison"])
        for key, values in baseline_comparison.items():
            lines.append(
                f"- {key}: current={values['current']} baseline={values['baseline']} "
                f"delta={values['delta']} pct={values['percent_change']}"
            )
    lines.extend(["", "## Gates"])
    lines.extend(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in gates.items())
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "dashboard.html").write_text(_html(summary), encoding="utf-8")
    print(f"[INFO] Wrote diagnostics to: {output_dir}")
    print(f"[INFO] Decision: {decision}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
