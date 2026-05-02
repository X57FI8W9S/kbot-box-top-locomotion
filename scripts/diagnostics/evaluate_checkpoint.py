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
parser.add_argument("--checkpoint", type=str, required=True)
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
parser.add_argument("--lateral_drift_per_meter_limit", type=float, default=0.20)
parser.add_argument("--min_timeout_fraction", type=float, default=1.0)
parser.add_argument("--min_speed_tracking_ratio", type=float, default=0.80)
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
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


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


def _rms_centered(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    centered = values - np.mean(values)
    return float(np.sqrt(np.mean(np.square(centered))))


def _joint_ids(joint_names: list[str], pattern: str) -> list[int]:
    return [i for i, name in enumerate(joint_names) if pattern in name]


def _contact_body_ids(contact_sensor: ContactSensor, names: tuple[str, str]) -> list[int]:
    body_names = getattr(contact_sensor, "body_names", None)
    if body_names is None:
        body_names = getattr(contact_sensor.data, "body_names", None)
    if body_names is not None:
        return [list(body_names).index(name) for name in names]
    return [0, 1]


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


def _events(time: np.ndarray, contact: np.ndarray, foot_positions: np.ndarray) -> tuple[list[dict], list[dict]]:
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
                "from_foot": prev["foot"],
                "to_foot": current["foot"],
                "start_time": prev["time"],
                "end_time": current["time"],
                "duration_s": current["time"] - prev["time"],
                "step_length_m": current["x"] - prev["x"],
                "step_width_m": abs(current["y"] - prev["y"]),
                "start_frame": prev["frame"],
                "end_frame": current["frame"],
            }
        )
    return events, steps


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
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, str(installed_version))
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
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    contact_sensor: ContactSensor = env.unwrapped.scene.sensors["contact_forces"]
    contact_body_ids = _contact_body_ids(contact_sensor, ("foot1", "foot3"))
    command_manager = getattr(env.unwrapped, "command_manager", None)
    joint_names = list(robot.data.joint_names)
    hip_roll_ids = _joint_ids(joint_names, "hip_roll")
    hip_yaw_ids = _joint_ids(joint_names, "hip_yaw")
    knee_ids = _joint_ids(joint_names, "knee")

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
        "foot_pos": [],
        "foot_up_z": [],
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
        contact = contact_time > 0.0
        foot_body_ids = [robot.body_names.index("foot1"), robot.body_names.index("foot3")]
        foot_pos = robot.data.body_pos_w[0, foot_body_ids].detach().cpu().numpy()
        foot_quat = robot.data.body_quat_w[0, foot_body_ids]
        up_b = torch.zeros(2, 3, device=foot_quat.device)
        up_b[:, 2] = 1.0
        up_z = quat_apply(foot_quat, up_b)[:, 2].detach().cpu().numpy()
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
        series["foot_pos"].append(foot_pos)
        series["foot_up_z"].append(up_z)
        series["stance_slip"].append(stance_slip)

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if installed_version >= version.parse("4.0.0"):
                policy.reset(dones)

    arrays = {key: np.asarray(value) for key, value in series.items()}
    events, steps = _events(arrays["time"], arrays["contact"].astype(bool), arrays["foot_pos"])
    window_start = _cycle_window_start(events, args_cli.cycle_window, step_dt)
    window_mask = arrays["time"] >= window_start
    joint_window = arrays["joint_pos"][window_mask]
    root_roll_window = arrays["root_roll_proxy"][window_mask]
    hip_roll_window = joint_window[:, hip_roll_ids] if hip_roll_ids else np.zeros((0, 1))
    hip_yaw_window = joint_window[:, hip_yaw_ids] if hip_yaw_ids else np.zeros((0, 1))
    knee_window = np.abs(joint_window[:, knee_ids]) if knee_ids else np.zeros((0, 1))
    contact = arrays["contact"].astype(bool)
    double_support = contact[:, 0] & contact[:, 1]
    airborne = ~contact[:, 0] & ~contact[:, 1]
    full_sole_proxy = contact & (arrays["foot_up_z"] > 0.96)
    distance = max(float(arrays["root_x"][-1] - arrays["root_x"][0]), 1.0e-6)
    yaw_drift = float(np.trapz(arrays["yaw_rate"], arrays["time"]))
    lateral_drift = float(arrays["root_y"][-1] - arrays["root_y"][0])
    speed_ratio = float(np.mean(arrays["vx"]) / max(np.mean(arrays["cmd_vx"]), 1.0e-6))

    step_lengths = [row["step_length_m"] for row in steps]
    step_durations = [row["duration_s"] for row in steps]
    symmetry = _paired_joint_symmetry(joint_names, joint_window) if joint_window.size else {}
    scorecard = {
        "distance_m": distance,
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
        "double_support_fraction": float(np.mean(double_support)),
        "airborne_fraction": float(np.mean(airborne)),
        "full_sole_proxy_fraction_left": float(np.mean(full_sole_proxy[:, 0])),
        "full_sole_proxy_fraction_right": float(np.mean(full_sole_proxy[:, 1])),
        "stance_slip_mean_mps": float(np.mean(arrays["stance_slip"])),
        "step_count": len(steps),
        "step_length_mean_m": float(np.mean(step_lengths)) if step_lengths else 0.0,
        "step_duration_mean_s": float(np.mean(step_durations)) if step_durations else 0.0,
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
    }
    decision = "APPROVE" if all(gates.values()) else "REJECT"
    if not all(gates.values()) and gates["alternating_steps"] and gates["yaw_drift"] and gates["lateral_drift"]:
        decision = "REVIEW_VIDEO"

    summary = {
        "checkpoint": checkpoint_path,
        "task": args_cli.task,
        "dt": step_dt,
        "duration_s": args_cli.video_length * step_dt,
        "cycle_window": args_cli.cycle_window,
        "cycle_window_start_s": window_start,
        "joint_names": joint_names,
        "contact_quality_note": "No heel/toe/edge contact bodies are available; full_sole metrics are orientation/contact proxies.",
        "metrics": {
            "vx": _summary(arrays["vx"].tolist()),
            "cmd_vx": _summary(arrays["cmd_vx"].tolist()),
            "vy": _summary(arrays["vy"].tolist()),
            "yaw_rate": _summary(arrays["yaw_rate"].tolist()),
            "root_height": _summary(arrays["root_z"].tolist()),
            "root_roll_proxy": _summary(arrays["root_roll_proxy"].tolist()),
        },
        "scorecard": scorecard,
        "gates": gates,
        "decision": decision,
    }

    _write_csv(output_dir / "step_events.csv", events)
    _write_csv(output_dir / "gait_cycles.csv", steps)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# KBot Diagnostic Summary", "", f"Decision: {decision}", "", "## Scorecard"]
    lines.extend(f"- {key}: {value}" for key, value in scorecard.items())
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
