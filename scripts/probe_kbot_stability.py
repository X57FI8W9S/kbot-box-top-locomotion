#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = REPO_ROOT / "isaac_lab" / "IsaacLab"

for path in (
    REPO_ROOT / "source" / "kbot_loco",
    ISAACLAB_ROOT / "source" / "isaaclab",
    ISAACLAB_ROOT / "source" / "isaaclab_assets",
    ISAACLAB_ROOT / "source" / "isaaclab_rl",
    ISAACLAB_ROOT / "source" / "isaaclab_tasks",
):
    sys.path.insert(0, str(path))

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--left-roll", type=float, default=0.0)
parser.add_argument("--right-roll", type=float, default=0.0)
parser.add_argument("--left-knee", type=float, default=0.75)
parser.add_argument("--right-knee", type=float, default=-0.75)
parser.add_argument("--root-height", type=float, default=0.72)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import kbot_loco  # noqa: F401,E402
from kbot_loco.tasks.locomotion.env_cfg import KBotForwardFlatEnvCfg_PLAY  # noqa: E402


def main() -> None:
    cfg = KBotForwardFlatEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.scene.robot.init_state.pos = (0.0, 0.0, args.root_height)
    cfg.scene.robot.init_state.joint_pos["left_hip_roll_03"] = args.left_roll
    cfg.scene.robot.init_state.joint_pos["right_hip_roll_03"] = args.right_roll
    cfg.scene.robot.init_state.joint_pos["left_knee_04"] = args.left_knee
    cfg.scene.robot.init_state.joint_pos["right_knee_04"] = args.right_knee
    cfg.terminations.low_body = None
    cfg.terminations.base_contact = None
    cfg.terminations.locked_knees = None

    env = gym.make("Isaac-KBot-Forward-Flat-Play-v0", cfg=cfg)
    env.reset()
    unwrapped = env.unwrapped
    action = torch.zeros((1, unwrapped.action_manager.total_action_dim), device=unwrapped.device)
    robot = unwrapped.scene["robot"]
    heights = [float(robot.data.root_pos_w[0, 2].item())]
    rolls = [0.0]
    pitches = [0.0]
    for _ in range(args.steps):
        env.step(action)
        robot = unwrapped.scene["robot"]
        heights.append(float(robot.data.root_pos_w[0, 2].item()))
        gravity = robot.data.projected_gravity_b[0]
        rolls.append(float(gravity[1].item()))
        pitches.append(float(gravity[0].item()))

    robot = unwrapped.scene["robot"]
    print(f"min_z={min(heights):.4f} final_z={heights[-1]:.4f} max_abs_gravity_xy={max(max(abs(v) for v in rolls), max(abs(v) for v in pitches)):.4f}")
    print("body_z=", dict(zip(robot.body_names, [round(float(v), 4) for v in robot.data.body_pos_w[0, :, 2].tolist()])))
    print(
        "body_xyz=",
        {
            name: [round(float(v), 4) for v in robot.data.body_pos_w[0, body_id].tolist()]
            for body_id, name in enumerate(robot.body_names)
        },
    )
    print("final_joint_pos=", [round(float(v), 4) for v in robot.data.joint_pos[0].tolist()])
    env.close()


try:
    main()
finally:
    simulation_app.close()
