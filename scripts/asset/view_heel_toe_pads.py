#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
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
parser.add_argument("--root-height", type=float, default=0.88)
parser.add_argument("--hold-pose", action="store_true", help="Keep the root and joints in the initial pose.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import kbot_loco  # noqa: F401,E402
from kbot_loco.tasks.locomotion.env_cfg import KBotForwardFlatV2EnvCfg_PLAY  # noqa: E402


def main() -> None:
    cfg = KBotForwardFlatV2EnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.scene.robot.init_state.pos = (0.0, 0.0, args.root_height)
    cfg.terminations.low_body = None
    cfg.terminations.bad_orientation = None
    cfg.terminations.base_contact = None
    cfg.terminations.locked_knees = None
    cfg.events.add_base_mass = None
    cfg.events.base_com = None

    env = gym.make("Isaac-KBot-Forward-Flat-V2-Play-v0", cfg=cfg, render_mode="human")
    env.reset()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    action = torch.zeros((1, unwrapped.action_manager.total_action_dim), device=unwrapped.device)

    root_pose = robot.data.root_pose_w.clone()
    root_pose[:, 0:3] = torch.tensor((0.0, 0.0, args.root_height), device=unwrapped.device)
    root_pose[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=unwrapped.device)
    root_velocity = torch.zeros((1, 6), device=unwrapped.device)
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.joint_vel)

    print("Loaded V2 pad asset.")
    print("Pads are colored blue=heel and orange=toe.")
    print("Close the Isaac Sim window or press Ctrl+C in the terminal to exit.")

    while simulation_app.is_running():
        if args.hold_pose:
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(root_velocity)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
        env.step(action)

    env.close()


try:
    main()
finally:
    simulation_app.close()
