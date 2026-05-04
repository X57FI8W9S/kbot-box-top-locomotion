#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = REPO_ROOT / "isaac_lab" / "IsaacLab"
KBOT_PADS_USD = REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top3_pads.usd"

for path in (
    REPO_ROOT / "source" / "kbot_loco",
    ISAACLAB_ROOT / "source" / "isaaclab",
    ISAACLAB_ROOT / "source" / "isaaclab_assets",
    ISAACLAB_ROOT / "source" / "isaaclab_rl",
    ISAACLAB_ROOT / "source" / "isaaclab_tasks",
):
    sys.path.insert(0, str(path))

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description="Open Isaac Sim and view the kbot_box_top3_pads.usd robot.")
parser.add_argument("--env", action="store_true", help="View the robot through the RL environment instead of raw USD.")
parser.add_argument("--root-height", type=float, default=0.88)
parser.add_argument("--hold-pose", action="store_true", help="Freeze the root and joints in the initial pose.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

def _view_raw_usd() -> None:
    if not KBOT_PADS_USD.exists():
        raise FileNotFoundError(f"Missing robot USD: {KBOT_PADS_USD}")

    import omni.kit.app  # noqa: E402
    import omni.usd  # noqa: E402

    opened = omni.usd.get_context().open_stage(str(KBOT_PADS_USD))
    if not opened:
        raise RuntimeError(f"Could not open USD stage: {KBOT_PADS_USD}")

    print(f"Viewing raw robot USD: {KBOT_PADS_USD}")
    print(f"Using IsaacLab root: {ISAACLAB_ROOT}")
    print("Close the Isaac Sim window or press Ctrl+C to exit.")

    while simulation_app.is_running():
        omni.kit.app.get_app().update()


def _view_env() -> None:
    if not KBOT_PADS_USD.exists():
        raise FileNotFoundError(f"Missing robot USD: {KBOT_PADS_USD}")

    import gymnasium as gym  # noqa: E402
    import torch  # noqa: E402

    import isaaclab_tasks  # noqa: F401,E402
    import kbot_loco  # noqa: F401,E402
    from kbot_loco.tasks.locomotion.assets import KBOT_PADS_CFG  # noqa: E402
    from kbot_loco.tasks.locomotion.env_cfg import KBotForwardFlatV2EnvCfg_PLAY  # noqa: E402

    cfg = KBotForwardFlatV2EnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.scene.robot = KBOT_PADS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.robot.init_state.pos = (0.0, 0.0, args.root_height)
    cfg.observations.policy.enable_corruption = False
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
    unwrapped.sim.set_camera_view(eye=(1.4, -2.0, 1.0), target=(0.0, 0.0, 0.45))

    zero_action = torch.zeros((1, unwrapped.action_manager.total_action_dim), device=unwrapped.device)
    root_pose = robot.data.root_pose_w.clone()
    root_pose[:, 0:3] = torch.tensor((0.0, 0.0, args.root_height), device=unwrapped.device)
    root_velocity = torch.zeros((1, 6), device=unwrapped.device)
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.joint_vel)

    print(f"Viewing robot USD: {KBOT_PADS_USD}")
    print(f"Using IsaacLab root: {ISAACLAB_ROOT}")
    print(f"Root position: {[round(float(v), 4) for v in robot.data.root_pos_w[0].tolist()]}")
    print("Close the Isaac Sim window or press Ctrl+C to exit.")

    while simulation_app.is_running():
        if args.hold_pose:
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(root_velocity)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
        env.step(zero_action)

    env.close()


def main() -> None:
    if args.env:
        _view_env()
    else:
        _view_raw_usd()


try:
    main()
finally:
    simulation_app.close()
