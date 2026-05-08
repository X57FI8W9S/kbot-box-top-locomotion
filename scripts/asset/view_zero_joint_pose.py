#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = REPO_ROOT / "isaac_lab" / "IsaacLab"
KBOT_USD = REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top3.usd"

for path in (
    REPO_ROOT / "source" / "kbot_loco",
    ISAACLAB_ROOT / "source" / "isaaclab",
    ISAACLAB_ROOT / "source" / "isaaclab_assets",
    ISAACLAB_ROOT / "source" / "isaaclab_rl",
    ISAACLAB_ROOT / "source" / "isaaclab_tasks",
):
    sys.path.insert(0, str(path))

from isaaclab.app import AppLauncher  # noqa: E402


JOINT_NAMES = (
    "left_hip_pitch_04",
    "right_hip_pitch_04",
    "left_hip_roll_03",
    "right_hip_roll_03",
    "left_hip_yaw_03",
    "right_hip_yaw_03",
    "left_knee_04",
    "right_knee_04",
    "left_ankle_02",
    "right_ankle_02",
)


parser = argparse.ArgumentParser(description="View the KBot box-top robot with every controlled joint at zero.")
parser.add_argument("--env", action="store_true", help="Use the RL environment path instead of raw USD inspection.")
parser.add_argument("--root-height", type=float, default=1.05)
parser.add_argument("--camera", choices=("front", "side", "iso"), default="front")
parser.add_argument("--no-hold", action="store_true", help="Let physics run after reset instead of holding the zero pose.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import kbot_loco  # noqa: F401,E402
from kbot_loco.tasks.locomotion.env_cfg import KBotForwardFlatV2EnvCfg_PLAY  # noqa: E402


def _view_raw_usd() -> None:
    if not KBOT_USD.exists():
        raise FileNotFoundError(f"Missing robot USD: {KBOT_USD}")

    import omni.kit.app  # noqa: E402
    import omni.usd  # noqa: E402
    from pxr import Gf, UsdGeom  # noqa: E402

    context = omni.usd.get_context()
    opened = context.open_stage(str(KBOT_USD))
    if not opened:
        raise RuntimeError(f"Could not open USD stage: {KBOT_USD}")

    stage = context.get_stage()
    app = omni.kit.app.get_app()
    while stage is None:
        app.update()
        stage = context.get_stage()

    robot_prim = stage.GetDefaultPrim()
    if not robot_prim:
        children = list(stage.GetPseudoRoot().GetChildren())
        if not children:
            raise RuntimeError(f"No root prims found in USD stage: {KBOT_USD}")
        robot_prim = children[0]

    xformable = UsdGeom.Xformable(robot_prim)
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            translate_op = op
            break
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(0.0, 0.0, args.root_height))

    print("Viewing raw KBot USD at authored zero joint pose.", flush=True)
    print(f"USD: {KBOT_USD}", flush=True)
    print(f"Lifted prim {robot_prim.GetPath()} to z={args.root_height:.3f} m", flush=True)
    print("This path does not run the RL environment or step physics.", flush=True)
    print("Press Ctrl+C in this terminal when finished.", flush=True)

    try:
        while True:
            app.update()
    except KeyboardInterrupt:
        pass


def _disable_training_noise(cfg) -> None:
    cfg.observations.policy.enable_corruption = False
    cfg.terminations.low_body = None
    cfg.terminations.bad_orientation = None
    cfg.terminations.base_contact = None
    cfg.terminations.locked_knees = None
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
    cfg.events.push_robot = None
    cfg.events.base_external_force_torque = None
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.events.reset_base.params["velocity_range"] = {}
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)


def _set_camera(unwrapped, camera: str) -> None:
    if camera == "front":
        unwrapped.sim.set_camera_view(eye=(1.8, -3.0, 0.95), target=(0.0, 0.0, 0.45))
    elif camera == "side":
        unwrapped.sim.set_camera_view(eye=(3.0, 0.0, 0.9), target=(0.0, 0.0, 0.45))
    else:
        unwrapped.sim.set_camera_view(eye=(2.0, -2.2, 1.25), target=(0.0, 0.0, 0.45))


def _view_env_zero_pose() -> None:
    cfg = KBotForwardFlatV2EnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.scene.robot.init_state.pos = (0.0, 0.0, args.root_height)
    cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    cfg.scene.robot.init_state.joint_pos = {joint_name: 0.0 for joint_name in JOINT_NAMES}
    _disable_training_noise(cfg)

    env = gym.make("Isaac-KBot-Forward-Flat-V2-Play-v0", cfg=cfg, render_mode="human")
    env.reset()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    _set_camera(unwrapped, args.camera)

    zero_joint_pos = torch.zeros_like(robot.data.joint_pos)
    zero_joint_vel = torch.zeros_like(robot.data.joint_vel)
    root_pose = robot.data.root_pose_w.clone()
    root_pose[:, 0:3] = torch.tensor((0.0, 0.0, args.root_height), device=unwrapped.device)
    root_pose[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=unwrapped.device)
    root_velocity = torch.zeros((1, 6), device=unwrapped.device)
    zero_action = torch.zeros((1, unwrapped.action_manager.total_action_dim), device=unwrapped.device)

    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(root_velocity)
    robot.write_joint_state_to_sim(zero_joint_pos, zero_joint_vel)
    unwrapped.sim.forward()

    print("Viewing KBot zero-joint pose.")
    print(f"Root height: {args.root_height:.3f} m")
    print("All controlled joint targets are 0.0 rad:")
    for joint_name in robot.joint_names:
        value = float(robot.data.joint_pos[0, robot.joint_names.index(joint_name)].item())
        print(f"  {joint_name}: {value:+.4f}")
    print("Close the Isaac Sim window or press Ctrl+C to exit.")

    while simulation_app.is_running():
        if not args.no_hold:
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(root_velocity)
            robot.write_joint_state_to_sim(zero_joint_pos, zero_joint_vel)
        env.step(zero_action)

    env.close()


try:
    if args.env:
        _view_env_zero_pose()
    else:
        _view_raw_usd()
finally:
    simulation_app.close()
