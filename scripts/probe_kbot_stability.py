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
parser.add_argument("--left-hip-pitch", type=float, default=0.0)
parser.add_argument("--right-hip-pitch", type=float, default=0.0)
parser.add_argument("--left-knee", type=float, default=0.75)
parser.add_argument("--right-knee", type=float, default=-0.75)
parser.add_argument("--left-ankle", type=float, default=0.0)
parser.add_argument("--right-ankle", type=float, default=0.0)
parser.add_argument("--root-height", type=float, default=0.72)
parser.add_argument("--v2", action="store_true", help="Use the V2 play config instead of the V1 play config.")
parser.add_argument("--task-id", type=str, default=None, help="Gym task id to instantiate. Defaults to the V1/V2 play task.")
parser.add_argument("--use-task-defaults", action="store_true", help="Keep the task config's default root height and joint pose.")
parser.add_argument("--exact-reset", action="store_true", help="Disable reset pose, velocity, and joint-position noise.")
parser.add_argument("--usd-path", type=Path, default=None, help="Override the robot USD spawned by the selected task config.")
parser.add_argument("--decimation", type=int, default=None, help="Override task decimation for the probe.")
parser.add_argument("--disable-physics-material-event", action="store_true", help="Disable startup physics material randomization.")
parser.add_argument(
    "--action",
    type=float,
    nargs=10,
    default=None,
    metavar=("L_HIP_P", "R_HIP_P", "L_HIP_R", "R_HIP_R", "L_HIP_Y", "R_HIP_Y", "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE"),
    help="Constant normalized joint-position action to apply for the whole probe.",
)
parser.add_argument("--debug-reset", action="store_true", help="Print reset/action/actuator buffers before and after the first step.")
parser.add_argument(
    "--prime-default-targets",
    action="store_true",
    help="Set robot joint position targets to default joint positions immediately after reset.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import kbot_loco  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from kbot_loco.tasks.locomotion.env_cfg import KBotForwardFlatEnvCfg_PLAY, KBotForwardFlatV2EnvCfg_PLAY  # noqa: E402


def main() -> None:
    task_id = args.task_id
    cfg_cls = KBotForwardFlatV2EnvCfg_PLAY if args.v2 else KBotForwardFlatEnvCfg_PLAY
    if task_id is None:
        task_id = "Isaac-KBot-Forward-Flat-V2-Play-v0" if args.v2 else "Isaac-KBot-Forward-Flat-Play-v0"
        cfg = cfg_cls()
        config_name = cfg_cls.__name__
    else:
        cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
        config_name = type(cfg).__name__
    cfg.scene.num_envs = 1
    if args.decimation is not None:
        cfg.decimation = args.decimation
    if args.disable_physics_material_event:
        cfg.events.physics_material = None
    cfg.episode_length_s = max(cfg.episode_length_s, (args.steps + 10) * cfg.sim.dt * cfg.decimation)
    if args.usd_path is not None:
        cfg.scene.robot.spawn.usd_path = str(args.usd_path.resolve())
    if not args.use_task_defaults:
        cfg.scene.robot.init_state.pos = (0.0, 0.0, args.root_height)
        cfg.scene.robot.init_state.joint_pos["left_hip_pitch_04"] = args.left_hip_pitch
        cfg.scene.robot.init_state.joint_pos["right_hip_pitch_04"] = args.right_hip_pitch
        cfg.scene.robot.init_state.joint_pos["left_hip_roll_03"] = args.left_roll
        cfg.scene.robot.init_state.joint_pos["right_hip_roll_03"] = args.right_roll
        cfg.scene.robot.init_state.joint_pos["left_knee_04"] = args.left_knee
        cfg.scene.robot.init_state.joint_pos["right_knee_04"] = args.right_knee
        cfg.scene.robot.init_state.joint_pos["left_ankle_02"] = args.left_ankle
        cfg.scene.robot.init_state.joint_pos["right_ankle_02"] = args.right_ankle
    if args.exact_reset:
        cfg.events.reset_base.params["pose_range"] = {}
        cfg.events.reset_base.params["velocity_range"] = {}
        cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.terminations.low_body = None
    cfg.terminations.base_contact = None
    cfg.terminations.locked_knees = None

    env = gym.make(task_id, cfg=cfg)
    env.reset()
    unwrapped = env.unwrapped
    action = torch.zeros((1, unwrapped.action_manager.total_action_dim), device=unwrapped.device)
    if args.action is not None:
        action[:] = torch.tensor(args.action, device=unwrapped.device, dtype=action.dtype).reshape(1, -1)
    robot = unwrapped.scene["robot"]
    if args.prime_default_targets:
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.write_data_to_sim()
    if args.debug_reset:
        print("DEBUG joint_names=", robot.joint_names, flush=True)
        print("DEBUG default_joint_pos=", [round(float(v), 6) for v in robot.data.default_joint_pos[0].tolist()], flush=True)
        print("DEBUG reset_joint_pos=", [round(float(v), 6) for v in robot.data.joint_pos[0].tolist()], flush=True)
        print("DEBUG reset_joint_pos_target=", [round(float(v), 6) for v in robot.data.joint_pos_target[0].tolist()], flush=True)
        print(
            "DEBUG actuators=",
            {
                name: {
                    "class": type(actuator).__name__,
                    "joint_indices": list(actuator.joint_indices),
                    "stiffness": [round(float(v), 6) for v in actuator.stiffness[0].tolist()],
                    "damping": [round(float(v), 6) for v in actuator.damping[0].tolist()],
                }
                for name, actuator in robot.actuators.items()
            },
            flush=True,
        )
        for term_name, term in unwrapped.action_manager._terms.items():
            print(
                f"DEBUG action_term {term_name} class={type(term).__name__} "
                f"joint_ids={getattr(term, '_joint_ids', None)} "
                f"joint_names={getattr(term, '_joint_names', None)}",
                flush=True,
            )
            for attr in ("_scale", "_offset", "_raw_actions", "_processed_actions"):
                value = getattr(term, attr, None)
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().tolist()
                    print(f"DEBUG action_term {term_name} {attr}=", value, flush=True)
    heights = [float(robot.data.root_pos_w[0, 2].item())]
    rolls = [0.0]
    pitches = [0.0]
    for step_index in range(args.steps):
        env.step(action)
        robot = unwrapped.scene["robot"]
        if args.debug_reset and step_index == 0:
            print("DEBUG after_step1_joint_pos=", [round(float(v), 6) for v in robot.data.joint_pos[0].tolist()], flush=True)
            print("DEBUG after_step1_joint_pos_target=", [round(float(v), 6) for v in robot.data.joint_pos_target[0].tolist()], flush=True)
            print("DEBUG after_step1_root_pos=", [round(float(v), 6) for v in robot.data.root_pos_w[0].tolist()], flush=True)
            for term_name, term in unwrapped.action_manager._terms.items():
                value = getattr(term, "_processed_actions", None)
                if value is not None:
                    print(f"DEBUG action_term {term_name} after_step1_processed=", value.detach().cpu().tolist(), flush=True)
        heights.append(float(robot.data.root_pos_w[0, 2].item()))
        gravity = robot.data.projected_gravity_b[0]
        rolls.append(float(gravity[1].item()))
        pitches.append(float(gravity[0].item()))

    robot = unwrapped.scene["robot"]
    print(
        f"STABILITY task_id={task_id} config={config_name} steps={args.steps} "
        f"default_pose={args.use_task_defaults} root_height={cfg.scene.robot.init_state.pos[2]:.3f} "
        f"left_hip_pitch={args.left_hip_pitch:.3f} right_hip_pitch={args.right_hip_pitch:.3f} "
        f"left_knee={args.left_knee:.3f} right_knee={args.right_knee:.3f} "
        f"left_ankle={args.left_ankle:.3f} right_ankle={args.right_ankle:.3f} "
        f"action={args.action} "
        f"min_z={min(heights):.4f} final_z={heights[-1]:.4f} "
        f"max_abs_gravity_xy={max(max(abs(v) for v in rolls), max(abs(v) for v in pitches)):.4f}",
        flush=True,
    )
    print(
        "STABILITY body_z=",
        dict(zip(robot.body_names, [round(float(v), 4) for v in robot.data.body_pos_w[0, :, 2].tolist()])),
        flush=True,
    )
    print(
        "STABILITY body_xyz=",
        {
            name: [round(float(v), 4) for v in robot.data.body_pos_w[0, body_id].tolist()]
            for body_id, name in enumerate(robot.body_names)
        },
        flush=True,
    )
    print("STABILITY final_joint_pos=", [round(float(v), 4) for v in robot.data.joint_pos[0].tolist()], flush=True)
    env.close()


try:
    main()
finally:
    simulation_app.close()
