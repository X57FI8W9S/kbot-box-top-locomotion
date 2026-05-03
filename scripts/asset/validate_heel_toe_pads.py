#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
parser.add_argument("--steps", type=int, default=120)
parser.add_argument(
    "--scenario",
    choices=["flat_default", "toe_guess", "heel_guess", "air_high_start", "all"],
    default="flat_default",
)
parser.add_argument("--fix-root", action="store_true", help="Fix the root link for static contact debugging.")
parser.add_argument("--hold-pose", action="store_true", help="Rewrite root and joint state before each step.")
parser.add_argument("--root-height", type=float, default=None, help="Override scenario root height.")
parser.add_argument("--left-ankle", type=float, default=None, help="Override scenario left ankle angle in rad.")
parser.add_argument("--right-ankle", type=float, default=None, help="Override scenario right ankle angle in rad.")
parser.add_argument("--root-pitch", type=float, default=0.0, help="Held root pitch angle in rad.")
parser.add_argument("--scan", action="store_true", help="Scan held root heights and pitches in one environment.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import kbot_loco  # noqa: F401,E402
from kbot_loco.tasks.locomotion.env_cfg import KBotForwardFlatV2EnvCfg_PLAY  # noqa: E402


PAD_NAMES = ["left_heel_pad", "left_toe_pad", "right_heel_pad", "right_toe_pad"]


def _pad_ids(sensor, names: list[str]) -> list[int]:
    body_names = list(getattr(sensor, "body_names", None) or getattr(sensor.data, "body_names", None) or [])
    missing = [name for name in names if name not in body_names]
    if missing:
        raise RuntimeError(f"Missing contact sensor bodies {missing}. Available bodies: {body_names}")
    return [body_names.index(name) for name in names]


def run_scenario(name: str, *, root_height: float, left_ankle: float, right_ankle: float, steps: int) -> dict[str, float]:
    if args.root_height is not None:
        root_height = args.root_height
    if args.left_ankle is not None:
        left_ankle = args.left_ankle
    if args.right_ankle is not None:
        right_ankle = args.right_ankle
    cfg = KBotForwardFlatV2EnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.scene.robot.init_state.pos = (0.0, 0.0, root_height)
    cfg.scene.robot.init_state.joint_pos["left_ankle_02"] = left_ankle
    cfg.scene.robot.init_state.joint_pos["right_ankle_02"] = right_ankle
    if args.fix_root:
        cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    cfg.terminations.low_body = None
    cfg.terminations.bad_orientation = None
    cfg.terminations.base_contact = None
    cfg.terminations.locked_knees = None
    cfg.events.add_base_mass = None
    cfg.events.base_com = None

    env = gym.make("Isaac-KBot-Forward-Flat-V2-Play-v0", cfg=cfg, render_mode=None)
    env.reset()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    sensor = unwrapped.scene.sensors["contact_forces"]
    ids = _pad_ids(sensor, PAD_NAMES)

    print(f"\n{name}", flush=True)
    print(f"  robot bodies include pads: {all(pad in robot.body_names for pad in PAD_NAMES)}", flush=True)
    print(f"  pad body ids: {dict(zip(PAD_NAMES, ids))}", flush=True)
    debug_body_names = ["foot1", "foot3", *PAD_NAMES]
    debug_positions = {}
    for body_name in debug_body_names:
        if body_name in robot.body_names:
            body_id = robot.body_names.index(body_name)
            debug_positions[body_name] = [round(float(v), 4) for v in robot.data.body_pos_w[0, body_id].tolist()]
    print("  body_pos_w_at_reset:", debug_positions, flush=True)

    action = torch.zeros((1, unwrapped.action_manager.total_action_dim), device=unwrapped.device)
    root_pose = robot.data.root_pose_w.clone()
    root_pose[:, 0:3] = torch.tensor((0.0, 0.0, root_height), device=unwrapped.device)
    half_pitch = 0.5 * args.root_pitch
    root_pose[:, 3:7] = torch.tensor(
        (math.cos(half_pitch), 0.0, math.sin(half_pitch), 0.0),
        device=unwrapped.device,
    )
    root_velocity = torch.zeros((1, 6), device=unwrapped.device)
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.joint_vel)
    joint_name_to_id = {joint_name: i for i, joint_name in enumerate(robot.data.joint_names)}
    joint_pos[:, joint_name_to_id["left_ankle_02"]] = left_ankle
    joint_pos[:, joint_name_to_id["right_ankle_02"]] = right_ankle
    def set_pose(height: float, pitch: float) -> None:
        root_pose[:, 0:3] = torch.tensor((0.0, 0.0, height), device=unwrapped.device)
        half_pitch_local = 0.5 * pitch
        root_pose[:, 3:7] = torch.tensor(
            (math.cos(half_pitch_local), 0.0, math.sin(half_pitch_local), 0.0),
            device=unwrapped.device,
        )

    def sample_contact(sample_steps: int) -> tuple[torch.Tensor, list[float]]:
        contacts_local = []
        for _ in range(sample_steps):
            if args.hold_pose or args.scan:
                robot.write_root_pose_to_sim(root_pose)
                robot.write_root_velocity_to_sim(root_velocity)
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
            env.step(action)
            contact_time = sensor.data.current_contact_time[0, ids].detach().cpu()
            contacts_local.append((contact_time > 0.0).float())
        forces_local = sensor.data.net_forces_w[0, ids].norm(dim=-1).detach().cpu().tolist()
        return torch.stack(contacts_local, dim=0), forces_local

    set_pose(root_height, args.root_pitch)
    if args.scan:
        print("  scan rows: height pitch contact_fraction final_force_N", flush=True)
        for height in (0.68, 0.70, 0.72, 0.74, 0.76):
            for pitch in (-0.8, -0.4, 0.0, 0.4, 0.8):
                set_pose(height, pitch)
                contact_tensor, forces = sample_contact(max(6, min(steps, 12)))
                fractions = contact_tensor.mean(dim=0).tolist()
                print(
                    f"  scan h={height:.2f} pitch={pitch:+.2f} "
                    f"frac={{{', '.join(f'{k}:{v:.2f}' for k, v in zip(PAD_NAMES, fractions))}}} "
                    f"force={{{', '.join(f'{k}:{v:.1f}' for k, v in zip(PAD_NAMES, forces))}}}",
                    flush=True,
                )
        env.close()
        return {}

    contact_tensor, forces = sample_contact(steps)
    fractions = contact_tensor.mean(dim=0).tolist()
    result = dict(zip(PAD_NAMES, fractions))
    print("  contact_fraction:", {k: round(v, 3) for k, v in result.items()}, flush=True)
    print("  final_force_N:", {k: round(v, 3) for k, v in zip(PAD_NAMES, forces)}, flush=True)
    print("  root_z_m:", round(float(robot.data.root_pos_w[0, 2].item()), 4), flush=True)
    env.close()
    return result


def main() -> None:
    scenarios_by_name = {
        "flat_default": ("flat_default", 0.78, 0.0, 0.0),
        "toe_guess": ("toe_guess", 0.78, 0.45, -0.45),
        "heel_guess": ("heel_guess", 0.78, -0.45, 0.45),
        "air_high_start": ("air_high_start", 1.10, 0.0, 0.0),
    }
    scenarios = list(scenarios_by_name.values()) if args.scenario == "all" else [scenarios_by_name[args.scenario]]
    for name, root_height, left_ankle, right_ankle in scenarios:
        run_scenario(name, root_height=root_height, left_ankle=left_ankle, right_ankle=right_ankle, steps=args.steps)


try:
    main()
finally:
    simulation_app.close()
