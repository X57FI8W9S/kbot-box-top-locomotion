#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
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


HAND_POSE_DEG = {
    "left_hip_pitch_04": 17.0,
    "right_hip_pitch_04": -17.0,
    "left_hip_roll_03": 0.0,
    "right_hip_roll_03": 0.0,
    "left_hip_yaw_03": 0.0,
    "right_hip_yaw_03": 0.0,
    "left_knee_04": 29.5,
    "right_knee_04": -29.5,
    "left_ankle_02": -12.0,
    "right_ankle_02": 12.0,
}

RAW_USD_SETTLED_POSE_RAD = {
    "left_hip_pitch_04": 0.2843153178691864,
    "right_hip_pitch_04": -0.2841152250766754,
    "left_hip_roll_03": 0.0017389939166605473,
    "right_hip_roll_03": 0.0019064429216086864,
    "left_hip_yaw_03": 0.0013319215504452586,
    "right_hip_yaw_03": 0.00043546810047701,
    "left_knee_04": 0.5073038935661316,
    "right_knee_04": -0.5059521198272705,
    "left_ankle_02": -0.24602758884429932,
    "right_ankle_02": 0.24722331762313843,
}

RAW_USD_INITIAL_POSE_RAD = {
    "left_hip_pitch_04": 0.15701022744178772,
    "right_hip_pitch_04": -0.1570812463760376,
    "left_hip_roll_03": 0.0005932870553806424,
    "right_hip_roll_03": 0.0006502520409412682,
    "left_hip_yaw_03": 0.0003773318021558225,
    "right_hip_yaw_03": -0.000575827609281987,
    "left_knee_04": 0.272339403629303,
    "right_knee_04": -0.2722611427307129,
    "left_ankle_02": -0.1372131109237671,
    "right_ankle_02": 0.1399385631084442,
}


parser = argparse.ArgumentParser(description="Probe KBot passive/held-pose stability outside the locomotion task.")
parser.add_argument("--usd-path", type=Path, default=REPO_ROOT / "kbot3_2.usd")
parser.add_argument("--pose", choices=("hand", "raw-usd-initial", "raw-usd-settled"), default="hand")
parser.add_argument(
    "--target-pose",
    choices=("hand", "raw-usd-initial", "raw-usd-settled"),
    default=None,
    help="Joint target pose. Defaults to --pose.",
)
parser.add_argument("--actuator", choices=("dc", "implicit"), default="dc")
parser.add_argument("--root-height", type=float, default=0.88)
parser.add_argument("--root-x", type=float, default=0.0)
parser.add_argument("--root-y", type=float, default=0.0)
parser.add_argument(
    "--root-rpy-deg",
    type=float,
    nargs=3,
    metavar=("ROLL", "PITCH", "YAW"),
    default=None,
    help="Override root orientation with XYZ Euler angles in degrees.",
)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--hold-target", action="store_true", help="Continuously command the hand pose as joint position target.")
parser.add_argument("--target-left-ankle", type=float, default=None, help="Override target left ankle position in radians.")
parser.add_argument("--target-right-ankle", type=float, default=None, help="Override target right ankle position in radians.")
parser.add_argument("--realtime", action="store_true", help="Throttle the loop to roughly realtime for GUI inspection.")
parser.add_argument("--hold-open", action="store_true", help="Keep the Isaac window open after the probe finishes.")
parser.add_argument(
    "--preserve-root-orientation",
    action="store_true",
    help="Keep the USD-authored floating-base orientation instead of resetting it to identity.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402


def _quat_from_xyz_euler_deg(roll_deg: float, pitch_deg: float, yaw_deg: float, device: str) -> torch.Tensor:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return torch.tensor(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        device=device,
    )


def _joint_pose_rad(name: str) -> dict[str, float]:
    if name == "raw-usd-settled":
        return RAW_USD_SETTLED_POSE_RAD
    if name == "raw-usd-initial":
        return RAW_USD_INITIAL_POSE_RAD
    return {name: math.radians(value) for name, value in HAND_POSE_DEG.items()}


def _make_robot_cfg(usd_path: Path) -> ArticulationCfg:
    if args.actuator == "implicit":
        actuators = {
            "hip_pitch_knee": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_pitch.*", ".*knee.*"],
                effort_limit_sim=120.0,
                velocity_limit_sim=6.283,
                stiffness={".*": 45.0},
                damping={".*": 4.0},
            ),
            "hip_roll": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_roll.*"],
                effort_limit_sim=60.0,
                velocity_limit_sim=6.283,
                stiffness={".*": 35.0},
                damping={".*": 3.0},
            ),
            "hip_yaw": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_yaw.*"],
                effort_limit_sim=60.0,
                velocity_limit_sim=6.283,
                stiffness={".*": 25.0},
                damping={".*": 2.0},
            ),
            "ankles": ImplicitActuatorCfg(
                joint_names_expr=[".*ankle.*"],
                effort_limit_sim=17.0,
                velocity_limit_sim=12.566,
                stiffness={".*": 12.0},
                damping={".*": 1.0},
            ),
        }
    else:
        actuators = {
            "hip_pitch_knee": DCMotorCfg(
                joint_names_expr=[".*hip_pitch.*", ".*knee.*"],
                effort_limit=120.0,
                saturation_effort=120.0,
                velocity_limit=6.283,
                stiffness={".*": 45.0},
                damping={".*": 4.0},
            ),
            "hip_roll": DCMotorCfg(
                joint_names_expr=[".*hip_roll.*"],
                effort_limit=60.0,
                saturation_effort=60.0,
                velocity_limit=6.283,
                stiffness={".*": 35.0},
                damping={".*": 3.0},
            ),
            "hip_yaw": DCMotorCfg(
                joint_names_expr=[".*hip_yaw.*"],
                effort_limit=60.0,
                saturation_effort=60.0,
                velocity_limit=6.283,
                stiffness={".*": 25.0},
                damping={".*": 2.0},
            ),
            "ankles": DCMotorCfg(
                joint_names_expr=[".*ankle.*"],
                effort_limit=17.0,
                saturation_effort=17.0,
                velocity_limit=12.566,
                stiffness={".*": 12.0},
                damping={".*": 1.0},
            ),
        }
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path.resolve()),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, args.root_height),
            joint_pos=_joint_pose_rad(args.pose),
        ),
        actuators=actuators,
    )


def _set_hand_pose(robot: Articulation, device: str) -> torch.Tensor:
    joint_pos = torch.zeros((1, robot.num_joints), device=device)
    initial_pose = _joint_pose_rad(args.pose)
    target_pose = _joint_pose_rad(args.target_pose or args.pose)
    if args.target_left_ankle is not None:
        target_pose["left_ankle_02"] = args.target_left_ankle
    if args.target_right_ankle is not None:
        target_pose["right_ankle_02"] = args.target_right_ankle
    for joint_name, value_rad in initial_pose.items():
        if joint_name not in robot.joint_names:
            raise RuntimeError(f"Joint {joint_name!r} is missing. Available joints: {robot.joint_names}")
        joint_pos[0, robot.joint_names.index(joint_name)] = value_rad
    joint_vel = torch.zeros_like(joint_pos)
    target_joint_pos = torch.zeros_like(joint_pos)
    for joint_name, value_rad in target_pose.items():
        target_joint_pos[0, robot.joint_names.index(joint_name)] = value_rad
    if args.root_rpy_deg is not None:
        root_quat = _quat_from_xyz_euler_deg(*args.root_rpy_deg, device=device)
    elif args.preserve_root_orientation:
        root_quat = robot.data.root_quat_w[0].clone()
    else:
        root_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    root_pose = torch.zeros((1, 7), device=device)
    root_pose[0, :3] = torch.tensor([args.root_x, args.root_y, args.root_height], device=device)
    root_pose[0, 3:] = root_quat
    root_vel = torch.zeros((1, 6), device=device)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(root_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(target_joint_pos)
    robot.write_data_to_sim()
    return target_joint_pos


def main() -> None:
    usd_path = args.usd_path.resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"USD path does not exist: {usd_path}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    sim.set_camera_view([2.0, -2.5, 1.2], [0.0, 0.0, 0.5])
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000)
    light_cfg.func("/World/Light/DomeLight", light_cfg)

    robot = Articulation(cfg=_make_robot_cfg(usd_path))
    sim.reset()
    sim_dt = sim.get_physics_dt()
    robot.update(sim_dt)
    print("ARTICULATION joint_names=", list(robot.joint_names), flush=True)
    print("ARTICULATION actuator_groups=", {name: list(actuator.joint_names) for name, actuator in robot.actuators.items()}, flush=True)
    target_joint_pos = _set_hand_pose(robot, sim.device)
    print(
        "ARTICULATION target_joint_pos=",
        dict(zip(robot.joint_names, [round(float(v), 4) for v in target_joint_pos[0].tolist()])),
        flush=True,
    )
    sim.step(render=not args.headless)
    robot.update(sim_dt)

    heights = [float(robot.data.root_pos_w[0, 2].item())]
    tilt_proxy = [0.0]
    for _ in range(args.steps):
        if args.hold_target:
            robot.set_joint_position_target(target_joint_pos)
            robot.write_data_to_sim()
        sim.step(render=not args.headless)
        if args.realtime:
            time.sleep(sim_dt)
        robot.update(sim_dt)
        heights.append(float(robot.data.root_pos_w[0, 2].item()))
        gravity = robot.data.projected_gravity_b[0]
        tilt_proxy.append(max(abs(float(gravity[0].item())), abs(float(gravity[1].item()))))

    body_z = dict(zip(robot.body_names, [round(float(v), 4) for v in robot.data.body_pos_w[0, :, 2].tolist()]))
    body_xyz = {
        name: [round(float(v), 4) for v in robot.data.body_pos_w[0, body_id].tolist()]
        for body_id, name in enumerate(robot.body_names)
    }
    print(
        f"ARTICULATION_STABILITY usd={usd_path} steps={args.steps} "
        f"pose={args.pose} target_pose={args.target_pose or args.pose} actuator={args.actuator} "
        f"root_pos=({args.root_x:.4f},{args.root_y:.4f},{args.root_height:.4f}) root_rpy_deg={args.root_rpy_deg} "
        f"hold_target={args.hold_target} preserve_root_orientation={args.preserve_root_orientation} "
        f"min_z={min(heights):.4f} final_z={heights[-1]:.4f} "
        f"max_abs_gravity_xy={max(tilt_proxy):.4f}",
        flush=True,
    )
    print("ARTICULATION body_z=", body_z, flush=True)
    print("ARTICULATION body_xyz=", body_xyz, flush=True)
    print(
        "ARTICULATION final_joint_pos=",
        [round(float(v), 4) for v in robot.data.joint_pos[0].tolist()],
        flush=True,
    )
    if args.hold_open:
        while simulation_app.is_running():
            sim.step(render=not args.headless)
            if args.realtime:
                time.sleep(sim_dt)


try:
    main()
finally:
    simulation_app.close()
