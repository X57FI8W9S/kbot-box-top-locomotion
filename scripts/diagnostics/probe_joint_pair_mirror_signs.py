#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


DEFAULT_POSE_RAD = {
    "left_hip_pitch_04": 0.0,
    "right_hip_pitch_04": 0.0,
    "left_hip_roll_03": 0.0,
    "right_hip_roll_03": 0.0,
    "left_hip_yaw_03": 0.0,
    "right_hip_yaw_03": 0.0,
    "left_knee_04": 0.75,
    "right_knee_04": -0.75,
    "left_ankle_02": 0.0,
    "right_ankle_02": 0.0,
}

SETTLED_POSE_RAD = {
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

JOINT_PAIRS = (
    ("left_hip_pitch_04", "right_hip_pitch_04"),
    ("left_hip_roll_03", "right_hip_roll_03"),
    ("left_hip_yaw_03", "right_hip_yaw_03"),
    ("left_knee_04", "right_knee_04"),
    ("left_ankle_02", "right_ankle_02"),
)

BODY_PAIRS = (
    ("leg0_shell", "leg0_shell_2"),
    ("leg1_shell", "leg1_shell3"),
    ("leg2_shell", "leg2_shell_2"),
    ("leg3_shell1", "leg3_shell11"),
    ("foot1", "foot3"),
)


parser = argparse.ArgumentParser(
    description=(
        "Perturb KBot left/right joint pairs and compare +1 vs -1 mirror signs "
        "using body-frame mirrored link motion."
    )
)
parser.add_argument("--usd-path", type=Path, default=REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top3.usd")
parser.add_argument("--pose", choices=("default", "settled"), default="default")
parser.add_argument("--delta-deg", type=float, default=8.0)
parser.add_argument("--root-height", type=float, default=0.88)
parser.add_argument(
    "--orientation-weight-m",
    type=float,
    default=0.05,
    help="Meters of combined score per unit RMS axis-vector mirror error.",
)
parser.add_argument("--json-output", type=Path, default=None)
parser.add_argument("--plot-dir", type=Path, default=None, help="Optional directory for PNG end-pose plots.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.actuators import DCMotorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402


def _make_robot_cfg(usd_path: Path) -> ArticulationCfg:
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
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, args.root_height),
            joint_pos=SETTLED_POSE_RAD if args.pose == "settled" else DEFAULT_POSE_RAD,
        ),
        actuators=actuators,
    )


def _write_state(robot: Articulation, sim: sim_utils.SimulationContext, joint_pos: torch.Tensor) -> None:
    joint_vel = torch.zeros_like(joint_pos)
    root_pose = torch.zeros((1, 7), dtype=joint_pos.dtype, device=joint_pos.device)
    root_pose[0, :3] = torch.tensor((0.0, 0.0, args.root_height), dtype=joint_pos.dtype, device=joint_pos.device)
    root_pose[0, 3] = 1.0
    root_vel = torch.zeros((1, 6), dtype=joint_pos.dtype, device=joint_pos.device)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(root_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()
    sim.step(render=not args.headless)
    robot.update(sim.get_physics_dt())


def _capture_body_state(robot: Articulation, body_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    body_pos_w = robot.data.body_pos_w[0, body_ids]
    body_quat_w = robot.data.body_quat_w[0, body_ids]
    root_pos_w = robot.data.root_pos_w[0]
    root_quat_w = robot.data.root_quat_w[0]
    pos_b = quat_apply_inverse(root_quat_w.expand(len(body_ids), -1), body_pos_w - root_pos_w)

    basis = torch.eye(3, dtype=body_pos_w.dtype, device=body_pos_w.device)
    basis = basis[None, :, :].expand(len(body_ids), -1, -1)
    quat = body_quat_w[:, None, :].expand(-1, 3, -1).reshape(-1, 4)
    axes_w = quat_apply(quat, basis.reshape(-1, 3)).reshape(len(body_ids), 3, 3)
    root_quat = root_quat_w[None, None, :].expand(len(body_ids), 3, -1).reshape(-1, 4)
    axes_b = quat_apply_inverse(root_quat, axes_w.reshape(-1, 3)).reshape(len(body_ids), 3, 3)
    return pos_b, axes_b


def _mirrored_delta_errors(
    base_pos: torch.Tensor,
    base_axes: torch.Tensor,
    cand_pos: torch.Tensor,
    cand_axes: torch.Tensor,
    pair_indices: list[tuple[int, int]],
) -> tuple[float, float, float]:
    pos_sq = []
    axis_sq = []
    mirror = torch.tensor((1.0, -1.0, 1.0), dtype=base_pos.dtype, device=base_pos.device)
    for left_i, right_i in pair_indices:
        left_delta = cand_pos[left_i] - base_pos[left_i]
        right_delta = cand_pos[right_i] - base_pos[right_i]
        pos_sq.append(torch.sum(torch.square(left_delta - mirror * right_delta)))

        left_axis_delta = cand_axes[left_i] - base_axes[left_i]
        right_axis_delta = cand_axes[right_i] - base_axes[right_i]
        axis_sq.append(torch.sum(torch.square(left_axis_delta - mirror[None, :] * right_axis_delta)))

    pos_rms = torch.sqrt(torch.mean(torch.stack(pos_sq))).item()
    axis_rms = torch.sqrt(torch.mean(torch.stack(axis_sq))).item()
    combined = pos_rms + args.orientation_weight_m * axis_rms
    return float(pos_rms), float(axis_rms), float(combined)


def _plot_end_poses(candidate_poses: dict[tuple[str, float], dict[str, list[float]]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    all_points = [point for poses in candidate_poses.values() for point in poses.values()]
    if not all_points:
        return

    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    zs = [point[2] for point in all_points]
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0e-3) * 0.55
    center = ((max(xs) + min(xs)) * 0.5, (max(ys) + min(ys)) * 0.5, (max(zs) + min(zs)) * 0.5)
    limits = (
        (center[0] - span, center[0] + span),
        (center[1] - span, center[1] + span),
        (center[2] - span, center[2] + span),
    )

    def draw_pose(ax, poses: dict[str, list[float]], title: str) -> None:
        for chain, color in (
            (("leg0_shell", "leg1_shell", "leg2_shell", "leg3_shell1", "foot1"), "tab:red"),
            (("leg0_shell_2", "leg1_shell3", "leg2_shell_2", "leg3_shell11", "foot3"), "tab:blue"),
        ):
            points = [poses[name] for name in chain if name in poses]
            if points:
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    [point[2] for point in points],
                    marker="o",
                    color=color,
                    linewidth=1.5,
                    markersize=3.0,
                )
        for left_name, right_name in BODY_PAIRS:
            if left_name in poses and right_name in poses:
                left = poses[left_name]
                right = poses[right_name]
                ax.plot([left[0], right[0]], [left[1], right[1]], [left[2], right[2]], color="0.7", linewidth=0.7)
        ax.set_title(title, fontsize=8)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
        ax.set_xlabel("body x", fontsize=7)
        ax.set_ylabel("body y", fontsize=7)
        ax.set_zlabel("body z", fontsize=7)
        ax.view_init(elev=18, azim=-60)
        ax.tick_params(labelsize=6)

    fig = plt.figure(figsize=(9, 15), constrained_layout=True)
    for row_i, (left_joint, _right_joint) in enumerate(JOINT_PAIRS):
        for col_i, sign in enumerate((1.0, -1.0)):
            ax = fig.add_subplot(len(JOINT_PAIRS), 2, row_i * 2 + col_i + 1, projection="3d")
            draw_pose(ax, candidate_poses[(left_joint, sign)], f"{left_joint} mirror {sign:+.0f}")
    fig.savefig(output_dir / "mirror_sign_end_poses_summary.png", dpi=180)
    plt.close(fig)

    for left_joint, _right_joint in JOINT_PAIRS:
        fig = plt.figure(figsize=(9, 4.5), constrained_layout=True)
        for col_i, sign in enumerate((1.0, -1.0)):
            ax = fig.add_subplot(1, 2, col_i + 1, projection="3d")
            draw_pose(ax, candidate_poses[(left_joint, sign)], f"{left_joint} mirror {sign:+.0f}")
        fig.savefig(output_dir / f"mirror_sign_end_pose_{left_joint}.png", dpi=180)
        plt.close(fig)
    print(f"wrote_plots={output_dir}", flush=True)


def main() -> None:
    usd_path = args.usd_path.resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"USD path does not exist: {usd_path}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    robot = Articulation(cfg=_make_robot_cfg(usd_path))
    sim.reset()
    robot.update(sim.get_physics_dt())

    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)
    missing_joints = [name for pair in JOINT_PAIRS for name in pair if name not in joint_names]
    if missing_joints:
        raise RuntimeError(f"Missing joints {missing_joints}. Available joints: {joint_names}")
    available_body_pairs = [(left, right) for left, right in BODY_PAIRS if left in body_names and right in body_names]
    if not available_body_pairs:
        raise RuntimeError(f"No configured mirrored body pairs exist. Available bodies: {body_names}")

    body_ids = sorted({body_names.index(name) for pair in available_body_pairs for name in pair})
    body_id_to_local = {body_id: local_i for local_i, body_id in enumerate(body_ids)}
    pair_indices = [
        (body_id_to_local[body_names.index(left)], body_id_to_local[body_names.index(right)])
        for left, right in available_body_pairs
    ]

    pose = SETTLED_POSE_RAD if args.pose == "settled" else DEFAULT_POSE_RAD
    base_joint_pos = torch.zeros((1, robot.num_joints), dtype=torch.float32, device=sim.device)
    for joint_name, value in pose.items():
        base_joint_pos[0, joint_names.index(joint_name)] = value

    _write_state(robot, sim, base_joint_pos)
    base_pos, base_axes = _capture_body_state(robot, body_ids)

    delta = math.radians(args.delta_deg)
    rows = []
    candidate_poses = {}
    for left_joint, right_joint in JOINT_PAIRS:
        left_id = joint_names.index(left_joint)
        right_id = joint_names.index(right_joint)
        candidates = {}
        for mirror_sign in (1.0, -1.0):
            joint_pos = base_joint_pos.clone()
            joint_pos[0, left_id] += delta
            joint_pos[0, right_id] += mirror_sign * delta
            _write_state(robot, sim, joint_pos)
            cand_pos, cand_axes = _capture_body_state(robot, body_ids)
            pos_rms, axis_rms, combined = _mirrored_delta_errors(base_pos, base_axes, cand_pos, cand_axes, pair_indices)
            candidate_poses[(left_joint, mirror_sign)] = {
                body_names[body_id]: [float(value) for value in cand_pos[local_i].tolist()]
                for local_i, body_id in enumerate(body_ids)
            }
            candidates[str(int(mirror_sign))] = {
                "position_rms_m": pos_rms,
                "axis_rms": axis_rms,
                "combined_error_m": combined,
            }
        best = min(candidates, key=lambda sign: candidates[sign]["combined_error_m"])
        rows.append(
            {
                "left_joint": left_joint,
                "right_joint": right_joint,
                "delta_deg": args.delta_deg,
                "recommended_mirror_sign": float(best),
                "plus": candidates["1"],
                "minus": candidates["-1"],
            }
        )

    result = {
        "usd_path": str(usd_path),
        "pose": args.pose,
        "delta_deg": args.delta_deg,
        "orientation_weight_m": args.orientation_weight_m,
        "body_pairs": available_body_pairs,
        "rows": rows,
    }
    print("MIRROR_SIGN_TEST", json.dumps(result, indent=2), flush=True)
    print("\nsummary:", flush=True)
    print("joint_pair recommended pos_rms(+1/-1) axis_rms(+1/-1) combined(+1/-1)", flush=True)
    for row in rows:
        plus = row["plus"]
        minus = row["minus"]
        print(
            f"{row['left_joint']} / {row['right_joint']}: "
            f"{row['recommended_mirror_sign']:+.0f} "
            f"pos={plus['position_rms_m']:.6f}/{minus['position_rms_m']:.6f} "
            f"axis={plus['axis_rms']:.6f}/{minus['axis_rms']:.6f} "
            f"combined={plus['combined_error_m']:.6f}/{minus['combined_error_m']:.6f}",
            flush=True,
        )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote_json={args.json_output}", flush=True)
    if args.plot_dir is not None:
        _plot_end_poses(candidate_poses, args.plot_dir)


try:
    main()
finally:
    simulation_app.close()
