#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Open a raw USD stage, press play, and sample authored prim world poses.")
parser.add_argument("usd_path", type=Path)
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.kit.app  # noqa: E402
import omni.isaac.dynamic_control._dynamic_control as dynamic_control  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402

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


def _world_xyz(stage, prim_path: str) -> tuple[float, float, float] | None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    translation = matrix.ExtractTranslation()
    return (float(translation[0]), float(translation[1]), float(translation[2]))


def _dc_xyz(dc, prim_path: str) -> tuple[float, float, float] | None:
    handle = dc.get_rigid_body(prim_path)
    if handle == dynamic_control.INVALID_HANDLE:
        return None
    pose = dc.get_rigid_body_pose(handle)
    return (float(pose.p.x), float(pose.p.y), float(pose.p.z))


def _dc_articulation(dc, *paths: str):
    for path in paths:
        handle = dc.get_articulation(path)
        if handle != dynamic_control.INVALID_HANDLE:
            return handle, path
    return dynamic_control.INVALID_HANDLE, None


def _dc_joint_pos(dc, articulation) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    if articulation == dynamic_control.INVALID_HANDLE:
        return {name: None for name in JOINT_NAMES}
    for name in JOINT_NAMES:
        dof = dc.find_articulation_dof(articulation, name)
        if dof == dynamic_control.INVALID_HANDLE:
            values[name] = None
            continue
        values[name] = float(dc.get_dof_state(dof, dynamic_control.STATE_POS).pos)
    return values


def main() -> None:
    usd_path = args.usd_path.resolve()
    context = omni.usd.get_context()
    if not context.open_stage(str(usd_path)):
        raise RuntimeError(f"Could not open USD stage: {usd_path}")

    app = omni.kit.app.get_app()
    stage = context.get_stage()
    while stage is None:
        app.update()
        stage = context.get_stage()

    default_prim = stage.GetDefaultPrim()
    root_path = default_prim.GetPath().pathString if default_prim else "/boxtop_sim"
    base_path = f"{root_path}/floating_base_link"
    foot_paths = (f"{root_path}/foot1", f"{root_path}/foot3")

    print(f"RAW_USD usd={usd_path}", flush=True)
    print(f"RAW_USD root={root_path}", flush=True)
    print(f"RAW_USD initial_base_xyz={_world_xyz(stage, base_path)}", flush=True)
    print(f"RAW_USD initial_foot_xyz={[ _world_xyz(stage, path) for path in foot_paths ]}", flush=True)

    timeline = omni.timeline.get_timeline_interface()
    dc = dynamic_control.acquire_dynamic_control_interface()
    timeline.play()
    for _ in range(10):
        app.update()
    articulation, articulation_path = _dc_articulation(dc, root_path, base_path)
    if articulation != dynamic_control.INVALID_HANDLE:
        dc.wake_up_articulation(articulation)
    print(f"RAW_USD articulation_path={articulation_path}", flush=True)
    print(f"RAW_USD initial_dc_base_xyz={_dc_xyz(dc, base_path)}", flush=True)
    print(f"RAW_USD initial_dc_foot_xyz={[ _dc_xyz(dc, path) for path in foot_paths ]}", flush=True)
    print(f"RAW_USD initial_dc_joint_pos={_dc_joint_pos(dc, articulation)}", flush=True)
    for _ in range(args.steps):
        app.update()

    print(f"RAW_USD steps={args.steps}", flush=True)
    print(f"RAW_USD final_base_xyz={_world_xyz(stage, base_path)}", flush=True)
    print(f"RAW_USD final_foot_xyz={[ _world_xyz(stage, path) for path in foot_paths ]}", flush=True)
    print(f"RAW_USD final_dc_base_xyz={_dc_xyz(dc, base_path)}", flush=True)
    print(f"RAW_USD final_dc_foot_xyz={[ _dc_xyz(dc, path) for path in foot_paths ]}", flush=True)
    print(f"RAW_USD final_dc_joint_pos={_dc_joint_pos(dc, articulation)}", flush=True)
    timeline.stop()
    app.update()


try:
    main()
finally:
    simulation_app.close()
