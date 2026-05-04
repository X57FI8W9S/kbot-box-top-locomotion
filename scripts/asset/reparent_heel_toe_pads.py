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
parser.add_argument("--source-pads-usd", type=Path, default=REPO_ROOT / "kbot_box_top3_pads.usd")
parser.add_argument("--base-usd", type=Path, default=REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top3.usd")
parser.add_argument("--output-usda", type=Path, default=REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top3_pads.usda")
parser.add_argument("--output-usd", type=Path, default=REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top3_pads.usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


PAD_TO_FOOT = {
    "left_heel_pad": "/boxtop_sim/foot1",
    "left_toe_pad": "/boxtop_sim/foot1",
    "right_heel_pad": "/boxtop_sim/foot3",
    "right_toe_pad": "/boxtop_sim/foot3",
}
PAD_PARENT_PATH = "/boxtop_sim"


def _xformable(stage: Usd.Stage, path: str) -> UsdGeom.Xformable:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Missing prim: {path}")
    return UsdGeom.Xformable(prim)


def _find_pad(stage: Usd.Stage, name: str) -> Usd.Prim:
    matches = [prim for prim in stage.Traverse() if prim.GetName() == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name}, found {[str(prim.GetPath()) for prim in matches]}")
    return matches[0]


def _copy_attr(src: Usd.Prim, dst: Usd.Prim, name: str) -> None:
    attr = src.GetAttribute(name)
    if attr:
        dst.CreateAttribute(name, attr.GetTypeName()).Set(attr.Get())


def _add_pad(stage: Usd.Stage, source_pad: Usd.Prim, parent_path: str, xform_matrix: Gf.Matrix4d) -> Sdf.Path:
    pad_path = Sdf.Path(f"{parent_path}/{source_pad.GetName()}")
    cube = UsdGeom.Cube.Define(stage, pad_path)
    cube.CreateSizeAttr(source_pad.GetAttribute("size").Get() if source_pad.GetAttribute("size") else 1.0)

    dst = cube.GetPrim()
    _copy_attr(source_pad, dst, "primvars:displayColor")
    _copy_attr(source_pad, dst, "physics:mass")
    _copy_attr(source_pad, dst, "physics:diagonalInertia")

    xform = UsdGeom.Xformable(dst)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(xform_matrix)

    UsdPhysics.RigidBodyAPI.Apply(dst)
    UsdPhysics.CollisionAPI.Apply(dst)
    UsdPhysics.MassAPI.Apply(dst)
    return pad_path


def _add_fixed_joint(stage: Usd.Stage, source_stage: Usd.Stage, name: str, foot_path: str, pad_path: Sdf.Path) -> None:
    old_joint = source_stage.GetPrimAtPath(f"/boxtop_sim/joints/{name}_fixedJoint")
    joint = UsdPhysics.FixedJoint.Define(stage, f"/boxtop_sim/joints/{name}_fixedJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(foot_path)])
    joint.CreateBody1Rel().SetTargets([pad_path])
    if old_joint:
        for attr_name in ("physics:localPos0", "physics:localPos1", "physics:localRot0", "physics:localRot1"):
            old_attr = old_joint.GetAttribute(attr_name)
            if old_attr:
                joint.GetPrim().CreateAttribute(attr_name, old_attr.GetTypeName()).Set(old_attr.Get())


def _write_layer(source_stage: Usd.Stage, source_world_by_pad: dict[str, Gf.Matrix4d]) -> None:
    args.output_usda.parent.mkdir(parents=True, exist_ok=True)
    for path in (args.output_usda, args.output_usd):
        if path.exists():
            path.unlink()

    stage = Usd.Stage.CreateNew(str(args.output_usda))
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().subLayerPaths.append(args.base_usd.name)
    default_prim = stage.GetPrimAtPath("/boxtop_sim")
    if default_prim:
        stage.SetDefaultPrim(default_prim)

    for name, foot_path in PAD_TO_FOOT.items():
        source_pad = _find_pad(source_stage, name)
        pad_path = _add_pad(stage, source_pad, PAD_PARENT_PATH, source_world_by_pad[name])
        _add_fixed_joint(stage, source_stage, name, foot_path, pad_path)

    stage.GetRootLayer().Save()

    usd_stage = Usd.Stage.CreateNew(str(args.output_usd))
    usd_stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(usd_stage, UsdGeom.Tokens.z)
    usd_stage.GetRootLayer().subLayerPaths.append(args.base_usd.name)
    default_prim = usd_stage.GetPrimAtPath("/boxtop_sim")
    if default_prim:
        usd_stage.SetDefaultPrim(default_prim)
    for name, foot_path in PAD_TO_FOOT.items():
        source_pad = _find_pad(source_stage, name)
        pad_path = _add_pad(usd_stage, source_pad, PAD_PARENT_PATH, source_world_by_pad[name])
        _add_fixed_joint(usd_stage, source_stage, name, foot_path, pad_path)
    usd_stage.GetRootLayer().Save()


def main() -> None:
    source_stage = Usd.Stage.Open(str(args.source_pads_usd))
    if source_stage is None:
        raise RuntimeError(f"Could not open {args.source_pads_usd}")

    source_world_by_pad: dict[str, Gf.Matrix4d] = {}
    for name, foot_path in PAD_TO_FOOT.items():
        source_pad = _find_pad(source_stage, name)
        pad_world = UsdGeom.Xformable(source_pad).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        source_world_by_pad[name] = pad_world

    _write_layer(source_stage, source_world_by_pad)

    output_stage = Usd.Stage.Open(str(args.output_usd))
    if output_stage is None:
        raise RuntimeError(f"Could not open written output {args.output_usd}")

    print(f"source kept unchanged: {args.source_pads_usd}", flush=True)
    print(f"wrote: {args.output_usda}", flush=True)
    print(f"wrote: {args.output_usd}", flush=True)
    print("reference-pose world translation check:", flush=True)
    for name, foot_path in PAD_TO_FOOT.items():
        new_path = f"{PAD_PARENT_PATH}/{name}"
        new_world = _xformable(output_stage, new_path).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        old_t = source_world_by_pad[name].ExtractTranslation()
        new_t = new_world.ExtractTranslation()
        delta = new_t - old_t
        print(
            f"  {name}: old=({old_t[0]:.6f}, {old_t[1]:.6f}, {old_t[2]:.6f}) "
            f"new=({new_t[0]:.6f}, {new_t[1]:.6f}, {new_t[2]:.6f}) "
            f"delta=({delta[0]:.2e}, {delta[1]:.2e}, {delta[2]:.2e})",
            flush=True,
        )


try:
    main()
finally:
    simulation_app.close()
