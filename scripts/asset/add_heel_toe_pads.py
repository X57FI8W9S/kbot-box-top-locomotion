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
parser.add_argument("--source-usd", type=Path, default=REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top3.usd")
parser.add_argument("--output-usd", type=Path, default=REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top3_pads.usda")
parser.add_argument("--pad-length", type=float, default=0.01, help="Pad size along foot/root X in m.")
parser.add_argument("--pad-width", type=float, default=0.01, help="Pad size along foot/root Y in m.")
parser.add_argument("--pad-thickness", type=float, default=0.001, help="Pad size along foot/root Z in m.")
parser.add_argument("--pad-inset", type=float, default=0.004, help="Inset from the front/back foot edge in m.")
parser.add_argument("--pad-lift", type=float, default=0.0, help="Lift above the sole bottom in m.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


FOOT_SIDES = {
    "left": "/boxtop_sim/foot1",
    "right": "/boxtop_sim/foot3",
}


def _local_bbox(cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if box.IsEmpty():
        raise RuntimeError(f"Empty bounds for {prim.GetPath()}")
    return box.GetMin(), box.GetMax()


def _define_pad(
    stage: Usd.Stage,
    source_stage: Usd.Stage,
    *,
    name: str,
    foot_path: str,
    center_root: Gf.Vec3d,
    size: Gf.Vec3d,
    color: tuple[float, float, float],
) -> None:
    foot_prim = source_stage.GetPrimAtPath(foot_path)
    foot_to_world = UsdGeom.Xformable(foot_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    center_local = foot_to_world.GetInverse().Transform(center_root)

    pad_path = Sdf.Path(f"{foot_path}/{name}")
    cube = UsdGeom.Cube.Define(stage, pad_path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(center_local)
    xform.AddScaleOp().Set(Gf.Vec3f(float(size[0]), float(size[1]), float(size[2])))

    prim = cube.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(0.005)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0e-6, 1.0e-6, 1.0e-6))

    joint = UsdPhysics.FixedJoint.Define(stage, f"/boxtop_sim/joints/{name}_fixedJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(foot_path)])
    joint.CreateBody1Rel().SetTargets([pad_path])
    joint.CreateLocalPos0Attr(Gf.Vec3f(float(center_local[0]), float(center_local[1]), float(center_local[2])))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def main() -> None:
    source_stage = Usd.Stage.Open(str(args.source_usd))
    if source_stage is None:
        raise RuntimeError(f"Could not open {args.source_usd}")

    args.output_usd.parent.mkdir(parents=True, exist_ok=True)
    if args.output_usd.exists():
        args.output_usd.unlink()
    stage = Usd.Stage.CreateNew(str(args.output_usd))
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().subLayerPaths.append(args.source_usd.name)

    default_prim = stage.GetPrimAtPath("/boxtop_sim")
    if default_prim:
        stage.SetDefaultPrim(default_prim)

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    pads: list[tuple[str, str, Gf.Vec3d, Gf.Vec3d, tuple[float, float, float]]] = []
    for side, foot_path in FOOT_SIDES.items():
        foot = source_stage.GetPrimAtPath(foot_path)
        mn, mx = _local_bbox(cache, foot)
        # Target centers are computed in root axes: X heel-to-toe, Y lateral, Z vertical.
        # Pad scale is authored in the foot child frame, where local Y maps to vertical.
        pad_size = Gf.Vec3d(args.pad_length, args.pad_thickness, args.pad_width)
        y = (mn[1] + mx[1]) * 0.5
        z = mn[2] + args.pad_lift + args.pad_thickness * 0.5
        heel_x = mn[0] + args.pad_inset + args.pad_length * 0.5
        toe_x = mx[0] - args.pad_inset - args.pad_length * 0.5
        pads.extend(
            [
                (f"{side}_heel_pad", foot_path, Gf.Vec3d(heel_x, y, z), pad_size, (0.1, 0.35, 1.0)),
                (f"{side}_toe_pad", foot_path, Gf.Vec3d(toe_x, y, z), pad_size, (1.0, 0.45, 0.05)),
            ]
        )

    for name, foot_path, center, size, color in pads:
        _define_pad(stage, source_stage, name=name, foot_path=foot_path, center_root=center, size=size, color=color)
        print(
            f"{name}: parent={foot_path} center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}) "
            f"size=({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})",
            flush=True,
        )

    stage.GetRootLayer().Save()
    print(f"wrote {args.output_usd}", flush=True)


try:
    main()
finally:
    simulation_app.close()
