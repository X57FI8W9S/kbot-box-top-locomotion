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
parser.add_argument("--usd", type=Path, default=REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top3.usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402


def _world_bbox(cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> tuple[Gf.Vec3d, Gf.Vec3d] | None:
    bbox = cache.ComputeWorldBound(prim)
    box = bbox.ComputeAlignedBox()
    if box.IsEmpty():
        return None
    return box.GetMin(), box.GetMax()


def main() -> None:
    print("inspect_kbot_usd: started", flush=True)
    stage = Usd.Stage.Open(str(args.usd))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {args.usd}")

    print(f"usd={args.usd}", flush=True)
    print(f"default_prim={stage.GetDefaultPrim().GetPath()}", flush=True)
    print("meters_per_unit=", UsdGeom.GetStageMetersPerUnit(stage), flush=True)
    print("up_axis=", UsdGeom.GetStageUpAxis(stage), flush=True)
    print(flush=True)

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    print("candidate foot prims:", flush=True)
    for prim in stage.Traverse():
        name = prim.GetName()
        if "foot" not in name.lower():
            continue
        apis = [str(api) for api in prim.GetAppliedSchemas()]
        bbox = _world_bbox(cache, prim)
        bbox_text = "empty"
        if bbox is not None:
            mn, mx = bbox
            size = mx - mn
            center = (mn + mx) * 0.5
            bbox_text = (
                f"center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}) "
                f"size=({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})"
            )
        rigid = bool(UsdPhysics.RigidBodyAPI(prim))
        collision = bool(UsdPhysics.CollisionAPI(prim))
        print(
            f"  {prim.GetPath()} type={prim.GetTypeName()} rigid={rigid} collision={collision} apis={apis} {bbox_text}",
            flush=True,
        )

    for foot_path in ("/boxtop_sim/foot1", "/boxtop_sim/foot3"):
        foot = stage.GetPrimAtPath(foot_path)
        print(flush=True)
        print(f"descendants of {foot_path}:", flush=True)
        for prim in Usd.PrimRange(foot):
            apis = [str(api) for api in prim.GetAppliedSchemas()]
            bbox = _world_bbox(cache, prim)
            bbox_text = "empty"
            if bbox is not None:
                mn, mx = bbox
                size = mx - mn
                center = (mn + mx) * 0.5
                bbox_text = (
                    f"center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}) "
                    f"size=({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})"
                )
            print(f"  {prim.GetPath()} type={prim.GetTypeName()} apis={apis} {bbox_text}", flush=True)

    print(flush=True)
    print("rigid bodies:", flush=True)
    for prim in stage.Traverse():
        if UsdPhysics.RigidBodyAPI(prim):
            print(f"  {prim.GetPath()}", flush=True)

    print(flush=True)
    print("collision prims:", flush=True)
    for prim in stage.Traverse():
        if UsdPhysics.CollisionAPI(prim):
            print(f"  {prim.GetPath()} type={prim.GetTypeName()}", flush=True)

    print(flush=True)
    print("joints containing foot:", flush=True)
    for prim in stage.Traverse():
        if "foot" in prim.GetName().lower() and prim.GetTypeName().endswith("Joint"):
            print(f"  {prim.GetPath()} type={prim.GetTypeName()}", flush=True)


try:
    main()
finally:
    simulation_app.close()
