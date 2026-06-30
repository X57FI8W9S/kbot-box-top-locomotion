#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


parser = argparse.ArgumentParser(
    description=(
        "Copy KBot rigid-body frames into a target USD while preserving target visual/collision child world "
        "transforms. Joint frames can either be copied as local attrs or preserved as world-space source frames."
    )
)
parser.add_argument("--source-usd", type=Path, default=REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top3.usd")
parser.add_argument("--target-usd", type=Path, default=REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top4.usd")
parser.add_argument(
    "--joint-source-usd",
    type=Path,
    default=None,
    help="Optional USD whose joint world frames should be preserved after body-frame transplant.",
)
parser.add_argument(
    "--joint-mode",
    choices=("local", "world"),
    default="local",
    help="Copy joint local attrs directly, or preserve source joint frames in world coordinates.",
)
args = parser.parse_args()

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


TIME = Usd.TimeCode.Default()
FRAME_ATTRS = (
    "physics:localPos0",
    "physics:localPos1",
    "physics:localRot0",
    "physics:localRot1",
    "physics:axis",
)


def _stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.Open(str(path.resolve()))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {path}")
    return stage


def _rigid_bodies(stage: Usd.Stage) -> dict[str, Usd.Prim]:
    return {prim.GetName(): prim for prim in stage.Traverse() if UsdPhysics.RigidBodyAPI(prim)}


def _joints(stage: Usd.Stage) -> dict[str, Usd.Prim]:
    return {prim.GetName(): prim for prim in stage.Traverse() if prim.GetTypeName().endswith("Joint")}


def _world_matrix(prim: Usd.Prim) -> Gf.Matrix4d:
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(TIME)


def _set_matrix_xform(prim: Usd.Prim, matrix: Gf.Matrix4d) -> None:
    xform_attr = prim.GetAttribute("xformOp:transform")
    if not xform_attr:
        xform_attr = prim.CreateAttribute("xformOp:transform", Sdf.ValueTypeNames.Matrix4d)
    xform_attr.Set(matrix)
    order_attr = prim.GetAttribute("xformOpOrder")
    if not order_attr:
        order_attr = prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray)
    order_attr.Set(["xformOp:transform"])


def _translation_delta(a: Gf.Matrix4d, b: Gf.Matrix4d) -> float:
    delta = a.ExtractTranslation() - b.ExtractTranslation()
    return max(abs(float(delta[0])), abs(float(delta[1])), abs(float(delta[2])))


def _copy_joint_frames(source_stage: Usd.Stage, target_stage: Usd.Stage) -> int:
    source_joints = _joints(source_stage)
    target_joints = _joints(target_stage)
    copied = 0
    for name, source_prim in source_joints.items():
        target_prim = target_joints.get(name)
        if target_prim is None:
            continue
        for attr_name in FRAME_ATTRS:
            source_attr = source_prim.GetAttribute(attr_name)
            if not source_attr:
                continue
            target_attr = target_prim.GetAttribute(attr_name)
            if not target_attr:
                target_attr = target_prim.CreateAttribute(attr_name, source_attr.GetTypeName())
            target_attr.Set(source_attr.Get())
        copied += 1
    return copied


def _matrix_from_joint_attrs(prim: Usd.Prim, suffix: str) -> Gf.Matrix4d:
    pos = prim.GetAttribute(f"physics:localPos{suffix}").Get()
    rot = prim.GetAttribute(f"physics:localRot{suffix}").Get()
    imaginary = rot.GetImaginary()
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(
        Gf.Quatd(
            float(rot.GetReal()),
            Gf.Vec3d(float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
        )
    )
    matrix.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    return matrix


def _decompose_joint_matrix(matrix: Gf.Matrix4d) -> tuple[Gf.Vec3f, Gf.Quatf]:
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    return (
        Gf.Vec3f(float(translation[0]), float(translation[1]), float(translation[2])),
        Gf.Quatf(
            float(rotation.GetReal()),
            Gf.Vec3f(float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
        ),
    )


def _copy_joint_world_frames(source_stage: Usd.Stage, target_stage: Usd.Stage) -> int:
    source_joints = _joints(source_stage)
    target_joints = _joints(target_stage)
    copied = 0
    for name, source_prim in source_joints.items():
        target_prim = target_joints.get(name)
        if target_prim is None:
            continue

        source_body0 = source_prim.GetRelationship("physics:body0").GetTargets()[0]
        source_body1 = source_prim.GetRelationship("physics:body1").GetTargets()[0]
        target_body0 = target_prim.GetRelationship("physics:body0").GetTargets()[0]
        target_body1 = target_prim.GetRelationship("physics:body1").GetTargets()[0]

        source_body0_world = _world_matrix(source_stage.GetPrimAtPath(source_body0))
        source_body1_world = _world_matrix(source_stage.GetPrimAtPath(source_body1))
        target_body0_world_inv = _world_matrix(target_stage.GetPrimAtPath(target_body0)).GetInverse()
        target_body1_world_inv = _world_matrix(target_stage.GetPrimAtPath(target_body1)).GetInverse()

        source_joint0_world = _matrix_from_joint_attrs(source_prim, "0") * source_body0_world
        source_joint1_world = _matrix_from_joint_attrs(source_prim, "1") * source_body1_world
        target_joint0_local = source_joint0_world * target_body0_world_inv
        target_joint1_local = source_joint1_world * target_body1_world_inv
        local_pos0, local_rot0 = _decompose_joint_matrix(target_joint0_local)
        local_pos1, local_rot1 = _decompose_joint_matrix(target_joint1_local)

        target_prim.GetAttribute("physics:localPos0").Set(local_pos0)
        target_prim.GetAttribute("physics:localPos1").Set(local_pos1)
        target_prim.GetAttribute("physics:localRot0").Set(local_rot0)
        target_prim.GetAttribute("physics:localRot1").Set(local_rot1)
        for attr_name in ("physics:axis",):
            source_attr = source_prim.GetAttribute(attr_name)
            if source_attr:
                target_prim.GetAttribute(attr_name).Set(source_attr.Get())
        copied += 1
    return copied


def _copy_body_frames(source_stage: Usd.Stage, target_stage: Usd.Stage) -> tuple[int, float]:
    source_bodies = _rigid_bodies(source_stage)
    target_bodies = _rigid_bodies(target_stage)
    child_world_before: dict[tuple[str, str], Gf.Matrix4d] = {}

    for body_name, target_body in target_bodies.items():
        if body_name not in source_bodies:
            continue
        for child in target_body.GetChildren():
            if UsdGeom.Xformable(child):
                child_world_before[(body_name, child.GetName())] = _world_matrix(child)

    copied = 0
    for body_name, source_body in source_bodies.items():
        target_body = target_bodies.get(body_name)
        if target_body is None:
            continue
        _set_matrix_xform(target_body, _world_matrix(source_body))
        copied += 1

        target_body_world_inv = _world_matrix(target_body).GetInverse()
        for child in target_body.GetChildren():
            before = child_world_before.get((body_name, child.GetName()))
            if before is None:
                continue
            _set_matrix_xform(child, before * target_body_world_inv)

    max_child_translation_error = 0.0
    for (body_name, child_name), before in child_world_before.items():
        target_body = target_bodies[body_name]
        child = target_body.GetChild(child_name)
        if not child:
            continue
        max_child_translation_error = max(max_child_translation_error, _translation_delta(before, _world_matrix(child)))

    return copied, max_child_translation_error


def main() -> None:
    source_stage = _stage(args.source_usd)
    target_stage = _stage(args.target_usd)
    joint_source_stage = _stage(args.joint_source_usd) if args.joint_source_usd else source_stage

    body_count, child_error = _copy_body_frames(source_stage, target_stage)
    if args.joint_mode == "world":
        joint_count = _copy_joint_world_frames(joint_source_stage, target_stage)
    else:
        joint_count = _copy_joint_frames(joint_source_stage, target_stage)
    target_stage.GetRootLayer().Save()

    print(f"source={args.source_usd}")
    print(f"target={args.target_usd}")
    print(f"copied_body_frames={body_count}")
    print(f"copied_joint_frames={joint_count}")
    print(f"max_preserved_child_translation_error_m={child_error:.9f}")


if __name__ == "__main__":
    main()
