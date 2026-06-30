#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


parser = argparse.ArgumentParser(description="Compare rigid body and joint frames between two KBot USDs.")
parser.add_argument("--base-usd", type=Path, default=REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top3.usd")
parser.add_argument("--test-usd", type=Path, default=REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top4.usd")
parser.add_argument("--only-different", action="store_true")
args = parser.parse_args()

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402


@dataclass(frozen=True)
class JointRecord:
    path: str
    type_name: str
    body0: str
    body1: str
    local_pos0: tuple[float, float, float] | None
    local_pos1: tuple[float, float, float] | None
    local_rot0: tuple[float, float, float, float] | None
    local_rot1: tuple[float, float, float, float] | None
    axis: str | None


def _tuple_attr(prim: Usd.Prim, attr_name: str) -> tuple[float, ...] | None:
    attr = prim.GetAttribute(attr_name)
    if not attr:
        return None
    value = attr.Get()
    if value is None:
        return None
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imaginary = value.GetImaginary()
        return (float(value.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2]))
    return tuple(float(component) for component in value)


def _token_attr(prim: Usd.Prim, attr_name: str) -> str | None:
    attr = prim.GetAttribute(attr_name)
    if not attr:
        return None
    value = attr.Get()
    return None if value is None else str(value)


def _rel_target(prim: Usd.Prim, rel_name: str) -> str:
    rel = prim.GetRelationship(rel_name)
    if not rel:
        return ""
    targets = rel.GetTargets()
    return str(targets[0]) if targets else ""


def _joints(stage: Usd.Stage) -> dict[str, JointRecord]:
    joints: dict[str, JointRecord] = {}
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if not type_name.endswith("Joint"):
            continue
        record = JointRecord(
            path=str(prim.GetPath()),
            type_name=type_name,
            body0=_rel_target(prim, "physics:body0"),
            body1=_rel_target(prim, "physics:body1"),
            local_pos0=_tuple_attr(prim, "physics:localPos0"),
            local_pos1=_tuple_attr(prim, "physics:localPos1"),
            local_rot0=_tuple_attr(prim, "physics:localRot0"),
            local_rot1=_tuple_attr(prim, "physics:localRot1"),
            axis=_token_attr(prim, "physics:axis"),
        )
        joints[prim.GetName()] = record
    return joints


def _rigid_body_centers(stage: Usd.Stage) -> dict[str, tuple[float, float, float]]:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    centers: dict[str, tuple[float, float, float]] = {}
    for prim in stage.Traverse():
        if not UsdPhysics.RigidBodyAPI(prim):
            continue
        bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        if bbox.IsEmpty():
            continue
        center = (bbox.GetMin() + bbox.GetMax()) * 0.5
        centers[prim.GetName()] = (float(center[0]), float(center[1]), float(center[2]))
    return centers


def _rigid_body_origins(stage: Usd.Stage) -> dict[str, tuple[float, float, float]]:
    origins: dict[str, tuple[float, float, float]] = {}
    for prim in stage.Traverse():
        if not UsdPhysics.RigidBodyAPI(prim):
            continue
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = transform.ExtractTranslation()
        origins[prim.GetName()] = (float(translation[0]), float(translation[1]), float(translation[2]))
    return origins


def _rigid_body_orientations(stage: Usd.Stage) -> dict[str, tuple[float, float, float, float]]:
    orientations: dict[str, tuple[float, float, float, float]] = {}
    for prim in stage.Traverse():
        if not UsdPhysics.RigidBodyAPI(prim):
            continue
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        quat = transform.ExtractRotationQuat()
        imaginary = quat.GetImaginary()
        orientations[prim.GetName()] = (
            float(quat.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        )
    return orientations


def _fmt_tuple(value: tuple[float, ...] | None) -> str:
    if value is None:
        return "None"
    return "(" + ", ".join(f"{component:+.6f}" for component in value) + ")"


def _tuple_delta(
    base: tuple[float, ...] | None,
    test: tuple[float, ...] | None,
) -> tuple[float, ...] | None:
    if base is None or test is None or len(base) != len(test):
        return None
    return tuple(test_i - base_i for base_i, test_i in zip(base, test))


def _same_record(base: JointRecord, test: JointRecord) -> bool:
    return (
        base.type_name == test.type_name
        and base.body0 == test.body0
        and base.body1 == test.body1
        and base.local_pos0 == test.local_pos0
        and base.local_pos1 == test.local_pos1
        and base.local_rot0 == test.local_rot0
        and base.local_rot1 == test.local_rot1
        and base.axis == test.axis
    )


def main() -> None:
    base_stage = Usd.Stage.Open(str(args.base_usd.resolve()))
    test_stage = Usd.Stage.Open(str(args.test_usd.resolve()))
    if base_stage is None:
        raise RuntimeError(f"Could not open {args.base_usd}")
    if test_stage is None:
        raise RuntimeError(f"Could not open {args.test_usd}")

    print(f"base={args.base_usd}")
    print(f"test={args.test_usd}")

    base_centers = _rigid_body_centers(base_stage)
    test_centers = _rigid_body_centers(test_stage)
    base_origins = _rigid_body_origins(base_stage)
    test_origins = _rigid_body_origins(test_stage)
    base_orientations = _rigid_body_orientations(base_stage)
    test_orientations = _rigid_body_orientations(test_stage)
    print("\nrigid body frame origin deltas test-base:")
    for name in sorted(set(base_origins) | set(test_origins)):
        base = base_origins.get(name)
        test = test_origins.get(name)
        delta = _tuple_delta(base, test)
        if args.only_different and delta == (0.0, 0.0, 0.0):
            continue
        print(f"  {name:20s} base={_fmt_tuple(base)} test={_fmt_tuple(test)} delta={_fmt_tuple(delta)}")

    print("\nrigid body frame orientation deltas test-base:")
    for name in sorted(set(base_orientations) | set(test_orientations)):
        base = base_orientations.get(name)
        test = test_orientations.get(name)
        delta = _tuple_delta(base, test)
        if args.only_different and delta == (0.0, 0.0, 0.0, 0.0):
            continue
        print(f"  {name:20s} base={_fmt_tuple(base)} test={_fmt_tuple(test)} delta={_fmt_tuple(delta)}")

    print("\nrigid body bbox center deltas test-base:")
    for name in sorted(set(base_centers) | set(test_centers)):
        base = base_centers.get(name)
        test = test_centers.get(name)
        delta = _tuple_delta(base, test)
        if args.only_different and delta == (0.0, 0.0, 0.0):
            continue
        print(f"  {name:20s} base={_fmt_tuple(base)} test={_fmt_tuple(test)} delta={_fmt_tuple(delta)}")

    base_joints = _joints(base_stage)
    test_joints = _joints(test_stage)
    print("\njoint frame differences:")
    for name in sorted(set(base_joints) | set(test_joints)):
        base = base_joints.get(name)
        test = test_joints.get(name)
        if base is None or test is None:
            print(f"  {name}: base={base is not None} test={test is not None}")
            continue
        if args.only_different and _same_record(base, test):
            continue
        print(f"  {name}")
        print(f"    type       base={base.type_name} test={test.type_name}")
        print(f"    body0      base={base.body0} test={test.body0}")
        print(f"    body1      base={base.body1} test={test.body1}")
        print(
            f"    localPos0  base={_fmt_tuple(base.local_pos0)} test={_fmt_tuple(test.local_pos0)} "
            f"delta={_fmt_tuple(_tuple_delta(base.local_pos0, test.local_pos0))}"
        )
        print(
            f"    localPos1  base={_fmt_tuple(base.local_pos1)} test={_fmt_tuple(test.local_pos1)} "
            f"delta={_fmt_tuple(_tuple_delta(base.local_pos1, test.local_pos1))}"
        )
        print(f"    localRot0  base={_fmt_tuple(base.local_rot0)} test={_fmt_tuple(test.local_rot0)}")
        print(f"    localRot1  base={_fmt_tuple(base.local_rot1)} test={_fmt_tuple(test.local_rot1)}")
        print(f"    axis       base={base.axis} test={test.axis}")


if __name__ == "__main__":
    main()
