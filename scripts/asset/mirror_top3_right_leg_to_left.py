#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


parser = argparse.ArgumentParser(
    description=(
        "Create a box-top USD from kbot_box_top3, replacing the left lower leg and foot with a mirrored "
        "copy of the right lower leg and foot while preserving the existing top3 articulation structure."
    )
)
parser.add_argument(
    "--source-usd",
    type=Path,
    default=REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top3.usd",
)
parser.add_argument(
    "--base-layer",
    type=Path,
    default=REPO_ROOT / "assets" / "robot" / "usd" / "configuration" / "kbot_box_top3_base.usd",
)
parser.add_argument(
    "--output-usd",
    type=Path,
    default=REPO_ROOT / "assets" / "robot" / "usd" / "kbot_box_top4.usd",
)
args = parser.parse_args()

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


TIME = Usd.TimeCode.Default()
MIRROR_Y = Gf.Matrix4d(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
BODY_MIRRORS = {
    "leg3_shell1": "leg3_shell11",
    "foot1": "foot3",
}
DIRECT_MESH_MIRRORS = {
    "/boxtop_sim/leg3_shell1/visuals/leg3_shell11": "/meshes/leg3_shell11",
    "/boxtop_sim/foot1/visuals/foot3": "/meshes/foot3",
    "/boxtop_sim/foot1/collisions/foot3_collision": "/meshes/foot3_collision",
}
DISABLED_LEFT_CHILDREN = (
    "/boxtop_sim/leg3_shell1/visuals/leg3_shell1",
    "/boxtop_sim/foot1/visuals/foot1",
    "/boxtop_sim/foot1/collisions/foot1_collision",
)
LEFT_FOOT_COLLIDER = "/boxtop_sim/foot1/collisions/foot3_collision"
LEFT_FOOT_COLLIDER_NODE = f"{LEFT_FOOT_COLLIDER}/node_STL_BINARY_"


def _stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.Open(str(path.resolve()))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {path}")
    return stage


def _world_matrix(stage: Usd.Stage, path: str) -> Gf.Matrix4d:
    return UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(TIME)


def _set_matrix_xform(prim: Usd.Prim, matrix: Gf.Matrix4d) -> None:
    for prop_name in prim.GetPropertyNames():
        if prop_name.startswith("xformOp:") and prop_name != "xformOp:transform":
            prim.GetAttribute(prop_name).Block()
    xform_attr = prim.GetAttribute("xformOp:transform")
    if not xform_attr:
        xform_attr = prim.CreateAttribute("xformOp:transform", Sdf.ValueTypeNames.Matrix4d)
    xform_attr.Set(matrix)
    order_attr = prim.GetAttribute("xformOpOrder")
    if not order_attr:
        order_attr = prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray)
    order_attr.Set(["xformOp:transform"])


def _ensure_parent_specs(layer: Sdf.Layer, path: Sdf.Path) -> None:
    prefixes = []
    parent = path.GetParentPath()
    while parent and parent != Sdf.Path.absoluteRootPath:
        prefixes.append(parent)
        parent = parent.GetParentPath()
    for prefix in reversed(prefixes):
        Sdf.CreatePrimInLayer(layer, prefix)


def _copy_spec(source_layer: Sdf.Layer, source_path: str, target_layer: Sdf.Layer, target_path: str) -> None:
    source = Sdf.Path(source_path)
    target = Sdf.Path(target_path)
    if source_layer.GetPrimAtPath(source) is None:
        raise RuntimeError(f"Source prim spec is missing: {source_layer.identifier}:{source}")
    _ensure_parent_specs(target_layer, target)
    if not Sdf.CopySpec(source_layer, source, target_layer, target):
        raise RuntimeError(f"Could not copy {source} to {target}")


def _mirror_vec3_y(value) -> Gf.Vec3f:
    return Gf.Vec3f(float(value[0]), -float(value[1]), float(value[2]))


def _mirror_points_y(values) -> list[Gf.Vec3f]:
    return [_mirror_vec3_y(value) for value in values]


def _reverse_face_varying(values, counts) -> list:
    reversed_values = []
    index = 0
    for count in counts:
        face_values = values[index : index + count]
        reversed_values.extend(reversed(face_values))
        index += count
    return reversed_values


def _mirror_mesh(mesh_prim: Usd.Prim) -> None:
    mesh = UsdGeom.Mesh(mesh_prim)
    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get()
    if points:
        mirrored_points = _mirror_points_y(points)
        points_attr.Set(mirrored_points)
        xs = [float(point[0]) for point in mirrored_points]
        ys = [float(point[1]) for point in mirrored_points]
        zs = [float(point[2]) for point in mirrored_points]
        mesh.GetExtentAttr().Set(
            [
                Gf.Vec3f(min(xs), min(ys), min(zs)),
                Gf.Vec3f(max(xs), max(ys), max(zs)),
            ]
        )

    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices_attr = mesh.GetFaceVertexIndicesAttr()
    indices = indices_attr.Get()
    if counts and indices:
        indices_attr.Set(_reverse_face_varying(list(indices), counts))

    normals_attr = mesh.GetNormalsAttr()
    normals = normals_attr.Get()
    if normals:
        mirrored_normals = _mirror_points_y(normals)
        if counts and len(mirrored_normals) == sum(counts):
            mirrored_normals = _reverse_face_varying(mirrored_normals, counts)
        normals_attr.Set(mirrored_normals)

    mesh.GetOrientationAttr().Set(UsdGeom.Tokens.rightHanded)


def _mirror_geometry(stage: Usd.Stage, base_layer: Sdf.Layer) -> None:
    root_layer = stage.GetRootLayer()
    for wrapper_path in sorted({Sdf.Path(path).GetParentPath() for path in DIRECT_MESH_MIRRORS}):
        wrapper_spec = Sdf.CreatePrimInLayer(root_layer, wrapper_path)
        wrapper_spec.referenceList.ClearEditsAndMakeExplicit()
        wrapper_spec.SetInfo("instanceable", False)
    for target_path, source_path in DIRECT_MESH_MIRRORS.items():
        _copy_spec(base_layer, source_path, root_layer, target_path)
    for prim_path in DISABLED_LEFT_CHILDREN:
        Sdf.CreatePrimInLayer(root_layer, Sdf.Path(prim_path)).active = False
    root_layer.Save()
    stage.Reload()
    for target_path in DIRECT_MESH_MIRRORS:
        root_prim = stage.GetPrimAtPath(target_path)
        if not root_prim:
            raise RuntimeError(f"Mirrored direct mesh target is missing after copy: {target_path}")
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Mesh):
                _mirror_mesh(prim)
    _restore_left_foot_collision_metadata(stage)


def _restore_left_foot_collision_metadata(stage: Usd.Stage) -> None:
    collider = stage.GetPrimAtPath(LEFT_FOOT_COLLIDER)
    collider_node = stage.GetPrimAtPath(LEFT_FOOT_COLLIDER_NODE)
    if not collider:
        raise RuntimeError(f"Left foot collider wrapper is missing: {LEFT_FOOT_COLLIDER}")
    if not collider_node:
        raise RuntimeError(f"Left foot collider node is missing: {LEFT_FOOT_COLLIDER_NODE}")

    UsdGeom.Imageable(collider).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)

    collision_api = UsdPhysics.CollisionAPI.Apply(collider_node)
    collision_api.CreateCollisionEnabledAttr(True)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(collider_node)
    mesh_collision_api.CreateApproximationAttr().Set("convexHull")


def _copy_mirrored_body_frames(stage: Usd.Stage) -> None:
    for left_name, right_name in BODY_MIRRORS.items():
        left_prim = stage.GetPrimAtPath(f"/boxtop_sim/{left_name}")
        right_matrix = _world_matrix(stage, f"/boxtop_sim/{right_name}")
        _set_matrix_xform(left_prim, MIRROR_Y * right_matrix * MIRROR_Y)

        right_prim = stage.GetPrimAtPath(f"/boxtop_sim/{right_name}")
        for attr_name in ("physics:mass", "physics:diagonalInertia"):
            right_attr = right_prim.GetAttribute(attr_name)
            left_attr = left_prim.GetAttribute(attr_name)
            if right_attr and left_attr:
                left_attr.Set(right_attr.Get())
        right_com = right_prim.GetAttribute("physics:centerOfMass")
        left_com = left_prim.GetAttribute("physics:centerOfMass")
        if right_com and left_com and right_com.Get() is not None:
            left_com.Set(_mirror_vec3_y(right_com.Get()))


def _joint_local_matrix(prim: Usd.Prim, suffix: str) -> Gf.Matrix4d:
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


def _rel_target(prim: Usd.Prim, rel_name: str) -> Sdf.Path:
    targets = prim.GetRelationship(rel_name).GetTargets()
    if not targets:
        raise RuntimeError(f"{prim.GetPath()} has no {rel_name} target")
    return targets[0]


def _reclose_left_joint(stage: Usd.Stage, joint_name: str) -> float:
    joint = stage.GetPrimAtPath(f"/boxtop_sim/joints/{joint_name}")
    body0_path = _rel_target(joint, "physics:body0")
    body1_path = _rel_target(joint, "physics:body1")
    body0_world = _world_matrix(stage, str(body0_path))
    body1_world = _world_matrix(stage, str(body1_path))
    pivot_world = body1_world.ExtractTranslation()

    local_pos0 = body0_world.GetInverse().Transform(pivot_world)
    joint.GetAttribute("physics:localPos0").Set(Gf.Vec3f(*[float(v) for v in local_pos0]))
    joint.GetAttribute("physics:localPos1").Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Keep the parent-side joint frame from top3 so left/right actuator sign conventions stay unchanged.
    local0 = _joint_local_matrix(joint, "0")
    joint0_world = local0 * body0_world
    local1 = joint0_world * body1_world.GetInverse()
    _, local_rot1 = _decompose_joint_matrix(local1)
    joint.GetAttribute("physics:localRot1").Set(local_rot1)

    return (body0_world.Transform(local_pos0) - body1_world.ExtractTranslation()).GetLength()


def main() -> None:
    source_usd = args.source_usd.resolve()
    base_layer_path = args.base_layer.resolve()
    output_usd = args.output_usd.resolve()
    if not source_usd.exists():
        raise FileNotFoundError(source_usd)
    if not base_layer_path.exists():
        raise FileNotFoundError(base_layer_path)
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_usd, output_usd)

    base_layer = Sdf.Layer.FindOrOpen(str(base_layer_path))
    if base_layer is None:
        raise RuntimeError(f"Could not open base layer: {base_layer_path}")
    stage = _stage(output_usd)
    _mirror_geometry(stage, base_layer)
    _copy_mirrored_body_frames(stage)
    left_knee_error = _reclose_left_joint(stage, "left_knee_04")
    left_ankle_error = _reclose_left_joint(stage, "left_ankle_02")
    stage.GetRootLayer().Save()

    print(f"source={source_usd}")
    print(f"base_layer={base_layer_path}")
    print(f"output={output_usd}")
    print(f"left_knee_closure_error_m={left_knee_error:.9f}")
    print(f"left_ankle_closure_error_m={left_ankle_error:.9f}")


if __name__ == "__main__":
    main()
