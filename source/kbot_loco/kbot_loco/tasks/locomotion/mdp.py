from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period_s: float) -> torch.Tensor:
    """Return sin/cos phase features for a nominal walking cycle."""
    phase = torch.remainder(env.episode_length_buf.float() * env.step_dt / period_s, 1.0)
    phase_angle = 2.0 * torch.pi * phase
    return torch.stack((torch.sin(phase_angle), torch.cos(phase_angle)), dim=1)


def alternating_foot_phase_reward(
    env: ManagerBasedRLEnv,
    period_s: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward alternating single-foot contacts against a light nominal gait phase."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    left_contact = in_contact[:, 0]
    right_contact = in_contact[:, 1]
    phase = torch.remainder(env.episode_length_buf.float() * env.step_dt / period_s, 1.0)
    left_stance_phase = phase < 0.5
    right_stance_phase = ~left_stance_phase

    left_single = left_contact & ~right_contact
    right_single = right_contact & ~left_contact
    scheduled_single = (left_stance_phase & left_single) | (right_stance_phase & right_single)
    double_support = left_contact & right_contact
    airborne = ~left_contact & ~right_contact

    return scheduled_single.float() + 0.25 * double_support.float() - airborne.float()


def lateral_velocity_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize body-frame lateral root velocity."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def yaw_rate_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize body-frame yaw rate for straight-line walking."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def root_lateral_tilt_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize side lean while still allowing sagittal pitch for walking."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])


def root_lateral_tilt_ema_l2(
    env: ManagerBasedRLEnv,
    tau_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize persistent side lean with an exponential moving average."""
    asset = env.scene[asset_cfg.name]
    tilt = asset.data.projected_gravity_b[:, 1]
    buffer_name = "_kbot_root_lateral_tilt_ema"
    ema = getattr(env, buffer_name, None)
    if ema is None or ema.shape != tilt.shape or ema.device != tilt.device:
        ema = torch.zeros_like(tilt)

    alpha = min(max(env.step_dt / max(tau_s, 1.0e-6), 0.0), 1.0)
    reset = env.episode_length_buf <= 1
    ema = torch.where(reset, tilt, (1.0 - alpha) * ema + alpha * tilt)
    setattr(env, buffer_name, ema)
    return torch.square(ema)


def world_heading_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize root yaw away from the world +X walking direction."""
    asset = env.scene[asset_cfg.name]
    forward_b = torch.zeros(env.num_envs, 3, device=asset.data.root_pos_w.device)
    forward_b[:, 0] = 1.0
    forward_w = quat_apply(asset.data.root_quat_w, forward_b)
    return torch.square(forward_w[:, 1]) + torch.square(torch.clamp(-forward_w[:, 0], min=0.0))


def backward_velocity_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize moving backward in the body frame."""
    asset = env.scene[asset_cfg.name]
    return torch.square(torch.clamp(asset.data.root_lin_vel_b[:, 0], max=0.0))


def forward_velocity_below_l2(
    env: ManagerBasedRLEnv,
    minimum_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize shuffling or standing when a forward walking command is active."""
    asset = env.scene[asset_cfg.name]
    return torch.square(torch.clamp(minimum_velocity - asset.data.root_lin_vel_b[:, 0], min=0.0))


def root_height_below_l2(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize root height below a desired clearance without ending the episode."""
    asset = env.scene[asset_cfg.name]
    return torch.square(torch.clamp(minimum_height - asset.data.root_pos_w[:, 2], min=0.0))


def root_height_below(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the root drops below a minimum usable standing height."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def upright_alive(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    max_tilt: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward only episodes that are still tall and reasonably upright."""
    asset = env.scene[asset_cfg.name]
    height_ok = asset.data.root_pos_w[:, 2] > minimum_height
    tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    tilt_ok = tilt < max_tilt
    return (height_ok & tilt_ok).float()


def knee_extension_l1(
    env: ManagerBasedRLEnv,
    min_bend: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*knee.*"]),
) -> torch.Tensor:
    """Penalize knees getting close to the straight, mechanically locked pose."""
    asset = env.scene[asset_cfg.name]
    knee_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.clamp(min_bend - torch.abs(knee_pos), min=0.0), dim=1)


def foot_lateral_spacing_l1(
    env: ManagerBasedRLEnv,
    target_width: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Penalize feet crossing or collapsing into a narrow support line."""
    asset = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    root_pos_w = asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    feet_pos_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), (feet_pos_w - root_pos_w).reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )
    width = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    return torch.abs(width - target_width)


def foot_signed_lateral_clearance_l1(
    env: ManagerBasedRLEnv,
    minimum_width: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Penalize crossed feet by preserving left-foot/right-foot lateral ordering."""
    asset = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    root_pos_w = asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    feet_pos_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), (feet_pos_w - root_pos_w).reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )
    signed_width = feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1]
    return torch.clamp(minimum_width - signed_width, min=0.0)


def foot_lateral_lane_l1(
    env: ManagerBasedRLEnv,
    target_left_y: float,
    target_right_y: float,
    tolerance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Penalize feet leaving their neutral body-frame lateral lanes."""
    asset = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    root_pos_w = asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    feet_pos_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), (feet_pos_w - root_pos_w).reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )

    left_error = torch.abs(feet_pos_b[:, 0, 1] - target_left_y)
    right_error = torch.abs(feet_pos_b[:, 1, 1] - target_right_y)
    return torch.clamp(left_error + right_error - tolerance, min=0.0)


def foot_lateral_lane_max_l1(
    env: ManagerBasedRLEnv,
    target_left_y: float,
    target_right_y: float,
    tolerance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Penalize the worst foot's body-frame lateral lane error."""
    asset = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    root_pos_w = asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    feet_pos_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), (feet_pos_w - root_pos_w).reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )

    foot_errors = torch.stack(
        (
            torch.abs(feet_pos_b[:, 0, 1] - target_left_y),
            torch.abs(feet_pos_b[:, 1, 1] - target_right_y),
        ),
        dim=1,
    )
    return torch.clamp(torch.max(foot_errors, dim=1).values - tolerance, min=0.0)


def foot_sole_lateral_lane_max_l1(
    env: ManagerBasedRLEnv,
    target_left_y: float,
    target_right_y: float,
    tolerance: float,
    foot_local_offsets: list[tuple[float, float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Penalize the worst sole-center body-frame lateral lane error."""
    asset = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    local_offsets = torch.tensor(foot_local_offsets, dtype=foot_pos_w.dtype, device=foot_pos_w.device)
    sole_pos_w = foot_pos_w + quat_apply(
        foot_quat_w.reshape(-1, 4), local_offsets[None, :, :].expand(env.num_envs, -1, -1).reshape(-1, 3)
    ).reshape(env.num_envs, len(asset_cfg.body_ids), 3)

    root_pos_w = asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    sole_pos_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), (sole_pos_w - root_pos_w).reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )
    sole_errors = torch.stack(
        (
            torch.abs(sole_pos_b[:, 0, 1] - target_left_y),
            torch.abs(sole_pos_b[:, 1, 1] - target_right_y),
        ),
        dim=1,
    )
    return torch.clamp(torch.max(sole_errors, dim=1).values - tolerance, min=0.0)


def leg_frontal_plane_l1(
    env: ManagerBasedRLEnv,
    tolerance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["leg0_shell", "leg0_shell_2", "leg3_shell1", "leg3_shell11", "foot1", "foot3"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Penalize legs leaning inward/outward instead of moving in sagittal planes."""
    asset = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    root_pos_w = asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    body_pos_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), (body_pos_w - root_pos_w).reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )

    left_hip_y = body_pos_b[:, 0, 1]
    right_hip_y = body_pos_b[:, 1, 1]
    left_shin_y = body_pos_b[:, 2, 1]
    right_shin_y = body_pos_b[:, 3, 1]
    left_foot_y = body_pos_b[:, 4, 1]
    right_foot_y = body_pos_b[:, 5, 1]

    left_error = torch.abs(left_shin_y - left_hip_y) + torch.abs(left_foot_y - left_hip_y)
    right_error = torch.abs(right_shin_y - right_hip_y) + torch.abs(right_foot_y - right_hip_y)
    return torch.clamp(left_error + right_error - tolerance, min=0.0)


def leg_frontal_plane_side_l1(
    env: ManagerBasedRLEnv,
    side: str,
    tolerance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["leg0_shell", "leg0_shell_2", "leg3_shell1", "leg3_shell11", "foot1", "foot3"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Penalize one leg's shin and foot leaving its hip-centered sagittal lane."""
    asset = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    root_pos_w = asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    body_pos_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), (body_pos_w - root_pos_w).reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )

    if side == "left":
        hip_y = body_pos_b[:, 0, 1]
        shin_y = body_pos_b[:, 2, 1]
        foot_y = body_pos_b[:, 4, 1]
    elif side == "right":
        hip_y = body_pos_b[:, 1, 1]
        shin_y = body_pos_b[:, 3, 1]
        foot_y = body_pos_b[:, 5, 1]
    else:
        raise ValueError(f"Unknown leg side: {side}")

    error = torch.abs(shin_y - hip_y) + torch.abs(foot_y - hip_y)
    return torch.clamp(error - tolerance, min=0.0)


def leg_frontal_plane_max_l1(
    env: ManagerBasedRLEnv,
    tolerance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["leg0_shell", "leg0_shell_2", "leg3_shell1", "leg3_shell11", "foot1", "foot3"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Penalize the worst individual shin/foot lateral deviation from its sagittal lane."""
    asset = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    root_pos_w = asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    body_pos_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), (body_pos_w - root_pos_w).reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )

    left_hip_y = body_pos_b[:, 0, 1]
    right_hip_y = body_pos_b[:, 1, 1]
    segment_errors = torch.stack(
        (
            torch.abs(body_pos_b[:, 2, 1] - left_hip_y),
            torch.abs(body_pos_b[:, 4, 1] - left_hip_y),
            torch.abs(body_pos_b[:, 3, 1] - right_hip_y),
            torch.abs(body_pos_b[:, 5, 1] - right_hip_y),
        ),
        dim=1,
    )
    return torch.clamp(torch.max(segment_errors, dim=1).values - tolerance, min=0.0)


def leg_frontal_sole_plane_max_l1(
    env: ManagerBasedRLEnv,
    tolerance: float,
    foot_local_offsets: list[tuple[float, float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["leg0_shell", "leg0_shell_2", "leg3_shell1", "leg3_shell11", "foot1", "foot3"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Penalize the worst hip-to-shin/sole lateral column error."""
    asset = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    local_offsets = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, dtype=body_pos_w.dtype, device=body_pos_w.device)
    local_offsets[:, 4:, :] = torch.tensor(foot_local_offsets, dtype=body_pos_w.dtype, device=body_pos_w.device)[None, :, :]
    body_pos_w = body_pos_w + quat_apply(body_quat_w.reshape(-1, 4), local_offsets.reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )

    root_pos_w = asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    body_pos_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), (body_pos_w - root_pos_w).reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )

    left_hip_y = body_pos_b[:, 0, 1]
    right_hip_y = body_pos_b[:, 1, 1]
    segment_errors = torch.stack(
        (
            torch.abs(body_pos_b[:, 2, 1] - left_hip_y),
            torch.abs(body_pos_b[:, 4, 1] - left_hip_y),
            torch.abs(body_pos_b[:, 3, 1] - right_hip_y),
            torch.abs(body_pos_b[:, 5, 1] - right_hip_y),
        ),
        dim=1,
    )
    return torch.clamp(torch.max(segment_errors, dim=1).values - tolerance, min=0.0)


def foot_sagittal_separation_l1(
    env: ManagerBasedRLEnv,
    target_length: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Penalize short fore-aft foot separation during single-stance walking steps."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    root_pos_w = asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    feet_pos_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), (feet_pos_w - root_pos_w).reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )
    step_length = torch.abs(feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0])
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    single_stance = torch.sum((contact_time > 0.0).int(), dim=1) == 1
    return torch.clamp(target_length - step_length, min=0.0) * single_stance


def swing_foot_overtake_l1(
    env: ManagerBasedRLEnv,
    target_length: float,
    grace_time: float,
    target_air_time: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Penalize half-steps where the swing foot does not pass the stance foot before landing."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    root_pos_w = asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    feet_pos_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), (feet_pos_w - root_pos_w).reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )

    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_air = ~in_contact
    single_stance = torch.sum(in_contact.int(), dim=1) == 1

    swing_progress = torch.clamp((air_time - grace_time) / max(target_air_time - grace_time, 1.0e-6), min=0.0, max=1.0)
    required_lead = target_length * swing_progress

    left_ahead = feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0]
    right_ahead = feet_pos_b[:, 1, 0] - feet_pos_b[:, 0, 0]
    left_swing = in_air[:, 0] & in_contact[:, 1] & single_stance
    right_swing = in_air[:, 1] & in_contact[:, 0] & single_stance

    left_penalty = torch.clamp(required_lead[:, 0] - left_ahead, min=0.0) * left_swing
    right_penalty = torch.clamp(required_lead[:, 1] - right_ahead, min=0.0) * right_swing
    return left_penalty + right_penalty


def foot_parallel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Penalize feet that are not parallel to the base walking direction."""
    asset = env.scene[asset_cfg.name]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    root_quat_w = asset.data.root_quat_w
    forward_b = torch.zeros(env.num_envs, 3, device=asset.data.root_pos_w.device)
    forward_b[:, 0] = 1.0
    root_forward_w = quat_apply(root_quat_w, forward_b)
    root_forward_xy = torch.nn.functional.normalize(root_forward_w[:, :2], dim=1)
    foot_forward_b = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=asset.data.root_pos_w.device)
    foot_forward_b[..., 0] = 1.0
    foot_forward_w = quat_apply(foot_quat_w.reshape(-1, 4), foot_forward_b.reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )
    foot_forward_xy = torch.nn.functional.normalize(foot_forward_w[..., :2], dim=2)
    alignment = torch.abs(torch.sum(foot_forward_xy * root_forward_xy[:, None, :], dim=2))
    return torch.sum(torch.square(1.0 - alignment), dim=1)


def foot_world_parallel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Penalize feet yawing away from the world +X walking direction."""
    asset = env.scene[asset_cfg.name]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    foot_forward_b = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=asset.data.root_pos_w.device)
    foot_forward_b[..., 0] = 1.0
    foot_forward_w = quat_apply(foot_quat_w.reshape(-1, 4), foot_forward_b.reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )
    foot_forward_xy = torch.nn.functional.normalize(foot_forward_w[..., :2], dim=2)
    return torch.sum(torch.square(foot_forward_xy[..., 1]) + torch.square(torch.clamp(-foot_forward_xy[..., 0], min=0.0)), dim=1)


def foot_world_parallel_max_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Penalize the worst foot yawing away from world +X."""
    asset = env.scene[asset_cfg.name]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    foot_forward_b = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=asset.data.root_pos_w.device)
    foot_forward_b[..., 0] = 1.0
    foot_forward_w = quat_apply(foot_quat_w.reshape(-1, 4), foot_forward_b.reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )
    foot_forward_xy = torch.nn.functional.normalize(foot_forward_w[..., :2], dim=2)
    foot_errors = torch.square(foot_forward_xy[..., 1]) + torch.square(torch.clamp(-foot_forward_xy[..., 0], min=0.0))
    return torch.max(foot_errors, dim=1).values


def foot_toe_in_l2(
    env: ManagerBasedRLEnv,
    tolerance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Penalize toes yawing inward toward the robot centerline."""
    asset = env.scene[asset_cfg.name]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    foot_forward_b = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=asset.data.root_pos_w.device)
    foot_forward_b[..., 0] = 1.0
    foot_forward_w = quat_apply(foot_quat_w.reshape(-1, 4), foot_forward_b.reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    foot_forward_root = quat_apply_inverse(root_quat_w.reshape(-1, 4), foot_forward_w.reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )
    foot_forward_xy = torch.nn.functional.normalize(foot_forward_root[..., :2], dim=2)
    left_toe_in = torch.clamp(-foot_forward_xy[:, 0, 1] - tolerance, min=0.0)
    right_toe_in = torch.clamp(foot_forward_xy[:, 1, 1] - tolerance, min=0.0)
    return torch.square(left_toe_in) + torch.square(right_toe_in)


def foot_flat_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Penalize tiptoe-like foot pitch/roll by keeping foot local up close to world up."""
    asset = env.scene[asset_cfg.name]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    up_b = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=asset.data.root_pos_w.device)
    up_b[..., 2] = 1.0
    up_w = quat_apply(foot_quat_w.reshape(-1, 4), up_b.reshape(-1, 3)).reshape(env.num_envs, len(asset_cfg.body_ids), 3)
    return torch.sum(1.0 - torch.square(up_w[..., 2]), dim=1)


def stance_foot_flat_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Penalize tiptoe stance by keeping contacting feet flat on the ground."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    up_b = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=asset.data.root_pos_w.device)
    up_b[..., 2] = 1.0
    up_w = quat_apply(foot_quat_w.reshape(-1, 4), up_b.reshape(-1, 3)).reshape(env.num_envs, len(asset_cfg.body_ids), 3)
    return torch.sum((1.0 - torch.square(up_w[..., 2])) * in_contact, dim=1)


def single_stance_foot_flat_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Penalize foot pitch/roll only for the single support foot."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    single_support = torch.sum(in_contact.int(), dim=1, keepdim=True) == 1
    stance_mask = in_contact & single_support
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    up_b = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=asset.data.root_pos_w.device)
    up_b[..., 2] = 1.0
    up_w = quat_apply(foot_quat_w.reshape(-1, 4), up_b.reshape(-1, 3)).reshape(env.num_envs, len(asset_cfg.body_ids), 3)
    return torch.sum((1.0 - torch.square(up_w[..., 2])) * stance_mask, dim=1)


def locked_knees(
    env: ManagerBasedRLEnv,
    min_bend: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*knee.*"]),
) -> torch.Tensor:
    """Terminate when both knees are almost straight at the same time."""
    asset = env.scene[asset_cfg.name]
    knee_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.all(torch.abs(knee_pos) < min_bend, dim=1)


def joint_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize selected joint velocity for visible wobble without damping every actuator."""
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_vel), dim=1)


def joint_position_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize selected joints moving far from their neutral position."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_pos - default_joint_pos), dim=1)


def mirrored_joint_position_l2(
    env: ManagerBasedRLEnv,
    joint_pairs: list[tuple[str, str, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize persistent left/right joint asymmetry after mirror-sign normalization."""
    asset = env.scene[asset_cfg.name]
    name_to_id = {name: index for index, name in enumerate(asset.data.joint_names)}
    penalty = torch.zeros(env.num_envs, dtype=asset.data.joint_pos.dtype, device=asset.data.joint_pos.device)
    for left_name, right_name, mirror_sign in joint_pairs:
        left_id = name_to_id[left_name]
        right_id = name_to_id[right_name]
        left_error = asset.data.joint_pos[:, left_id] - asset.data.default_joint_pos[:, left_id]
        right_error = asset.data.joint_pos[:, right_id] - asset.data.default_joint_pos[:, right_id]
        penalty += torch.square(left_error - mirror_sign * right_error)
    return penalty


def joint_target_position_l2(
    env: ManagerBasedRLEnv,
    targets: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from an explicit centered posture target."""
    asset = env.scene[asset_cfg.name]
    name_to_id = {name: index for index, name in enumerate(asset.data.joint_names)}
    penalty = torch.zeros(env.num_envs, dtype=asset.data.joint_pos.dtype, device=asset.data.joint_pos.device)
    for joint_name, target in targets.items():
        joint_id = name_to_id[joint_name]
        penalty += torch.square(asset.data.joint_pos[:, joint_id] - target)
    return penalty


def joint_position_ema_l2(
    env: ManagerBasedRLEnv,
    tau_s: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize selected joints holding a persistent offset from neutral."""
    asset = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    buffer_name = f"_kbot_joint_position_ema_{asset_cfg.name}_{len(asset_cfg.joint_ids)}"
    ema = getattr(env, buffer_name, None)
    if ema is None or ema.shape != joint_error.shape or ema.device != joint_error.device:
        ema = torch.zeros_like(joint_error)

    alpha = min(max(env.step_dt / max(tau_s, 1.0e-6), 0.0), 1.0)
    reset = (env.episode_length_buf <= 1).unsqueeze(1)
    ema = torch.where(reset, joint_error, (1.0 - alpha) * ema + alpha * joint_error)
    setattr(env, buffer_name, ema)
    return torch.sum(torch.square(ema), dim=1)


def early_action_sequence_l2(
    env: ManagerBasedRLEnv,
    targets: list[list[float]],
    duration_s: float,
) -> torch.Tensor:
    """Penalize deviation from a short hand-authored bootstrap action sequence."""
    actions = env.action_manager.action
    target = torch.tensor(targets, dtype=actions.dtype, device=actions.device)
    if target.ndim != 2 or target.shape[1] != actions.shape[1]:
        raise ValueError(f"Expected action target shape (*, {actions.shape[1]}), got {tuple(target.shape)}")

    elapsed_s = env.episode_length_buf.float() * env.step_dt
    active = elapsed_s <= duration_s
    phase = torch.clamp(elapsed_s / max(duration_s, 1.0e-6), min=0.0, max=1.0)
    target_index = torch.clamp((phase * (target.shape[0] - 1)).long(), min=0, max=target.shape[0] - 1)
    selected_target = target[target_index]
    return torch.sum(torch.square(actions - selected_target), dim=1) * active
