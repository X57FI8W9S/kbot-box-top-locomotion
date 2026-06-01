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


def root_lateral_position_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize lateral root displacement from the environment origin."""
    asset = env.scene[asset_cfg.name]
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is None:
        lateral_pos = asset.data.root_pos_w[:, 1]
    else:
        lateral_pos = asset.data.root_pos_w[:, 1] - env_origins[:, 1]
    return torch.square(lateral_pos)


def lateral_away_from_center_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize world-frame lateral velocity only when it moves farther from center."""
    asset = env.scene[asset_cfg.name]
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is None:
        lateral_pos = asset.data.root_pos_w[:, 1]
    else:
        lateral_pos = asset.data.root_pos_w[:, 1] - env_origins[:, 1]
    away_speed = torch.clamp(lateral_pos * asset.data.root_lin_vel_w[:, 1], min=0.0)
    return torch.square(away_speed)


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


def signed_joint_pair_ema_symmetry_l2(
    env: ManagerBasedRLEnv,
    joint_pairs: list[tuple[str, str, float]],
    tau_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize persistent left/right joint-pair bias after sign normalization."""
    asset = env.scene[asset_cfg.name]
    name_to_id = {name: index for index, name in enumerate(asset.data.joint_names)}
    pair_errors = []
    for left_name, right_name, mirror_sign in joint_pairs:
        left_id = name_to_id[left_name]
        right_id = name_to_id[right_name]
        pair_errors.append(asset.data.joint_pos[:, left_id] - mirror_sign * asset.data.joint_pos[:, right_id])

    if not pair_errors:
        return torch.zeros(env.num_envs, dtype=asset.data.joint_pos.dtype, device=asset.data.joint_pos.device)

    error = torch.stack(pair_errors, dim=1)
    buffer_name = "_kbot_signed_joint_pair_symmetry_ema"
    ema = getattr(env, buffer_name, None)
    if ema is None or ema.shape != error.shape or ema.device != error.device:
        ema = torch.zeros_like(error)

    alpha = min(max(env.step_dt / max(tau_s, 1.0e-6), 0.0), 1.0)
    reset = (env.episode_length_buf <= 1).unsqueeze(1)
    ema = torch.where(reset, error, (1.0 - alpha) * ema + alpha * error)
    setattr(env, buffer_name, ema)
    return torch.sum(torch.square(ema), dim=1)


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


def world_forward_velocity_below_l2(
    env: ManagerBasedRLEnv,
    minimum_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lack of actual world +X progress, not just body-frame shuffling."""
    asset = env.scene[asset_cfg.name]
    root_lin_vel_w = quat_apply(asset.data.root_quat_w, asset.data.root_lin_vel_b)
    return torch.square(torch.clamp(minimum_velocity - root_lin_vel_w[:, 0], min=0.0))


def world_forward_velocity_clip(
    env: ManagerBasedRLEnv,
    max_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward real world +X progress while clipping incentives above the starter command range."""
    asset = env.scene[asset_cfg.name]
    root_lin_vel_w = quat_apply(asset.data.root_quat_w, asset.data.root_lin_vel_b)
    return torch.clamp(root_lin_vel_w[:, 0] / max(max_velocity, 1.0e-6), min=0.0, max=1.0)


def _upright_health_gate(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    max_tilt: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return one only while the root is tall and reasonably upright."""
    asset = env.scene[asset_cfg.name]
    height_ok = asset.data.root_pos_w[:, 2] > minimum_height
    tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    return (height_ok & (tilt < max_tilt)).float()


def upright_gated_track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    minimum_height: float,
    max_tilt: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward commanded body-frame velocity tracking only while upright."""
    asset = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2) * _upright_health_gate(env, minimum_height, max_tilt, asset_cfg)


def upright_gated_world_forward_velocity_clip(
    env: ManagerBasedRLEnv,
    max_velocity: float,
    minimum_height: float,
    max_tilt: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward clipped world +X progress only while upright."""
    return world_forward_velocity_clip(env, max_velocity, asset_cfg) * _upright_health_gate(
        env, minimum_height, max_tilt, asset_cfg
    )


def swing_sole_clearance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float,
    drag_floor: float,
    drag_weight: float,
    minimum_height: float,
    max_tilt: float,
    foot_local_offsets: list[tuple[float, float, float]],
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward newly reached swing clearance and penalize dragging during single support."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    offsets = torch.tensor(foot_local_offsets, dtype=foot_pos_w.dtype, device=foot_pos_w.device)
    sole_pos_w = foot_pos_w + quat_apply(foot_quat_w, offsets.unsqueeze(0).expand(env.num_envs, -1, -1))
    sole_height = sole_pos_w[:, :, 2]
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        sole_height = sole_height - env_origins[:, None, 2]

    left_swing = ~in_contact[:, 0] & in_contact[:, 1]
    right_swing = ~in_contact[:, 1] & in_contact[:, 0]
    single_support_swing = torch.stack((left_swing, right_swing), dim=1)
    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    upright = _upright_health_gate(env, minimum_height, max_tilt, asset_cfg)

    max_height_name = "_kbot_swing_sole_clearance_max"
    previous_max = getattr(env, max_height_name, None)
    if previous_max is None or previous_max.shape != sole_height.shape or previous_max.device != sole_height.device:
        previous_max = torch.zeros_like(sole_height)
    previous_max = torch.where(in_contact, torch.zeros_like(previous_max), previous_max)

    capped_height = torch.clamp(sole_height, min=0.0, max=target_height)
    clearance_reward = torch.clamp(capped_height - previous_max, min=0.0) / max(target_height, 1.0e-6)
    updated_max = torch.maximum(previous_max, capped_height)
    setattr(env, max_height_name, updated_max)

    drag_penalty = drag_weight * torch.clamp(drag_floor - sole_height, min=0.0) / max(drag_floor, 1.0e-6)

    active = single_support_swing & moving[:, None] & (upright[:, None] > 0.0)
    return torch.sum((clearance_reward - drag_penalty) * active.float(), dim=1)


def walking_cycle_cadence_above_l2(
    env: ManagerBasedRLEnv,
    max_cycle_hz: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize per-foot same-foot cadence above walking range."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    now = env.episode_length_buf.float() * env.step_dt

    last_touchdown_name = "_kbot_last_cycle_touchdown_time"
    cycle_hz_name = "_kbot_cycle_hz"
    last_touchdown = getattr(env, last_touchdown_name, None)
    cycle_hz = getattr(env, cycle_hz_name, None)
    if last_touchdown is None or last_touchdown.shape != contact_time.shape or last_touchdown.device != contact_time.device:
        last_touchdown = torch.full_like(contact_time, -1.0)
    if cycle_hz is None or cycle_hz.shape != contact_time.shape or cycle_hz.device != contact_time.device:
        cycle_hz = torch.zeros_like(contact_time)

    reset = env.episode_length_buf <= 1
    last_touchdown = torch.where(reset[:, None], torch.full_like(last_touchdown, -1.0), last_touchdown)
    cycle_hz = torch.where(reset[:, None], torch.zeros_like(cycle_hz), cycle_hz)

    duration = now[:, None] - last_touchdown
    has_previous = last_touchdown >= 0.0
    valid_cycle = first_contact & has_previous & (duration > env.step_dt)
    measured_hz = torch.where(valid_cycle, 1.0 / torch.clamp(duration, min=env.step_dt), cycle_hz)
    cycle_hz = torch.where(valid_cycle, measured_hz, cycle_hz)
    last_touchdown = torch.where(first_contact, now[:, None].expand_as(last_touchdown), last_touchdown)

    setattr(env, last_touchdown_name, last_touchdown)
    setattr(env, cycle_hz_name, cycle_hz)
    moving = torch.any(in_contact, dim=1)
    return torch.sum(torch.square(torch.clamp(cycle_hz - max_cycle_hz, min=0.0)), dim=1) * moving.float()


def contact_chatter_l1(
    env: ManagerBasedRLEnv,
    min_air_time: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize touchdown events that did not spend enough time in swing."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    penalty = torch.sum(torch.clamp(min_air_time - last_air_time, min=0.0) * first_contact, dim=1)
    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    return penalty * moving.float()


def valid_step_root_advance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_cycle_hz: float,
    min_step_advance: float,
    min_air_time: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward alternating touchdowns only when the root actually advanced."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    command = env.command_manager.get_command(command_name)
    cmd_vx = torch.clamp(command[:, 0], min=0.0)
    target_advance = torch.clamp(cmd_vx / max(2.0 * target_cycle_hz, 1.0e-6), min=min_step_advance, max=0.12)
    root_x = asset.data.root_pos_w[:, 0]

    last_root_name = "_kbot_last_step_touchdown_root_x"
    last_foot_name = "_kbot_last_step_touchdown_foot"
    last_root = getattr(env, last_root_name, None)
    last_foot = getattr(env, last_foot_name, None)
    if last_root is None or last_root.shape != root_x.shape or last_root.device != root_x.device:
        last_root = root_x.clone()
    if last_foot is None or last_foot.shape != root_x.shape or last_foot.device != root_x.device:
        last_foot = torch.full((env.num_envs,), -1, dtype=torch.long, device=root_x.device)

    reset = env.episode_length_buf <= 1
    last_root = torch.where(reset, root_x, last_root)
    last_foot = torch.where(reset, torch.full_like(last_foot, -1), last_foot)

    reward = torch.zeros_like(root_x)
    for foot_i in range(first_contact.shape[1]):
        touchdown = first_contact[:, foot_i]
        alternating = last_foot >= 0
        alternating &= last_foot != foot_i
        enough_air = last_air_time[:, foot_i] >= min_air_time
        advance = root_x - last_root
        normalized = torch.clamp((advance - min_step_advance) / torch.clamp(target_advance - min_step_advance, min=1.0e-4), min=0.0, max=1.0)
        reward = torch.where(touchdown & alternating & enough_air, torch.maximum(reward, normalized), reward)
        last_root = torch.where(touchdown, root_x, last_root)
        last_foot = torch.where(touchdown, torch.full_like(last_foot, foot_i), last_foot)

    moving = torch.norm(command[:, :2], dim=1) > 0.05
    setattr(env, last_root_name, last_root)
    setattr(env, last_foot_name, last_foot)
    return reward * moving.float()


def step_advance_margin_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_cycle_hz: float,
    min_step_advance: float,
    max_step_advance: float,
    short_step_penalty_scale: float,
    min_step_duration: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Signed completed-step reward: negative below the minimum, positive toward the speed target."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    command = env.command_manager.get_command(command_name)
    cmd_vx = torch.clamp(command[:, 0], min=0.0)
    target_advance = torch.clamp(
        cmd_vx / max(2.0 * target_cycle_hz, 1.0e-6),
        min=min_step_advance,
        max=max_step_advance,
    )
    root_x = asset.data.root_pos_w[:, 0]

    last_root_name = "_kbot_step_margin_touchdown_root_x"
    last_foot_name = "_kbot_step_margin_touchdown_foot"
    last_time_name = "_kbot_step_margin_touchdown_time"
    last_root = getattr(env, last_root_name, None)
    last_foot = getattr(env, last_foot_name, None)
    last_time = getattr(env, last_time_name, None)
    if last_root is None or last_root.shape != root_x.shape or last_root.device != root_x.device:
        last_root = root_x.clone()
    if last_foot is None or last_foot.shape != root_x.shape or last_foot.device != root_x.device:
        last_foot = torch.full((env.num_envs,), -1, dtype=torch.long, device=root_x.device)
    current_time = env.episode_length_buf.to(root_x.dtype) * env.step_dt
    if last_time is None or last_time.shape != root_x.shape or last_time.device != root_x.device:
        last_time = current_time.clone()

    reset = env.episode_length_buf <= 1
    last_root = torch.where(reset, root_x, last_root)
    last_foot = torch.where(reset, torch.full_like(last_foot, -1), last_foot)
    last_time = torch.where(reset, current_time, last_time)

    reward = torch.zeros_like(root_x)
    for foot_i in range(first_contact.shape[1]):
        touchdown = first_contact[:, foot_i]
        alternating = last_foot >= 0
        alternating &= last_foot != foot_i
        enough_time = (current_time - last_time) >= min_step_duration
        valid_event = touchdown & alternating

        advance = root_x - last_root
        margin = advance - min_step_advance
        positive = torch.clamp(
            margin / torch.clamp(target_advance - min_step_advance, min=1.0e-4),
            min=0.0,
            max=1.0,
        )
        negative = torch.clamp(margin / max(min_step_advance, 1.0e-6), min=-1.0, max=0.0)
        signed = torch.where(enough_time, positive, torch.zeros_like(positive)) + short_step_penalty_scale * negative
        reward = torch.where(valid_event, signed, reward)

        last_root = torch.where(touchdown, root_x, last_root)
        last_foot = torch.where(touchdown, torch.full_like(last_foot, foot_i), last_foot)
        last_time = torch.where(touchdown, current_time, last_time)

    moving = torch.norm(command[:, :2], dim=1) > 0.05
    setattr(env, last_root_name, last_root)
    setattr(env, last_foot_name, last_foot)
    setattr(env, last_time_name, last_time)
    return reward * moving.float()


def dense_single_support_step_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_cycle_hz: float,
    max_step_advance: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense root-advance progress during single support, reset at each touchdown."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    single_support = torch.sum(in_contact.int(), dim=1) == 1
    command = env.command_manager.get_command(command_name)
    cmd_vx = torch.clamp(command[:, 0], min=0.0)
    target_advance = torch.clamp(cmd_vx / max(2.0 * target_cycle_hz, 1.0e-6), min=1.0e-4, max=max_step_advance)
    root_x = asset.data.root_pos_w[:, 0]

    last_root_name = "_kbot_dense_step_progress_root_x"
    last_root = getattr(env, last_root_name, None)
    if last_root is None or last_root.shape != root_x.shape or last_root.device != root_x.device:
        last_root = root_x.clone()

    reset = env.episode_length_buf <= 1
    last_root = torch.where(reset, root_x, last_root)
    advance = root_x - last_root
    progress = torch.clamp(advance / target_advance, min=0.0, max=1.0)

    any_touchdown = torch.any(first_contact, dim=1)
    last_root = torch.where(any_touchdown, root_x, last_root)

    moving = torch.norm(command[:, :2], dim=1) > 0.05
    setattr(env, last_root_name, last_root)
    return progress * single_support.float() * moving.float()


def contact_duty_symmetry_l2(
    env: ManagerBasedRLEnv,
    tau_s: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize persistent left/right stance-duty imbalance."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    contact = (contact_time > 0.0).float()

    buffer_name = "_kbot_contact_duty_symmetry_ema"
    ema = getattr(env, buffer_name, None)
    if ema is None or ema.shape != contact.shape or ema.device != contact.device:
        ema = torch.zeros_like(contact)

    alpha = min(max(env.step_dt / max(tau_s, 1.0e-6), 0.0), 1.0)
    reset = (env.episode_length_buf <= 1).unsqueeze(1)
    ema = torch.where(reset, contact, (1.0 - alpha) * ema + alpha * contact)
    setattr(env, buffer_name, ema)
    return torch.square(ema[:, 0] - ema[:, 1])


def alternating_step_symmetry_l2(
    env: ManagerBasedRLEnv,
    tau_s: float,
    advance_scale: float,
    duration_scale: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize persistent left/right alternating-step advance and timing imbalance."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    root_x = asset.data.root_pos_w[:, 0]
    current_time = env.episode_length_buf.to(root_x.dtype) * env.step_dt

    last_root_name = "_kbot_step_symmetry_touchdown_root_x"
    last_time_name = "_kbot_step_symmetry_touchdown_time"
    last_foot_name = "_kbot_step_symmetry_touchdown_foot"
    advance_ema_name = "_kbot_step_symmetry_advance_ema"
    duration_ema_name = "_kbot_step_symmetry_duration_ema"
    seen_name = "_kbot_step_symmetry_seen"

    last_root = getattr(env, last_root_name, None)
    last_time = getattr(env, last_time_name, None)
    last_foot = getattr(env, last_foot_name, None)
    advance_ema = getattr(env, advance_ema_name, None)
    duration_ema = getattr(env, duration_ema_name, None)
    seen = getattr(env, seen_name, None)

    if last_root is None or last_root.shape != root_x.shape or last_root.device != root_x.device:
        last_root = root_x.clone()
    if last_time is None or last_time.shape != root_x.shape or last_time.device != root_x.device:
        last_time = current_time.clone()
    if last_foot is None or last_foot.shape != root_x.shape or last_foot.device != root_x.device:
        last_foot = torch.full((env.num_envs,), -1, dtype=torch.long, device=root_x.device)
    if advance_ema is None or advance_ema.shape != first_contact.shape or advance_ema.device != root_x.device:
        advance_ema = torch.zeros(env.num_envs, first_contact.shape[1], dtype=root_x.dtype, device=root_x.device)
    if duration_ema is None or duration_ema.shape != first_contact.shape or duration_ema.device != root_x.device:
        duration_ema = torch.zeros(env.num_envs, first_contact.shape[1], dtype=root_x.dtype, device=root_x.device)
    if seen is None or seen.shape != first_contact.shape or seen.device != root_x.device:
        seen = torch.zeros(env.num_envs, first_contact.shape[1], dtype=torch.bool, device=root_x.device)

    reset = env.episode_length_buf <= 1
    last_root = torch.where(reset, root_x, last_root)
    last_time = torch.where(reset, current_time, last_time)
    last_foot = torch.where(reset, torch.full_like(last_foot, -1), last_foot)
    advance_ema = torch.where(reset[:, None], torch.zeros_like(advance_ema), advance_ema)
    duration_ema = torch.where(reset[:, None], torch.zeros_like(duration_ema), duration_ema)
    seen = torch.where(reset[:, None], torch.zeros_like(seen), seen)

    alpha = min(max(env.step_dt / max(tau_s, 1.0e-6), 0.0), 1.0)
    for foot_i in range(first_contact.shape[1]):
        touchdown = first_contact[:, foot_i]
        alternating = (last_foot >= 0) & (last_foot != foot_i)
        valid_event = touchdown & alternating

        advance = torch.clamp(root_x - last_root, min=0.0)
        duration = torch.clamp(current_time - last_time, min=0.0)
        prior_seen = seen[:, foot_i]
        new_advance = torch.where(prior_seen, (1.0 - alpha) * advance_ema[:, foot_i] + alpha * advance, advance)
        new_duration = torch.where(prior_seen, (1.0 - alpha) * duration_ema[:, foot_i] + alpha * duration, duration)
        advance_ema[:, foot_i] = torch.where(valid_event, new_advance, advance_ema[:, foot_i])
        duration_ema[:, foot_i] = torch.where(valid_event, new_duration, duration_ema[:, foot_i])
        seen[:, foot_i] = seen[:, foot_i] | valid_event

        last_root = torch.where(touchdown, root_x, last_root)
        last_time = torch.where(touchdown, current_time, last_time)
        last_foot = torch.where(touchdown, torch.full_like(last_foot, foot_i), last_foot)

    both_seen = seen[:, 0] & seen[:, 1]
    advance_error = (advance_ema[:, 0] - advance_ema[:, 1]) / max(advance_scale, 1.0e-6)
    duration_error = (duration_ema[:, 0] - duration_ema[:, 1]) / max(duration_scale, 1.0e-6)
    penalty = torch.square(advance_error) + torch.square(duration_error)

    setattr(env, last_root_name, last_root)
    setattr(env, last_time_name, last_time)
    setattr(env, last_foot_name, last_foot)
    setattr(env, advance_ema_name, advance_ema)
    setattr(env, duration_ema_name, duration_ema)
    setattr(env, seen_name, seen)
    return penalty * both_seen.float()


def supported_forward_velocity_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_single_support_fraction: float,
    max_velocity: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward velocity when support is not constant double-stance or flight."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    single_support = torch.sum(in_contact.int(), dim=1) == 1
    double_support = torch.sum(in_contact.int(), dim=1) == 2
    airborne = torch.sum(in_contact.int(), dim=1) == 0
    root_lin_vel_w = quat_apply(asset.data.root_quat_w, asset.data.root_lin_vel_b)
    forward_reward = torch.clamp(root_lin_vel_w[:, 0] / max(max_velocity, 1.0e-6), min=0.0, max=1.0)
    support_reward = single_support.float() + 0.25 * double_support.float() - airborne.float()
    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    return forward_reward * torch.clamp(support_reward, min=0.0) * moving.float()


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
