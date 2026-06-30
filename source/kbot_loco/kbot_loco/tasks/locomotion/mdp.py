from __future__ import annotations

from typing import TYPE_CHECKING

import math

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _debounced_toe_off_gait_cycle_state(
    env: ManagerBasedRLEnv,
    period_s: float,
    toe_off_debounce_s: float,
    sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Track the first debounced toe-off and return a toe-off-anchored phase."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0

    left_contact = in_contact[:, 0]
    right_contact = in_contact[:, 1]
    now = env.episode_length_buf.float() * env.step_dt

    f1_foot_name = "_kbot_gait_cycle_f1_foot_id"
    start_time_name = "_kbot_gait_cycle_start_time"
    f1_foot_id = getattr(env, f1_foot_name, None)
    start_time = getattr(env, start_time_name, None)
    if f1_foot_id is None or f1_foot_id.shape != now.shape or f1_foot_id.device != now.device:
        f1_foot_id = torch.full_like(env.episode_length_buf, -1, dtype=torch.long, device=now.device)
    if start_time is None or start_time.shape != now.shape or start_time.device != now.device:
        start_time = torch.zeros_like(now)

    reset = env.episode_length_buf <= 1
    f1_foot_id = torch.where(reset, torch.full_like(f1_foot_id, -1), f1_foot_id)
    start_time = torch.where(reset, now, start_time)

    debounce_s = max(toe_off_debounce_s, 0.0)
    unassigned = f1_foot_id < 0
    left_takeoff = (~left_contact) & right_contact & (air_time[:, 0] >= debounce_s)
    right_takeoff = (~right_contact) & left_contact & (air_time[:, 1] >= debounce_s)
    assign_left = unassigned & left_takeoff & ~right_takeoff
    assign_right = unassigned & right_takeoff & ~left_takeoff

    assigned_air_time = torch.zeros_like(now)
    assigned_air_time = torch.where(assign_left, air_time[:, 0], assigned_air_time)
    assigned_air_time = torch.where(assign_right, air_time[:, 1], assigned_air_time)
    f1_foot_id = torch.where(assign_left, torch.zeros_like(f1_foot_id), f1_foot_id)
    f1_foot_id = torch.where(assign_right, torch.ones_like(f1_foot_id), f1_foot_id)
    start_time = torch.where(assign_left | assign_right, now - assigned_air_time, start_time)

    assigned = f1_foot_id >= 0
    phase = torch.remainder((now - start_time) / max(period_s, 1.0e-6), 1.0)
    phase = torch.where(assigned, phase, torch.zeros_like(phase))

    setattr(env, f1_foot_name, f1_foot_id)
    setattr(env, start_time_name, start_time)
    return phase, f1_foot_id, assigned, in_contact


def _gait_cycle_phase_state(
    env: ManagerBasedRLEnv,
    period_s: float,
    swing_phase_fraction: float,
    toe_off_debounce_s: float,
    plant_phase_fraction: float,
    sensor_cfg: SceneEntityCfg,
) -> dict[str, torch.Tensor]:
    """Return shared toe-off-anchored gait-cycle phase enables."""
    phase, f1_foot_id, assigned, in_contact = _debounced_toe_off_gait_cycle_state(
        env, period_s, toe_off_debounce_s, sensor_cfg
    )
    left_contact = in_contact[:, 0]
    right_contact = in_contact[:, 1]
    f1_is_left = f1_foot_id == 0
    f1_contact = torch.where(f1_is_left, left_contact, right_contact)
    f2_contact = torch.where(f1_is_left, right_contact, left_contact)

    swing_fraction = min(max(swing_phase_fraction, 1.0e-6), 0.5)
    plant_fraction = min(max(plant_phase_fraction, 0.0), 1.0)
    plant_start = swing_fraction * (1.0 - plant_fraction)

    f1_swing = assigned & (phase < plant_start)
    f1_plant = assigned & (phase >= plant_start) & (phase < swing_fraction)
    f1_shift = assigned & (phase >= swing_fraction) & (phase < 0.5)
    f2_swing = assigned & (phase >= 0.5) & (phase < 0.5 + plant_start)
    f2_plant = assigned & (phase >= 0.5 + plant_start) & (phase < 0.5 + swing_fraction)
    f2_shift = assigned & (phase >= 0.5 + swing_fraction)

    return {
        "phase": phase,
        "f1_foot_id": f1_foot_id,
        "f1_is_left": f1_is_left,
        "assigned": assigned,
        "in_contact": in_contact,
        "f1_contact": f1_contact,
        "f2_contact": f2_contact,
        "f1_swing": f1_swing,
        "f1_plant": f1_plant,
        "f1_shift": f1_shift,
        "f2_swing": f2_swing,
        "f2_plant": f2_plant,
        "f2_shift": f2_shift,
    }


def _select_f1_f2(values: torch.Tensor, f1_is_left: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select f1/f2 values from a left/right tensor."""
    f1_value = torch.where(f1_is_left, values[:, 0], values[:, 1])
    f2_value = torch.where(f1_is_left, values[:, 1], values[:, 0])
    return f1_value, f2_value


def gait_phase(
    env: ManagerBasedRLEnv,
    period_s: float,
    sensor_cfg: SceneEntityCfg | None = None,
    start_on_first_toe_off: bool = False,
    toe_off_debounce_s: float = 0.0,
) -> torch.Tensor:
    """Return sin/cos phase features for a nominal walking cycle."""
    if start_on_first_toe_off:
        if sensor_cfg is None:
            raise ValueError("sensor_cfg is required when start_on_first_toe_off=True")
        phase, f1_foot_id, assigned, _ = _debounced_toe_off_gait_cycle_state(
            env, period_s, toe_off_debounce_s, sensor_cfg
        )
    else:
        phase = torch.remainder(env.episode_length_buf.float() * env.step_dt / period_s, 1.0)
    phase_angle = 2.0 * torch.pi * phase
    phase_features = torch.stack((torch.sin(phase_angle), torch.cos(phase_angle)), dim=1)
    if start_on_first_toe_off:
        side_sign = torch.where(f1_foot_id == 0, torch.ones_like(phase), -torch.ones_like(phase))
        phase_features = phase_features * side_sign.unsqueeze(1)
        phase_features = torch.where(assigned.unsqueeze(1), phase_features, torch.zeros_like(phase_features))
    return phase_features


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


def gait_cycle_support_reward(
    env: ManagerBasedRLEnv,
    period_s: float,
    swing_phase_fraction: float,
    toe_off_debounce_s: float,
    plant_phase_fraction: float,
    swing_double_support_reward: float,
    shift_single_support_reward: float,
    airborne_penalty: float,
    wrong_single_penalty: float,
    precycle_single_support_reward: float,
    precycle_double_support_reward: float,
    precycle_airborne_penalty: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward support contacts against a debounced, toe-off-anchored gait cycle.

    Swing/shift phases are phase-latched: one wrong sampled contact state makes
    the rest of that phase score as bad. Plant is a landing transition, so its
    correct single-support credit fades to zero instead of latching.
    """
    state = _gait_cycle_phase_state(
        env, period_s, swing_phase_fraction, toe_off_debounce_s, plant_phase_fraction, sensor_cfg
    )
    phase = state["phase"]
    assigned = state["assigned"]
    in_contact = state["in_contact"]
    f1_contact = state["f1_contact"]
    f2_contact = state["f2_contact"]

    double_support = f1_contact & f2_contact
    airborne = ~f1_contact & ~f2_contact
    f1_single = f1_contact & ~f2_contact
    f2_single = f2_contact & ~f1_contact

    left_contact = in_contact[:, 0]
    right_contact = in_contact[:, 1]
    precycle_single = left_contact ^ right_contact
    precycle_double = left_contact & right_contact
    precycle_airborne = ~left_contact & ~right_contact
    precycle_reward = (
        precycle_single_support_reward * precycle_single.float()
        + precycle_double_support_reward * precycle_double.float()
        - precycle_airborne_penalty * precycle_airborne.float()
    )

    swing_fraction = min(max(swing_phase_fraction, 1.0e-6), 0.5)
    plant_fraction = min(max(plant_phase_fraction, 0.0), 1.0)
    plant_start = swing_fraction * (1.0 - plant_fraction)
    plant_duration = max(swing_fraction - plant_start, 1.0e-6)
    f1_plant_t = torch.clamp((phase - plant_start) / plant_duration, min=0.0, max=1.0)
    f2_plant_t = torch.clamp((phase - (0.5 + plant_start)) / plant_duration, min=0.0, max=1.0)
    f1_plant_single_credit = 1.0 - f1_plant_t
    f2_plant_single_credit = 1.0 - f2_plant_t

    ones = torch.ones_like(phase)
    zeros = torch.zeros_like(phase)
    bad = -ones
    f1_swing_nominal = torch.where(f2_single, ones, bad)
    f2_swing_nominal = torch.where(f1_single, ones, bad)
    f1_shift_nominal = torch.where(double_support, ones, bad)
    f2_shift_nominal = torch.where(double_support, ones, bad)

    f1_plant_reward = torch.where(
        f2_single,
        f1_plant_single_credit,
        torch.where(double_support, zeros, bad),
    )
    f2_plant_reward = torch.where(
        f1_single,
        f2_plant_single_credit,
        torch.where(double_support, zeros, bad),
    )

    phase_id = torch.full_like(env.episode_length_buf, -1, dtype=torch.long, device=phase.device)
    phase_id = torch.where(state["f1_swing"], torch.full_like(phase_id, 0), phase_id)
    phase_id = torch.where(state["f1_plant"], torch.full_like(phase_id, 1), phase_id)
    phase_id = torch.where(state["f1_shift"], torch.full_like(phase_id, 2), phase_id)
    phase_id = torch.where(state["f2_swing"], torch.full_like(phase_id, 3), phase_id)
    phase_id = torch.where(state["f2_plant"], torch.full_like(phase_id, 4), phase_id)
    phase_id = torch.where(state["f2_shift"], torch.full_like(phase_id, 5), phase_id)

    previous_phase_name = "_kbot_gait_cycle_support_previous_phase_id"
    bad_phase_name = "_kbot_gait_cycle_support_bad_phase_seen"
    previous_phase = getattr(env, previous_phase_name, None)
    bad_phase_seen = getattr(env, bad_phase_name, None)
    if previous_phase is None or previous_phase.shape != phase_id.shape or previous_phase.device != phase_id.device:
        previous_phase = phase_id.clone()
    if bad_phase_seen is None or bad_phase_seen.shape != assigned.shape or bad_phase_seen.device != assigned.device:
        bad_phase_seen = torch.zeros_like(assigned)

    reset = env.episode_length_buf <= 1
    phase_changed = reset | (phase_id != previous_phase) | ~assigned
    bad_phase_seen = torch.where(phase_changed, torch.zeros_like(bad_phase_seen), bad_phase_seen)

    latch_active = state["f1_swing"] | state["f1_shift"] | state["f2_swing"] | state["f2_shift"]
    wrong_now = (
        (state["f1_swing"] & ~f2_single)
        | (state["f1_shift"] & ~double_support)
        | (state["f2_swing"] & ~f1_single)
        | (state["f2_shift"] & ~double_support)
    )
    bad_phase_seen = bad_phase_seen | (latch_active & wrong_now)

    swing_shift_reward = (
        state["f1_swing"].float() * f1_swing_nominal
        + state["f1_shift"].float() * f1_shift_nominal
        + state["f2_swing"].float() * f2_swing_nominal
        + state["f2_shift"].float() * f2_shift_nominal
    )
    swing_shift_reward = torch.where(latch_active & bad_phase_seen, bad, swing_shift_reward)
    plant_reward = (
        state["f1_plant"].float() * f1_plant_reward
        + state["f2_plant"].float() * f2_plant_reward
    )
    reward = swing_shift_reward + plant_reward

    setattr(env, previous_phase_name, phase_id)
    setattr(env, bad_phase_name, bad_phase_seen)
    return torch.where(assigned, reward, precycle_reward)


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


def _full_cycle_duration_ema(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    smoothing_cycles: float,
    min_cycle_duration_s: float,
    max_cycle_duration_s: float,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a shared per-env EMA of same-foot touchdown cycle duration."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    now = env.episode_length_buf.to(reference.dtype) * env.step_dt

    last_touchdown_name = "_kbot_full_cycle_duration_touchdown_time"
    duration_ema_name = "_kbot_full_cycle_duration_ema"
    seen_touchdown_name = "_kbot_full_cycle_duration_seen_touchdown"
    seen_duration_name = "_kbot_full_cycle_duration_seen_duration"
    update_step_name = "_kbot_full_cycle_duration_update_step"

    last_touchdown = getattr(env, last_touchdown_name, None)
    duration_ema = getattr(env, duration_ema_name, None)
    seen_touchdown = getattr(env, seen_touchdown_name, None)
    seen_duration = getattr(env, seen_duration_name, None)
    update_step = getattr(env, update_step_name, None)

    shape = first_contact.shape
    if last_touchdown is None or last_touchdown.shape != shape or last_touchdown.device != reference.device:
        last_touchdown = torch.zeros(shape, dtype=reference.dtype, device=reference.device)
    if duration_ema is None or duration_ema.shape != shape or duration_ema.device != reference.device:
        duration_ema = torch.zeros(shape, dtype=reference.dtype, device=reference.device)
    if seen_touchdown is None or seen_touchdown.shape != shape or seen_touchdown.device != reference.device:
        seen_touchdown = torch.zeros(shape, dtype=torch.bool, device=reference.device)
    if seen_duration is None or seen_duration.shape != shape or seen_duration.device != reference.device:
        seen_duration = torch.zeros(shape, dtype=torch.bool, device=reference.device)
    if update_step is None or update_step.shape != env.episode_length_buf.shape or update_step.device != reference.device:
        update_step = torch.full_like(env.episode_length_buf, -1)

    current_step = env.episode_length_buf
    reset = current_step <= 1
    needs_update = update_step != current_step
    last_touchdown = torch.where(reset[:, None], now[:, None].expand_as(last_touchdown), last_touchdown)
    duration_ema = torch.where(reset[:, None], torch.zeros_like(duration_ema), duration_ema)
    seen_touchdown = torch.where(reset[:, None], torch.zeros_like(seen_touchdown), seen_touchdown)
    seen_duration = torch.where(reset[:, None], torch.zeros_like(seen_duration), seen_duration)

    alpha = min(max(1.0 / max(smoothing_cycles, 1.0e-6), 0.0), 1.0)
    for foot_i in range(first_contact.shape[1]):
        touchdown = first_contact[:, foot_i] & needs_update
        measured = torch.clamp(now - last_touchdown[:, foot_i], min=min_cycle_duration_s, max=max_cycle_duration_s)
        valid_cycle = touchdown & seen_touchdown[:, foot_i] & (now - last_touchdown[:, foot_i] > env.step_dt)
        updated = torch.where(
            seen_duration[:, foot_i],
            (1.0 - alpha) * duration_ema[:, foot_i] + alpha * measured,
            measured,
        )
        duration_ema[:, foot_i] = torch.where(valid_cycle, updated, duration_ema[:, foot_i])
        seen_duration[:, foot_i] = seen_duration[:, foot_i] | valid_cycle
        last_touchdown[:, foot_i] = torch.where(touchdown, now, last_touchdown[:, foot_i])
        seen_touchdown[:, foot_i] = seen_touchdown[:, foot_i] | touchdown

    update_step = torch.where(needs_update, current_step, update_step)
    setattr(env, last_touchdown_name, last_touchdown)
    setattr(env, duration_ema_name, duration_ema)
    setattr(env, seen_touchdown_name, seen_touchdown)
    setattr(env, seen_duration_name, seen_duration)
    setattr(env, update_step_name, update_step)

    has_duration = torch.any(seen_duration, dim=1)
    duration_sum = torch.sum(torch.where(seen_duration, duration_ema, torch.zeros_like(duration_ema)), dim=1)
    duration_count = torch.clamp(torch.sum(seen_duration.float(), dim=1), min=1.0)
    mean_duration = duration_sum / duration_count
    return mean_duration, has_duration


def _adaptive_ema_alpha(
    env: ManagerBasedRLEnv,
    tau_s: float,
    reference: torch.Tensor,
    sensor_cfg: SceneEntityCfg | None,
    ema_cycle_count: float,
    cycle_duration_smoothing_cycles: float,
    min_cycle_duration_s: float,
    max_cycle_duration_s: float,
    min_tau_s: float,
    max_tau_s: float,
) -> torch.Tensor:
    """Compute fixed-time or measured-cycle adaptive EMA alpha per environment."""
    fallback_tau = max(tau_s, 1.0e-6)
    tau = torch.full((env.num_envs,), fallback_tau, dtype=reference.dtype, device=reference.device)
    if sensor_cfg is not None and ema_cycle_count > 0.0:
        cycle_duration, has_duration = _full_cycle_duration_ema(
            env,
            sensor_cfg,
            cycle_duration_smoothing_cycles,
            min_cycle_duration_s,
            max_cycle_duration_s,
            reference,
        )
        measured_tau = torch.clamp(cycle_duration * ema_cycle_count, min=min_tau_s, max=max_tau_s)
        tau = torch.where(has_duration, measured_tau, tau)
    return torch.clamp(env.step_dt / tau, min=0.0, max=1.0)


def root_lateral_tilt_ema_l2(
    env: ManagerBasedRLEnv,
    tau_s: float,
    sensor_cfg: SceneEntityCfg | None = None,
    ema_cycle_count: float = 0.0,
    cycle_duration_smoothing_cycles: float = 5.0,
    min_cycle_duration_s: float = 0.25,
    max_cycle_duration_s: float = 2.0,
    min_tau_s: float = 0.75,
    max_tau_s: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize persistent side lean with an exponential moving average."""
    asset = env.scene[asset_cfg.name]
    tilt = asset.data.projected_gravity_b[:, 1]
    buffer_name = "_kbot_root_lateral_tilt_ema"
    ema = getattr(env, buffer_name, None)
    if ema is None or ema.shape != tilt.shape or ema.device != tilt.device:
        ema = torch.zeros_like(tilt)

    alpha = _adaptive_ema_alpha(
        env,
        tau_s,
        tilt,
        sensor_cfg,
        ema_cycle_count,
        cycle_duration_smoothing_cycles,
        min_cycle_duration_s,
        max_cycle_duration_s,
        min_tau_s,
        max_tau_s,
    )
    reset = env.episode_length_buf <= 1
    ema = torch.where(reset, tilt, (1.0 - alpha) * ema + alpha * tilt)
    setattr(env, buffer_name, ema)
    return torch.square(ema)


def signed_joint_pair_ema_symmetry_l2(
    env: ManagerBasedRLEnv,
    joint_pairs: list[tuple[str, str, float]],
    tau_s: float,
    sensor_cfg: SceneEntityCfg | None = None,
    ema_cycle_count: float = 0.0,
    cycle_duration_smoothing_cycles: float = 5.0,
    min_cycle_duration_s: float = 0.25,
    max_cycle_duration_s: float = 2.0,
    min_tau_s: float = 0.75,
    max_tau_s: float = 10.0,
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

    alpha = _adaptive_ema_alpha(
        env,
        tau_s,
        error,
        sensor_cfg,
        ema_cycle_count,
        cycle_duration_smoothing_cycles,
        min_cycle_duration_s,
        max_cycle_duration_s,
        min_tau_s,
        max_tau_s,
    ).unsqueeze(1)
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


def _centerline_gaussian_gate(
    env: ManagerBasedRLEnv,
    target_y: float,
    width_sq: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return a smooth gate that is one on the center line and decays with lateral drift."""
    asset = env.scene[asset_cfg.name]
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is None:
        lateral_pos = asset.data.root_pos_w[:, 1]
    else:
        lateral_pos = asset.data.root_pos_w[:, 1] - env_origins[:, 1]
    return torch.exp(-torch.square(lateral_pos - target_y) / max(width_sq, 1.0e-6))


def _world_heading_gaussian_gate(
    env: ManagerBasedRLEnv,
    width_sq: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return one when the root faces world +X and decay as yaw diverges."""
    asset = env.scene[asset_cfg.name]
    forward_b = torch.zeros(env.num_envs, 3, device=asset.data.root_pos_w.device)
    forward_b[:, 0] = 1.0
    forward_w = quat_apply(asset.data.root_quat_w, forward_b)
    heading_error_sq = torch.square(forward_w[:, 1]) + torch.square(torch.clamp(-forward_w[:, 0], min=0.0))
    return torch.exp(-heading_error_sq / max(width_sq, 1.0e-6))


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


def upright_centerline_gated_track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    minimum_height: float,
    max_tilt: float,
    centerline_target_y: float,
    centerline_width_sq: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward commanded body-frame velocity tracking only while upright and near world y=0."""
    base_reward = upright_gated_track_lin_vel_xy_exp(env, std, command_name, minimum_height, max_tilt, asset_cfg)
    return base_reward * _centerline_gaussian_gate(env, centerline_target_y, centerline_width_sq, asset_cfg)


def upright_centerline_heading_gated_track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    minimum_height: float,
    max_tilt: float,
    centerline_target_y: float,
    centerline_width_sq: float,
    heading_width_sq: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward body-frame velocity tracking only while upright, centered, and facing world +X."""
    base_reward = upright_gated_track_lin_vel_xy_exp(env, std, command_name, minimum_height, max_tilt, asset_cfg)
    centerline_gate = _centerline_gaussian_gate(env, centerline_target_y, centerline_width_sq, asset_cfg)
    heading_gate = _world_heading_gaussian_gate(env, heading_width_sq, asset_cfg)
    return base_reward * centerline_gate * heading_gate


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


def upright_centerline_gated_world_forward_velocity_clip(
    env: ManagerBasedRLEnv,
    max_velocity: float,
    minimum_height: float,
    max_tilt: float,
    centerline_target_y: float,
    centerline_width_sq: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward clipped world +X progress only while upright and near world y=0."""
    base_reward = upright_gated_world_forward_velocity_clip(env, max_velocity, minimum_height, max_tilt, asset_cfg)
    return base_reward * _centerline_gaussian_gate(env, centerline_target_y, centerline_width_sq, asset_cfg)


def upright_centerline_heading_gated_world_forward_velocity_clip(
    env: ManagerBasedRLEnv,
    max_velocity: float,
    minimum_height: float,
    max_tilt: float,
    centerline_target_y: float,
    centerline_width_sq: float,
    heading_width_sq: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward clipped world +X progress only while upright, centered, and facing world +X."""
    base_reward = upright_gated_world_forward_velocity_clip(env, max_velocity, minimum_height, max_tilt, asset_cfg)
    centerline_gate = _centerline_gaussian_gate(env, centerline_target_y, centerline_width_sq, asset_cfg)
    heading_gate = _world_heading_gaussian_gate(env, heading_width_sq, asset_cfg)
    return base_reward * centerline_gate * heading_gate


def swing_sole_clearance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float,
    drag_floor: float,
    drag_weight: float,
    over_height: float,
    over_scale: float,
    over_weight: float,
    over_penalty_cap: float,
    minimum_height: float,
    max_tilt: float,
    period_s: float,
    swing_phase_fraction: float,
    toe_off_debounce_s: float,
    plant_phase_fraction: float,
    foot_local_offsets: list[tuple[float, float, float]],
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward newly reached swing clearance and penalize dragging or over-kicking during single support."""
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

    state = _gait_cycle_phase_state(
        env, period_s, swing_phase_fraction, toe_off_debounce_s, plant_phase_fraction, sensor_cfg
    )
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
    over_penalty = over_weight * torch.square(
        torch.clamp(sole_height - over_height, min=0.0) / max(over_scale, 1.0e-6)
    )
    over_penalty = torch.clamp(over_penalty, max=over_penalty_cap)

    per_foot_reward = clearance_reward - drag_penalty - over_penalty
    f1_reward, f2_reward = _select_f1_f2(per_foot_reward, state["f1_is_left"])
    f1_single_support_swing = ~state["f1_contact"] & state["f2_contact"]
    f2_single_support_swing = ~state["f2_contact"] & state["f1_contact"]
    reward = (
        state["f1_swing"].float() * f1_single_support_swing.float() * f1_reward
        + state["f2_swing"].float() * f2_single_support_swing.float() * f2_reward
    )
    return reward * moving.float() * upright


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
    ema_cycle_count: float = 0.0,
    cycle_duration_smoothing_cycles: float = 5.0,
    min_cycle_duration_s: float = 0.25,
    max_cycle_duration_s: float = 2.0,
    min_tau_s: float = 0.75,
    max_tau_s: float = 10.0,
) -> torch.Tensor:
    """Penalize persistent left/right stance-duty imbalance."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    contact = (contact_time > 0.0).float()

    buffer_name = "_kbot_contact_duty_symmetry_ema"
    ema = getattr(env, buffer_name, None)
    if ema is None or ema.shape != contact.shape or ema.device != contact.device:
        ema = torch.zeros_like(contact)

    alpha = _adaptive_ema_alpha(
        env,
        tau_s,
        contact,
        sensor_cfg,
        ema_cycle_count,
        cycle_duration_smoothing_cycles,
        min_cycle_duration_s,
        max_cycle_duration_s,
        min_tau_s,
        max_tau_s,
    ).unsqueeze(1)
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
    ema_cycle_count: float = 0.0,
    cycle_duration_smoothing_cycles: float = 5.0,
    min_cycle_duration_s: float = 0.25,
    max_cycle_duration_s: float = 2.0,
    min_tau_s: float = 0.75,
    max_tau_s: float = 10.0,
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

    alpha = _adaptive_ema_alpha(
        env,
        tau_s,
        root_x,
        sensor_cfg,
        ema_cycle_count,
        cycle_duration_smoothing_cycles,
        min_cycle_duration_s,
        max_cycle_duration_s,
        min_tau_s,
        max_tau_s,
    )
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


def alternating_step_duration_ema_l1(
    env: ManagerBasedRLEnv,
    target_duration_s: float,
    smoothing_events: float,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    minimum_command_speed: float = 0.05,
    min_duration_s: float = 0.05,
    max_duration_s: float = 2.0,
    penalty_cap: float = 1.0,
) -> torch.Tensor:
    """Penalize the worst per-foot EMA of alternating-step duration error."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    device = first_contact.device
    if first_contact.shape[1] < 2:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=device)

    reference = env.episode_length_buf.to(dtype=torch.float32, device=device)
    current_time = reference * env.step_dt

    last_time_name = "_kbot_step_duration_target_touchdown_time"
    last_foot_name = "_kbot_step_duration_target_touchdown_foot"
    error_ema_name = "_kbot_step_duration_target_error_ema"
    seen_name = "_kbot_step_duration_target_seen"

    last_time = getattr(env, last_time_name, None)
    last_foot = getattr(env, last_foot_name, None)
    error_ema = getattr(env, error_ema_name, None)
    seen = getattr(env, seen_name, None)

    if last_time is None or last_time.shape != current_time.shape or last_time.device != current_time.device:
        last_time = current_time.clone()
    if last_foot is None or last_foot.shape != current_time.shape or last_foot.device != current_time.device:
        last_foot = torch.full((env.num_envs,), -1, dtype=torch.long, device=device)
    if error_ema is None or error_ema.shape != first_contact.shape or error_ema.device != current_time.device:
        error_ema = torch.zeros(env.num_envs, first_contact.shape[1], dtype=current_time.dtype, device=device)
    if seen is None or seen.shape != first_contact.shape or seen.device != current_time.device:
        seen = torch.zeros(env.num_envs, first_contact.shape[1], dtype=torch.bool, device=device)

    reset = env.episode_length_buf <= 1
    last_time = torch.where(reset, current_time, last_time)
    last_foot = torch.where(reset, torch.full_like(last_foot, -1), last_foot)
    error_ema = torch.where(reset[:, None], torch.zeros_like(error_ema), error_ema)
    seen = torch.where(reset[:, None], torch.zeros_like(seen), seen)

    previous_time = last_time.clone()
    previous_foot = last_foot.clone()
    alpha = min(max(1.0 / max(smoothing_events, 1.0e-6), 0.0), 1.0)
    target_duration = max(target_duration_s, 1.0e-6)

    for foot_i in range(first_contact.shape[1]):
        touchdown = first_contact[:, foot_i]
        alternating = (previous_foot >= 0) & (previous_foot != foot_i)
        valid_event = touchdown & alternating

        duration = torch.clamp(current_time - previous_time, min=min_duration_s, max=max_duration_s)
        event_error = torch.abs(duration - target_duration) / target_duration
        if penalty_cap > 0.0:
            event_error = torch.clamp(event_error, max=penalty_cap)

        prior_seen = seen[:, foot_i]
        updated = torch.where(prior_seen, (1.0 - alpha) * error_ema[:, foot_i] + alpha * event_error, event_error)
        error_ema[:, foot_i] = torch.where(valid_event, updated, error_ema[:, foot_i])
        seen[:, foot_i] = seen[:, foot_i] | valid_event

        last_time = torch.where(touchdown, current_time, last_time)
        last_foot = torch.where(touchdown, torch.full_like(last_foot, foot_i), last_foot)

    command = env.command_manager.get_command(command_name)
    moving = torch.norm(command[:, :2], dim=1) > minimum_command_speed
    both_seen = seen[:, 0] & seen[:, 1]
    penalty = torch.maximum(error_ema[:, 0], error_ema[:, 1])

    setattr(env, last_time_name, last_time)
    setattr(env, last_foot_name, last_foot)
    setattr(env, error_ema_name, error_ema)
    setattr(env, seen_name, seen)
    return penalty * both_seen.float() * moving.float()


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
    foot_local_offsets: list[tuple[float, float, float]] | None = None,
) -> torch.Tensor:
    """Penalize feet crossing or collapsing into a narrow support line."""
    asset = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if foot_local_offsets is not None:
        foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
        local_offsets = torch.tensor(foot_local_offsets, dtype=feet_pos_w.dtype, device=feet_pos_w.device)
        feet_pos_w = feet_pos_w + quat_apply(
            foot_quat_w.reshape(-1, 4), local_offsets[None, :, :].expand(env.num_envs, -1, -1).reshape(-1, 3)
        ).reshape(env.num_envs, len(asset_cfg.body_ids), 3)
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
    foot_local_offsets: list[tuple[float, float, float]] | None = None,
) -> torch.Tensor:
    """Penalize crossed feet by preserving left-foot/right-foot lateral ordering."""
    asset = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if foot_local_offsets is not None:
        foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
        local_offsets = torch.tensor(foot_local_offsets, dtype=feet_pos_w.dtype, device=feet_pos_w.device)
        feet_pos_w = feet_pos_w + quat_apply(
            foot_quat_w.reshape(-1, 4), local_offsets[None, :, :].expand(env.num_envs, -1, -1).reshape(-1, 3)
        ).reshape(env.num_envs, len(asset_cfg.body_ids), 3)
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
    foot_local_offsets: list[tuple[float, float, float]] | None = None,
) -> torch.Tensor:
    """Penalize feet leaving their neutral body-frame lateral lanes."""
    asset = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if foot_local_offsets is not None:
        foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
        local_offsets = torch.tensor(foot_local_offsets, dtype=feet_pos_w.dtype, device=feet_pos_w.device)
        feet_pos_w = feet_pos_w + quat_apply(
            foot_quat_w.reshape(-1, 4), local_offsets[None, :, :].expand(env.num_envs, -1, -1).reshape(-1, 3)
        ).reshape(env.num_envs, len(asset_cfg.body_ids), 3)
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
    foot_local_offsets: list[tuple[float, float, float]] | None = None,
) -> torch.Tensor:
    """Penalize the worst foot's body-frame lateral lane error."""
    asset = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if foot_local_offsets is not None:
        foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
        local_offsets = torch.tensor(foot_local_offsets, dtype=feet_pos_w.dtype, device=feet_pos_w.device)
        feet_pos_w = feet_pos_w + quat_apply(
            foot_quat_w.reshape(-1, 4), local_offsets[None, :, :].expand(env.num_envs, -1, -1).reshape(-1, 3)
        ).reshape(env.num_envs, len(asset_cfg.body_ids), 3)
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
    first_target_fraction: float = 1.0,
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
    in_contact = contact_time > 0.0
    single_stance = torch.sum(in_contact.int(), dim=1) == 1

    previous_contact_name = "_kbot_foot_sagittal_previous_contact"
    has_takeoff_name = "_kbot_foot_sagittal_has_takeoff"
    first_target_done_name = "_kbot_foot_sagittal_first_target_done"
    previous_contact = getattr(env, previous_contact_name, None)
    has_takeoff = getattr(env, has_takeoff_name, None)
    first_target_done = getattr(env, first_target_done_name, None)
    if previous_contact is None or previous_contact.shape != in_contact.shape or previous_contact.device != in_contact.device:
        previous_contact = in_contact.clone()
    if has_takeoff is None or has_takeoff.shape != in_contact.shape or has_takeoff.device != in_contact.device:
        has_takeoff = torch.zeros_like(in_contact)
    if first_target_done is None or first_target_done.shape != env.episode_length_buf.shape or first_target_done.device != in_contact.device:
        first_target_done = torch.zeros(env.num_envs, dtype=torch.bool, device=in_contact.device)

    reset = (env.episode_length_buf <= 1).unsqueeze(1)
    previous_contact = torch.where(reset, in_contact, previous_contact)
    has_takeoff = torch.where(reset, torch.zeros_like(has_takeoff), has_takeoff)
    first_target_done = torch.where(reset.squeeze(1), torch.zeros_like(first_target_done), first_target_done)

    takeoff = previous_contact & ~in_contact
    has_takeoff = torch.where(takeoff, torch.ones_like(has_takeoff), has_takeoff)
    first_length = target_length * min(max(first_target_fraction, 0.0), 1.0)
    active_target = torch.where(
        first_target_done,
        torch.full_like(step_length, target_length),
        torch.full_like(step_length, first_length),
    )
    penalty = torch.clamp(active_target - step_length, min=0.0) * single_stance

    touchdown = in_contact & has_takeoff
    first_target_done = first_target_done | torch.any(touchdown, dim=1)
    has_takeoff = torch.where(in_contact, torch.zeros_like(has_takeoff), has_takeoff)
    setattr(env, previous_contact_name, in_contact)
    setattr(env, has_takeoff_name, has_takeoff)
    setattr(env, first_target_done_name, first_target_done)
    return penalty


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


def _signed_swing_step_lead(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    root_pos_w = asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    feet_pos_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), (feet_pos_w - root_pos_w).reshape(-1, 3)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 3
    )

    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_air = ~in_contact
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    left_swing = in_air[:, 0] & in_contact[:, 1] & single_stance
    right_swing = in_air[:, 1] & in_contact[:, 0] & single_stance
    swing_mask = torch.stack((left_swing, right_swing), dim=1)

    left_lead = feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0]
    right_lead = feet_pos_b[:, 1, 0] - feet_pos_b[:, 0, 0]
    signed_lead = torch.stack((left_lead, right_lead), dim=1)
    return signed_lead, swing_mask, asset.data.root_pos_w


def dense_swing_step_length_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_length: float,
    crossover_fraction: float,
    linear_gain: float,
    lambda_per_m: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Dense shaped reward for the swing foot leading the stance foot."""
    signed_lead, swing_mask, _ = _signed_swing_step_lead(env, sensor_cfg, asset_cfg)
    command = env.command_manager.get_command(command_name)
    moving = torch.norm(command[:, :2], dim=1) > 0.05

    target = max(target_length, 1.0e-6)
    lead = torch.clamp(signed_lead, min=0.0, max=target)
    lambda_value = 1.0 / target if lambda_per_m <= 0.0 else lambda_per_m
    linear_value = (1.0 / target) if linear_gain <= 0.0 else linear_gain
    crossover = min(max(crossover_fraction, 1.0e-6), 1.0)
    crossover_lead = crossover * target
    exp_at_crossover = max(math.exp(lambda_value * crossover_lead) - 1.0, 1.0e-6)
    exp_gain = linear_value * crossover_lead / exp_at_crossover

    shaped = linear_value * lead + exp_gain * torch.expm1(lambda_value * lead)
    reward = torch.sum(shaped * swing_mask.float(), dim=1)
    return reward * moving.float()


def dense_foot_swing_speed_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_length: float,
    target_air_time: float,
    max_step_credit: float,
    target_left_y: float,
    target_right_y: float,
    y_scale: float,
    y_linear_radius: float,
    min_height: float,
    max_height: float,
    z_scale: float,
    minimum_height: float,
    max_tilt: float,
    foot_local_offsets: list[tuple[float, float, float]],
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """Dense swing-foot speed reward gated by sole track and low clearance."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    _signed_lead, swing_mask, _ = _signed_swing_step_lead(env, sensor_cfg, asset_cfg)
    command = env.command_manager.get_command(command_name)
    moving = torch.norm(command[:, :2], dim=1) > 0.05
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    foot_vx_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 0]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    offsets = torch.tensor(foot_local_offsets, dtype=foot_pos_w.dtype, device=foot_pos_w.device)
    sole_pos_w = foot_pos_w + quat_apply(foot_quat_w, offsets.unsqueeze(0).expand(env.num_envs, -1, -1))

    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is None:
        origin_y = torch.zeros(env.num_envs, dtype=sole_pos_w.dtype, device=sole_pos_w.device)
        origin_z = torch.zeros(env.num_envs, dtype=sole_pos_w.dtype, device=sole_pos_w.device)
    else:
        origin_y = env_origins[:, 1].to(dtype=sole_pos_w.dtype, device=sole_pos_w.device)
        origin_z = env_origins[:, 2].to(dtype=sole_pos_w.dtype, device=sole_pos_w.device)

    target_y = torch.stack((origin_y + target_left_y, origin_y + target_right_y), dim=1)
    y_error = torch.abs(sole_pos_w[:, :, 1] - target_y)
    y_exp_score = torch.exp(-torch.square(y_error / max(y_scale, 1.0e-6)))
    y_linear_score = torch.clamp(1.0 - y_error / max(y_linear_radius, 1.0e-6), min=0.0, max=1.0)
    y_score = torch.maximum(y_exp_score, y_linear_score)

    sole_height = sole_pos_w[:, :, 2] - origin_z.unsqueeze(1)
    z_error = torch.clamp(min_height - sole_height, min=0.0) + torch.clamp(sole_height - max_height, min=0.0)
    z_score = torch.exp(-torch.square(z_error / max(z_scale, 1.0e-6)))

    watermark_name = "_kbot_dense_foot_swing_speed_wm_vx"
    wm_vx = getattr(env, watermark_name, None)
    if wm_vx is None or wm_vx.shape != foot_vx_w.shape or wm_vx.device != foot_vx_w.device:
        wm_vx = torch.zeros_like(foot_vx_w)

    target_swing_time = max(target_air_time, 1.0e-6)
    target_avg_speed = max(target_length / (0.5 * target_swing_time), 1.0e-6)
    reset = (env.episode_length_buf <= 1).unsqueeze(1)
    early_swing = air_time <= (0.5 * target_swing_time)
    upright = _upright_health_gate(env, minimum_height, max_tilt, asset_cfg)
    active = swing_mask & early_swing & ~reset & (upright[:, None] > 0.0)

    current_vx = torch.clamp(foot_vx_w, min=0.0)
    wm_vx = torch.where(active, torch.maximum(wm_vx, current_vx), torch.zeros_like(wm_vx))
    credit = torch.clamp(wm_vx / target_avg_speed, min=0.0, max=max_step_credit)
    reward = torch.sum(credit * y_score * z_score * active.float(), dim=1) * moving.float()

    setattr(env, watermark_name, wm_vx)
    return reward


def foot_retreat(
    env: ManagerBasedRLEnv,
    retreat_epsilon: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
) -> torch.Tensor:
    """One-shot penalty when a contacted foot falls behind its max airborne x."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_air = ~in_contact
    foot_x = asset.data.body_pos_w[:, asset_cfg.body_ids, 0]

    air_peak_x_name = "_kbot_foot_retreat_air_peak_x"
    stance_anchor_x_name = "_kbot_foot_retreat_stance_anchor_x"
    retreat_seen_name = "_kbot_foot_retreat_seen"
    air_peak_x = getattr(env, air_peak_x_name, None)
    stance_anchor_x = getattr(env, stance_anchor_x_name, None)
    retreat_seen = getattr(env, retreat_seen_name, None)
    if air_peak_x is None or air_peak_x.shape != foot_x.shape or air_peak_x.device != foot_x.device:
        air_peak_x = foot_x.clone()
    if stance_anchor_x is None or stance_anchor_x.shape != foot_x.shape or stance_anchor_x.device != foot_x.device:
        stance_anchor_x = foot_x.clone()
    if retreat_seen is None or retreat_seen.shape != in_contact.shape or retreat_seen.device != in_contact.device:
        retreat_seen = torch.zeros_like(in_contact)

    reset = (env.episode_length_buf <= 1).unsqueeze(1)
    air_peak_x = torch.where(reset, foot_x, air_peak_x)
    stance_anchor_x = torch.where(reset, foot_x, stance_anchor_x)
    retreat_seen = torch.where(reset, torch.zeros_like(retreat_seen), retreat_seen)

    active_air_peak_x = torch.where(in_air, torch.maximum(air_peak_x, foot_x), air_peak_x)
    stance_anchor_x = torch.where(first_contact, active_air_peak_x, stance_anchor_x)
    retreat_seen = torch.where(first_contact | ~in_contact, torch.zeros_like(retreat_seen), retreat_seen)

    retreated = in_contact & ~retreat_seen & (foot_x < (stance_anchor_x - retreat_epsilon))
    penalty = torch.sum(retreated.float(), dim=1)
    retreat_seen = retreat_seen | retreated
    air_peak_x = torch.where(in_air, active_air_peak_x, foot_x)

    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    setattr(env, air_peak_x_name, air_peak_x)
    setattr(env, stance_anchor_x_name, stance_anchor_x)
    setattr(env, retreat_seen_name, retreat_seen)
    return penalty * moving.float() / max(env.step_dt, 1.0e-6)


def dense_swing_foot_target_location_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_length: float,
    target_left_y: float,
    target_right_y: float,
    x_scale: float,
    y_scale: float,
    minimum_height: float,
    max_tilt: float,
    linear_progress_scale: float,
    smooth_max_lambda: float,
    first_target_fraction: float,
    period_s: float,
    swing_phase_fraction: float,
    toe_off_debounce_s: float,
    plant_phase_fraction: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
    foot_local_offsets: list[tuple[float, float, float]] | None = None,
) -> torch.Tensor:
    """Dense swing reward for moving the foot toward a world-frame target stride and lane."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if foot_local_offsets is not None:
        foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
        local_offsets = torch.tensor(foot_local_offsets, dtype=foot_pos_w.dtype, device=foot_pos_w.device)
        foot_pos_w = foot_pos_w + quat_apply(
            foot_quat_w.reshape(-1, 4), local_offsets[None, :, :].expand(env.num_envs, -1, -1).reshape(-1, 3)
        ).reshape(env.num_envs, len(asset_cfg.body_ids), 3)

    previous_contact_name = "_kbot_swing_target_previous_contact"
    takeoff_x_name = "_kbot_swing_target_takeoff_x"
    has_takeoff_name = "_kbot_swing_target_has_takeoff"
    first_target_done_name = "_kbot_swing_target_first_target_done"
    previous_contact = getattr(env, previous_contact_name, None)
    takeoff_x = getattr(env, takeoff_x_name, None)
    has_takeoff = getattr(env, has_takeoff_name, None)
    first_target_done = getattr(env, first_target_done_name, None)
    if previous_contact is None or previous_contact.shape != in_contact.shape or previous_contact.device != in_contact.device:
        previous_contact = in_contact.clone()
    if takeoff_x is None or takeoff_x.shape != foot_pos_w[:, :, 0].shape or takeoff_x.device != foot_pos_w.device:
        takeoff_x = foot_pos_w[:, :, 0].clone()
    if has_takeoff is None or has_takeoff.shape != in_contact.shape or has_takeoff.device != in_contact.device:
        has_takeoff = torch.zeros_like(in_contact)
    if first_target_done is None or first_target_done.shape != env.episode_length_buf.shape or first_target_done.device != in_contact.device:
        first_target_done = torch.zeros(env.num_envs, dtype=torch.bool, device=in_contact.device)

    reset = (env.episode_length_buf <= 1).unsqueeze(1)
    previous_contact = torch.where(reset, in_contact, previous_contact)
    takeoff_x = torch.where(reset, foot_pos_w[:, :, 0], takeoff_x)
    has_takeoff = torch.where(reset, torch.zeros_like(has_takeoff), has_takeoff)
    first_target_done = torch.where(reset.squeeze(1), torch.zeros_like(first_target_done), first_target_done)

    takeoff = previous_contact & ~in_contact
    takeoff_x = torch.where(takeoff, foot_pos_w[:, :, 0], takeoff_x)
    has_takeoff = torch.where(takeoff, torch.ones_like(has_takeoff), has_takeoff)

    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is None:
        origin_y = torch.zeros(env.num_envs, dtype=foot_pos_w.dtype, device=foot_pos_w.device)
    else:
        origin_y = env_origins[:, 1].to(dtype=foot_pos_w.dtype, device=foot_pos_w.device)
    foot_x = foot_pos_w[:, :, 0]
    target_y = torch.stack((origin_y + target_left_y, origin_y + target_right_y), dim=1)
    first_length = target_length * min(max(first_target_fraction, 0.0), 1.0)
    target_x = takeoff_x + torch.where(
        first_target_done.unsqueeze(1),
        torch.full_like(takeoff_x, target_length),
        torch.full_like(takeoff_x, first_length),
    )
    x_error = (foot_x - target_x) / max(x_scale, 1.0e-6)
    y_error = (foot_pos_w[:, :, 1] - target_y) / max(y_scale, 1.0e-6)
    target_reward = torch.exp(-(torch.square(x_error) + torch.square(y_error)))

    stance_x = torch.stack((foot_x[:, 1], foot_x[:, 0]), dim=1)
    progress_span = target_x - stance_x
    linear_progress = torch.where(
        progress_span > 1.0e-6,
        torch.clamp((foot_x - stance_x) / torch.clamp(progress_span, min=1.0e-6), min=0.0, max=1.0),
        torch.zeros_like(progress_span),
    )

    state = _gait_cycle_phase_state(
        env, period_s, swing_phase_fraction, toe_off_debounce_s, plant_phase_fraction, sensor_cfg
    )
    progress_reward = linear_progress_scale * linear_progress
    smooth_lambda = max(smooth_max_lambda, 0.0)
    if smooth_lambda > 0.0:
        delta = target_reward - progress_reward
        abs_delta = torch.abs(delta)
        smooth = 0.5 * (target_reward + progress_reward) + torch.square(delta) / (4.0 * smooth_lambda) + (
            0.25 * smooth_lambda
        )
        per_foot_reward = torch.where(abs_delta < smooth_lambda, smooth, torch.maximum(target_reward, progress_reward))
    else:
        per_foot_reward = torch.maximum(target_reward, progress_reward)
    f1_reward, f2_reward = _select_f1_f2(per_foot_reward, state["f1_is_left"])
    f1_has_takeoff, f2_has_takeoff = _select_f1_f2(has_takeoff, state["f1_is_left"])
    f1_active = (state["f1_swing"] | state["f1_plant"]) & ~state["f1_contact"] & f1_has_takeoff
    f2_active = (state["f2_swing"] | state["f2_plant"]) & ~state["f2_contact"] & f2_has_takeoff
    reward = f1_active.float() * f1_reward + f2_active.float() * f2_reward
    touchdown = in_contact & has_takeoff
    first_target_done = first_target_done | torch.any(touchdown, dim=1)
    has_takeoff = torch.where(in_contact, torch.zeros_like(has_takeoff), has_takeoff)

    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    upright = _upright_health_gate(env, minimum_height, max_tilt, asset_cfg)
    setattr(env, previous_contact_name, in_contact)
    setattr(env, takeoff_x_name, takeoff_x)
    setattr(env, has_takeoff_name, has_takeoff)
    setattr(env, first_target_done_name, first_target_done)
    return reward * moving.float() * upright


def gait_cycle_plant_water_level_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    period_s: float,
    swing_phase_fraction: float,
    toe_off_debounce_s: float,
    plant_phase_fraction: float,
    water_level: float,
    minimum_water_level: float,
    z_epsilon: float,
    up_penalty: float,
    retreat_epsilon: float,
    retreat_scale: float,
    retreat_penalty: float,
    minimum_height: float,
    max_tilt: float,
    foot_local_offsets: list[tuple[float, float, float]],
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
    post_plant_up_penalty: float = 1.0,
    post_plant_lift_penalty: float = 1.0,
    outside_plant_touchdown_penalty: float = 1.0,
    extra_swing_takeoff_penalty: float = 1.0,
) -> torch.Tensor:
    """Reward scheduled plant lowering and penalize phase-event microsteps."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    offsets = torch.tensor(foot_local_offsets, dtype=foot_pos_w.dtype, device=foot_pos_w.device)
    sole_pos_w = foot_pos_w + quat_apply(foot_quat_w, offsets.unsqueeze(0).expand(env.num_envs, -1, -1))
    sole_height = sole_pos_w[:, :, 2]
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        sole_height = sole_height - env_origins[:, None, 2]
    sole_x = sole_pos_w[:, :, 0]

    state = _gait_cycle_phase_state(
        env, period_s, swing_phase_fraction, toe_off_debounce_s, plant_phase_fraction, sensor_cfg
    )
    f1_height, f2_height = _select_f1_f2(sole_height, state["f1_is_left"])
    f1_x, f2_x = _select_f1_f2(sole_x, state["f1_is_left"])
    f1_first_contact, f2_first_contact = _select_f1_f2(first_contact, state["f1_is_left"])
    f1_contact = state["f1_contact"]
    f2_contact = state["f2_contact"]
    reset = env.episode_length_buf <= 1

    water = torch.full_like(f1_height, max(water_level, minimum_water_level))
    floor = torch.full_like(f1_height, min(water_level, minimum_water_level))
    span = torch.clamp(water - floor, min=1.0e-6)
    f1_z_credit = torch.clamp(f1_height, min=floor, max=water)
    f2_z_credit = torch.clamp(f2_height, min=floor, max=water)

    min_z_f1_name = "_kbot_gait_cycle_plant_min_z_f1"
    min_z_f2_name = "_kbot_gait_cycle_plant_min_z_f2"
    prev_z_f1_name = "_kbot_gait_cycle_plant_prev_z_f1"
    prev_z_f2_name = "_kbot_gait_cycle_plant_prev_z_f2"
    peak_x_f1_name = "_kbot_gait_cycle_plant_peak_x_f1"
    peak_x_f2_name = "_kbot_gait_cycle_plant_peak_x_f2"
    plant_seen_f1_name = "_kbot_gait_cycle_plant_seen_f1"
    plant_seen_f2_name = "_kbot_gait_cycle_plant_seen_f2"
    post_up_seen_f1_name = "_kbot_gait_cycle_plant_post_up_seen_f1"
    post_up_seen_f2_name = "_kbot_gait_cycle_plant_post_up_seen_f2"
    post_lift_seen_f1_name = "_kbot_gait_cycle_plant_post_lift_seen_f1"
    post_lift_seen_f2_name = "_kbot_gait_cycle_plant_post_lift_seen_f2"
    prev_contact_f1_name = "_kbot_gait_cycle_plant_prev_contact_f1"
    prev_contact_f2_name = "_kbot_gait_cycle_plant_prev_contact_f2"
    swing_takeoff_seen_f1_name = "_kbot_gait_cycle_plant_swing_takeoff_seen_f1"
    swing_takeoff_seen_f2_name = "_kbot_gait_cycle_plant_swing_takeoff_seen_f2"
    min_z_f1 = getattr(env, min_z_f1_name, None)
    min_z_f2 = getattr(env, min_z_f2_name, None)
    prev_z_f1 = getattr(env, prev_z_f1_name, None)
    prev_z_f2 = getattr(env, prev_z_f2_name, None)
    peak_x_f1 = getattr(env, peak_x_f1_name, None)
    peak_x_f2 = getattr(env, peak_x_f2_name, None)
    plant_seen_f1 = getattr(env, plant_seen_f1_name, None)
    plant_seen_f2 = getattr(env, plant_seen_f2_name, None)
    post_up_seen_f1 = getattr(env, post_up_seen_f1_name, None)
    post_up_seen_f2 = getattr(env, post_up_seen_f2_name, None)
    post_lift_seen_f1 = getattr(env, post_lift_seen_f1_name, None)
    post_lift_seen_f2 = getattr(env, post_lift_seen_f2_name, None)
    prev_contact_f1 = getattr(env, prev_contact_f1_name, None)
    prev_contact_f2 = getattr(env, prev_contact_f2_name, None)
    swing_takeoff_seen_f1 = getattr(env, swing_takeoff_seen_f1_name, None)
    swing_takeoff_seen_f2 = getattr(env, swing_takeoff_seen_f2_name, None)
    if min_z_f1 is None or min_z_f1.shape != f1_height.shape or min_z_f1.device != f1_height.device:
        min_z_f1 = water.clone()
    if min_z_f2 is None or min_z_f2.shape != f2_height.shape or min_z_f2.device != f2_height.device:
        min_z_f2 = water.clone()
    if prev_z_f1 is None or prev_z_f1.shape != f1_height.shape or prev_z_f1.device != f1_height.device:
        prev_z_f1 = f1_z_credit.clone()
    if prev_z_f2 is None or prev_z_f2.shape != f2_height.shape or prev_z_f2.device != f2_height.device:
        prev_z_f2 = f2_z_credit.clone()
    if peak_x_f1 is None or peak_x_f1.shape != f1_x.shape or peak_x_f1.device != f1_x.device:
        peak_x_f1 = f1_x.clone()
    if peak_x_f2 is None or peak_x_f2.shape != f2_x.shape or peak_x_f2.device != f2_x.device:
        peak_x_f2 = f2_x.clone()
    if plant_seen_f1 is None or plant_seen_f1.shape != f1_contact.shape or plant_seen_f1.device != f1_contact.device:
        plant_seen_f1 = torch.zeros_like(f1_contact)
    if plant_seen_f2 is None or plant_seen_f2.shape != f2_contact.shape or plant_seen_f2.device != f2_contact.device:
        plant_seen_f2 = torch.zeros_like(f2_contact)
    if post_up_seen_f1 is None or post_up_seen_f1.shape != f1_contact.shape or post_up_seen_f1.device != f1_contact.device:
        post_up_seen_f1 = torch.zeros_like(f1_contact)
    if post_up_seen_f2 is None or post_up_seen_f2.shape != f2_contact.shape or post_up_seen_f2.device != f2_contact.device:
        post_up_seen_f2 = torch.zeros_like(f2_contact)
    if post_lift_seen_f1 is None or post_lift_seen_f1.shape != f1_contact.shape or post_lift_seen_f1.device != f1_contact.device:
        post_lift_seen_f1 = torch.zeros_like(f1_contact)
    if post_lift_seen_f2 is None or post_lift_seen_f2.shape != f2_contact.shape or post_lift_seen_f2.device != f2_contact.device:
        post_lift_seen_f2 = torch.zeros_like(f2_contact)
    if prev_contact_f1 is None or prev_contact_f1.shape != f1_contact.shape or prev_contact_f1.device != f1_contact.device:
        prev_contact_f1 = f1_contact.clone()
    if prev_contact_f2 is None or prev_contact_f2.shape != f2_contact.shape or prev_contact_f2.device != f2_contact.device:
        prev_contact_f2 = f2_contact.clone()
    if swing_takeoff_seen_f1 is None or swing_takeoff_seen_f1.shape != f1_contact.shape or swing_takeoff_seen_f1.device != f1_contact.device:
        swing_takeoff_seen_f1 = torch.zeros_like(f1_contact)
    if swing_takeoff_seen_f2 is None or swing_takeoff_seen_f2.shape != f2_contact.shape or swing_takeoff_seen_f2.device != f2_contact.device:
        swing_takeoff_seen_f2 = torch.zeros_like(f2_contact)

    f1_inactive_or_reset = reset | ~state["f1_plant"]
    f2_inactive_or_reset = reset | ~state["f2_plant"]
    min_z_f1 = torch.where(f1_inactive_or_reset, water, min_z_f1)
    min_z_f2 = torch.where(f2_inactive_or_reset, water, min_z_f2)
    prev_z_f1 = torch.where(reset | state["f1_swing"], f1_z_credit, prev_z_f1)
    prev_z_f2 = torch.where(reset | state["f2_swing"], f2_z_credit, prev_z_f2)
    peak_x_f1 = torch.where(f1_inactive_or_reset, f1_x, peak_x_f1)
    peak_x_f2 = torch.where(f2_inactive_or_reset, f2_x, peak_x_f2)
    prev_contact_f1 = torch.where(reset, f1_contact, prev_contact_f1)
    prev_contact_f2 = torch.where(reset, f2_contact, prev_contact_f2)
    plant_latch_reset_f1 = reset | state["f1_swing"]
    plant_latch_reset_f2 = reset | state["f2_swing"]
    plant_seen_f1 = torch.where(plant_latch_reset_f1, torch.zeros_like(plant_seen_f1), plant_seen_f1)
    plant_seen_f2 = torch.where(plant_latch_reset_f2, torch.zeros_like(plant_seen_f2), plant_seen_f2)
    post_up_seen_f1 = torch.where(plant_latch_reset_f1, torch.zeros_like(post_up_seen_f1), post_up_seen_f1)
    post_up_seen_f2 = torch.where(plant_latch_reset_f2, torch.zeros_like(post_up_seen_f2), post_up_seen_f2)
    post_lift_seen_f1 = torch.where(plant_latch_reset_f1, torch.zeros_like(post_lift_seen_f1), post_lift_seen_f1)
    post_lift_seen_f2 = torch.where(plant_latch_reset_f2, torch.zeros_like(post_lift_seen_f2), post_lift_seen_f2)
    swing_takeoff_seen_f1 = torch.where(
        reset | ~state["f1_swing"], torch.zeros_like(swing_takeoff_seen_f1), swing_takeoff_seen_f1
    )
    swing_takeoff_seen_f2 = torch.where(
        reset | ~state["f2_swing"], torch.zeros_like(swing_takeoff_seen_f2), swing_takeoff_seen_f2
    )

    f1_downward_credit = torch.clamp(min_z_f1 - f1_z_credit, min=0.0) / span
    f2_downward_credit = torch.clamp(min_z_f2 - f2_z_credit, min=0.0) / span
    f1_low_seen = min_z_f1 < (water - max(z_epsilon, 0.0))
    f2_low_seen = min_z_f2 < (water - max(z_epsilon, 0.0))
    f1_up_after_low = f1_low_seen & (f1_z_credit > (prev_z_f1 + max(z_epsilon, 0.0)))
    f2_up_after_low = f2_low_seen & (f2_z_credit > (prev_z_f2 + max(z_epsilon, 0.0)))
    f1_planted_now = state["f1_plant"] & (f1_contact | (f1_z_credit < (water - max(z_epsilon, 0.0))))
    f2_planted_now = state["f2_plant"] & (f2_contact | (f2_z_credit < (water - max(z_epsilon, 0.0))))
    plant_seen_f1 = plant_seen_f1 | f1_planted_now
    plant_seen_f2 = plant_seen_f2 | f2_planted_now

    updated_peak_x_f1 = torch.where(state["f1_plant"], torch.maximum(peak_x_f1, f1_x), f1_x)
    updated_peak_x_f2 = torch.where(state["f2_plant"], torch.maximum(peak_x_f2, f2_x), f2_x)
    f1_retreat_excess = torch.clamp(updated_peak_x_f1 - f1_x - max(retreat_epsilon, 0.0), min=0.0)
    f2_retreat_excess = torch.clamp(updated_peak_x_f2 - f2_x - max(retreat_epsilon, 0.0), min=0.0)
    f1_retreat_score = torch.clamp(f1_retreat_excess / max(retreat_scale, 1.0e-6), min=0.0, max=1.0)
    f2_retreat_score = torch.clamp(f2_retreat_excess / max(retreat_scale, 1.0e-6), min=0.0, max=1.0)

    f1_reward_active = state["f1_plant"] & ~state["f1_contact"]
    f2_reward_active = state["f2_plant"] & ~state["f2_contact"]
    f1_reward = (
        f1_downward_credit
        - up_penalty * f1_up_after_low.float()
        - retreat_penalty * f1_retreat_score
    ) * f1_reward_active.float()
    f2_reward = (
        f2_downward_credit
        - up_penalty * f2_up_after_low.float()
        - retreat_penalty * f2_retreat_score
    ) * f2_reward_active.float()

    f1_hold_active = state["assigned"] & plant_seen_f1 & ~state["f1_plant"] & ~state["f1_swing"]
    f2_hold_active = state["assigned"] & plant_seen_f2 & ~state["f2_plant"] & ~state["f2_swing"]
    f1_post_up = f1_hold_active & ~post_up_seen_f1 & (f1_z_credit > (prev_z_f1 + max(z_epsilon, 0.0)))
    f2_post_up = f2_hold_active & ~post_up_seen_f2 & (f2_z_credit > (prev_z_f2 + max(z_epsilon, 0.0)))
    f1_post_lift = f1_hold_active & ~post_lift_seen_f1 & ~f1_contact
    f2_post_lift = f2_hold_active & ~post_lift_seen_f2 & ~f2_contact
    post_up_seen_f1 = post_up_seen_f1 | f1_post_up
    post_up_seen_f2 = post_up_seen_f2 | f2_post_up
    post_lift_seen_f1 = post_lift_seen_f1 | f1_post_lift
    post_lift_seen_f2 = post_lift_seen_f2 | f2_post_lift

    f1_outside_plant_touchdown = state["assigned"] & ~reset & f1_first_contact & ~state["f1_plant"]
    f2_outside_plant_touchdown = state["assigned"] & ~reset & f2_first_contact & ~state["f2_plant"]
    f1_takeoff = state["assigned"] & ~reset & state["f1_swing"] & prev_contact_f1 & ~f1_contact
    f2_takeoff = state["assigned"] & ~reset & state["f2_swing"] & prev_contact_f2 & ~f2_contact
    f1_extra_takeoff = f1_takeoff & swing_takeoff_seen_f1
    f2_extra_takeoff = f2_takeoff & swing_takeoff_seen_f2
    swing_takeoff_seen_f1 = swing_takeoff_seen_f1 | f1_takeoff
    swing_takeoff_seen_f2 = swing_takeoff_seen_f2 | f2_takeoff

    event_penalty = (
        post_plant_up_penalty * (f1_post_up.float() + f2_post_up.float())
        + post_plant_lift_penalty * (f1_post_lift.float() + f2_post_lift.float())
        + outside_plant_touchdown_penalty
        * (f1_outside_plant_touchdown.float() + f2_outside_plant_touchdown.float())
        + extra_swing_takeoff_penalty * (f1_extra_takeoff.float() + f2_extra_takeoff.float())
    ) / max(env.step_dt, 1.0e-6)

    updated_min_z_f1 = torch.where(state["f1_plant"], torch.minimum(min_z_f1, f1_z_credit), water)
    updated_min_z_f2 = torch.where(state["f2_plant"], torch.minimum(min_z_f2, f2_z_credit), water)
    updated_prev_z_f1 = torch.where(state["f1_plant"], f1_z_credit, f1_z_credit)
    updated_prev_z_f2 = torch.where(state["f2_plant"], f2_z_credit, f2_z_credit)
    setattr(env, min_z_f1_name, updated_min_z_f1)
    setattr(env, min_z_f2_name, updated_min_z_f2)
    setattr(env, prev_z_f1_name, updated_prev_z_f1)
    setattr(env, prev_z_f2_name, updated_prev_z_f2)
    setattr(env, peak_x_f1_name, updated_peak_x_f1)
    setattr(env, peak_x_f2_name, updated_peak_x_f2)
    setattr(env, plant_seen_f1_name, plant_seen_f1)
    setattr(env, plant_seen_f2_name, plant_seen_f2)
    setattr(env, post_up_seen_f1_name, post_up_seen_f1)
    setattr(env, post_up_seen_f2_name, post_up_seen_f2)
    setattr(env, post_lift_seen_f1_name, post_lift_seen_f1)
    setattr(env, post_lift_seen_f2_name, post_lift_seen_f2)
    setattr(env, prev_contact_f1_name, f1_contact)
    setattr(env, prev_contact_f2_name, f2_contact)
    setattr(env, swing_takeoff_seen_f1_name, swing_takeoff_seen_f1)
    setattr(env, swing_takeoff_seen_f2_name, swing_takeoff_seen_f2)

    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    upright = _upright_health_gate(env, minimum_height, max_tilt, asset_cfg)
    return (f1_reward + f2_reward - event_penalty) * moving.float() * upright


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
    sensor_cfg: SceneEntityCfg | None = None,
    ema_cycle_count: float = 0.0,
    cycle_duration_smoothing_cycles: float = 5.0,
    min_cycle_duration_s: float = 0.25,
    max_cycle_duration_s: float = 2.0,
    min_tau_s: float = 0.75,
    max_tau_s: float = 10.0,
) -> torch.Tensor:
    """Penalize selected joints holding a persistent offset from neutral."""
    asset = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    buffer_name = f"_kbot_joint_position_ema_{asset_cfg.name}_{len(asset_cfg.joint_ids)}"
    ema = getattr(env, buffer_name, None)
    if ema is None or ema.shape != joint_error.shape or ema.device != joint_error.device:
        ema = torch.zeros_like(joint_error)

    alpha = _adaptive_ema_alpha(
        env,
        tau_s,
        joint_error,
        sensor_cfg,
        ema_cycle_count,
        cycle_duration_smoothing_cycles,
        min_cycle_duration_s,
        max_cycle_duration_s,
        min_tau_s,
        max_tau_s,
    ).unsqueeze(1)
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
