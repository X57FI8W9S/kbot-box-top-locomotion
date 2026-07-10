# Box Top V4

Date: 2026-06-19

V4 is the next box-top policy branch. Its first purpose is not a new reward
idea by itself. Its purpose is to move training onto the corrected Top4 robot
asset and start from the new GUI-settled standing pose instead of the older
Top3 authored pose.

## Status

V4 preview task is registered in code:

```text
Isaac-KBot-Forward-Flat-V4-Top4Starter-v0
```

The preview task uses the clean Top4 USD, the settled starter pose, the scaled
implicit actuator groups that matched the V3 posed-start lineage, and the
conservative V4 gait-quality reward table. It also sets offset-based sole
rewards to use the verified symmetric Top4 sole center offsets.

## 2026-06-22 V4 Fall Exploit And May 31 Reference

Current V4 `model_300` is a fall exploit, not a walking keeper. The video shows
the policy leaning/falling forward to collect forward velocity reward.

Failed V4 run:

```text
logs/rsl_rl/kbot_forward_flat/2026-06-22_12-28-34_L3.1/model_300.pt
```

Artifacts:

```text
videos/play/trailing-hud-model_300-v4-top4-soletrack-30s.mp4
diagnostics/model_300_v4_top4_soletrack/summary.md
diagnostics/model_300_v4_top4_soletrack/diagnostic_graph.png
graficos/2026-06-22_12-28-34_combined_reward_components.png
```

Summary:

```text
training row 300:
  component_sum = 5.80
  track_lin_vel_xy_exp = 3.87
  world_forward_velocity_clip = 6.08
  termination_penalty = -3.75
  upright_alive = 2.13
  flat_orientation_l2 = -0.60
  bad_orientation termination = 1.0 in the final training iterations

rollout / diagnostics:
  mean speed = 0.422 m/s against command 0.375 m/s
  final x = 0.260-0.267 m over 30 s
  approved step fraction = 0.206 diagnostic, 0.30 HUD rollout
  double support fraction = 0.941
  mean swing sole clearance = about 1.0-1.4 mm
  decision = REJECT
```

The useful May 31 reference checkpoint is:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-31_21-14-35_I8.1.3/model_300.pt
```

Reference artifacts:

```text
videos/play/trailing-hud-model_300-v3-clearance-overtake030-w100-30s.mp4
graficos/2026-05-31_21-14-35_combined_reward_components.png
metricas/2026-05-31_21-14-35_reward_components.csv
```

Lineage:

```text
2026-05-31_03-42-10_I8
  -> model_99.pt

2026-05-31_03-56-23_I8.1
  resumed from 03:42 model_99.pt
  -> model_200.pt

2026-05-31_21-14-35_I8.1.3
  resumed from 03:56 model_200.pt
  changed foot_sagittal_separation_l1 weight -2 -> -4
  changed swing_foot_overtake_l1 weight -3 -> -100
  set swing_foot_overtake_l1 target_length = 0.30 m
  -> model_300.pt
```

May 31 `model_300` was not final-quality walking because it drifted badly in
world `y`, but it is the known useful branching point because it advanced
without the V4 fall exploit:

```text
mean speed = 0.303 m/s against command 0.75 m/s
final x = 5.05 m
final y = -5.96 m
fall_reset_count = 0
final HUD fsep = 0.288 m
final HUD ksep = 0.317 m
approved step fraction = 0.20
J/m = 2535.8

training row 300:
  component_sum = 7.51
  world_forward_velocity_clip = 9.44
  termination_penalty = -0.107
  upright_alive = 7.94
  flat_orientation_l2 = -0.086
```

Main differences between the May 31 reference and current V4:

```text
asset / pose:
  May 31: Top3 USD and old posed-start reset
  V4: corrected Top4 USD and settled Top4 starter pose

command / gait timing:
  May 31: command speed 0.75 m/s, alternating_foot_phase period 1.0 s
  V4: command speed 0.375 m/s, gait_phase period 1.6 s

forward reward:
  May 31: upright-gated track_lin_vel_xy_exp and upright-gated
          world_forward_velocity_clip
  V4: centerline+heading+upright gated track/clip, same nominal track weight
      but lower max_velocity

step timing:
  May 31: feet_air_time weight 1.0, alternating_foot_phase weight 0.18
  V4: feet_air_time 0, alternating_foot_phase removed, gait_cycle_support 1.0,
      gait_cycle_plant_water_level 1.0

swing shaping:
  May 31: swing_foot_overtake_l1 -100 with target_length 0.30 m,
          foot_sagittal_separation_l1 -4, swing_sole_clearance 1.0
  V4: swing_foot_overtake_l1 -80, foot_sagittal_separation_l1 -4,
      dense_foot_swing_speed 20, dense_swing_foot_target_location_exp 40,
      foot_retreat -6, swing_sole_clearance 1.0

tracks:
  May 31: Top3 lanes around +/-0.1582 m, width target 0.3164 m
  V4: Top4 corrected sole lanes +/-0.1427 m, width target 0.2854 m
```

Recommended V4 rebalance, before launching another scratch run:

```text
Goal:
  make V4 closer to the May 31 learning pressure while keeping the corrected
  Top4 asset, Top4 sole-center tracks, and current gait-cycle implementation
  available.

Do not continue the current V4 model_300 as a walking parent.

Suggested conservative V4 weights:

forward_heading_tracking:
  track_lin_vel_xy_exp: keep 30 if using the old upright-only function,
                        or lower to 15 if keeping centerline+heading gating
  world_forward_velocity_clip: 30 -> 10 or 15
  world_forward_velocity_below_l2: keep -24
  forward_velocity_below_l2: keep -6
  yaw_rate_l2: keep -20
  world_heading_l2: keep -90

gait_step_timing:
  feet_air_time: 0 -> 1.0
  alternating_foot_phase: removed
  gait_cycle_support: keep active without the legacy fixed-clock phase reward
  gait_cycle_plant_water_level: 1.0 -> 0.25
  dense_foot_swing_speed: 20 -> 0-5
  dense_swing_foot_target_location_exp: 40 -> 0-5
  foot_retreat: -6 -> 0 to -1
  swing_foot_overtake_l1: -80 -> -100
  swing_foot_overtake_l1.target_length: use 0.30 m
  foot_sagittal_separation_l1: keep -4
  swing_sole_clearance: keep 1.0

lateral_centerline_width:
  keep the corrected Top4 sole lanes:
    target_left_y = +0.1427 m
    target_right_y = -0.1427 m
    target_width = 0.2854 m
  keep foot_sole_lateral_lane_max_l1 around -44 at first
  do not raise lane/centerline weights until the robot is walking without
  falling

posture_survival:
  keep May 31-like posture terms for the compatibility run, but if the fall
  exploit repeats, increase survival before increasing gait shaping:
    termination_penalty: -750 -> -1500 or -3000
    upright_alive: 8 -> 10 or 12
    flat_orientation_l2: -20 -> -30 or -40
```

The key change is to make the new dense target/swing/cyclic terms small until
the Top4 robot first recovers the May 31 behavior: stay upright, advance, and
avoid the forward-fall velocity shortcut. Once that is true, increase the
current V4 cyclic/target terms gradually and use the term-contribution graph to
decide what old terms can be removed.

## 2026-06-23 V4 Conservative Rebalance Implementation

Implemented in the first V4 task so there is only one active V4 training
target:

```text
Isaac-KBot-Forward-Flat-V4-Top4Starter-v0
```

Code config:

```text
KBotForwardFlatV4Top4StarterEnvCfg
```

The V4 rebalance is implemented as a local grouped weight table in that
subclass, using the same theme names as the parent V3 reward block. Parameter
choices are kept in an adjacent `reward_params` table, like V3.

Concrete V4 effective weights:

```text
action_joint_regularization:
  action_rate_l2 = -0.09
  centered_joint_target_position_l2 = 0.0
  dof_acc_l2 = -1.0e-7
  dof_pos_limits = -2.0
  dof_torques_l2 = -5.0e-5
  hip_roll_position_ema_5cycle_l2 = 0.0
  hip_roll_yaw_position_ema_l2 = 0.0
  hip_roll_yaw_position_l2 = 0.0
  knee_extension_l1 = 0.0
  signed_joint_pair_ema_symmetry_l2 = -3.0
  stand_joint_position_l2 = -0.5
  wobble_joint_vel_l2 = -0.04

forward_heading_tracking:
  backward_velocity_l2 = -2.0
  forward_velocity_below_l2 = -6.0
  track_ang_vel_z_exp = 1.0
  track_lin_vel_xy_exp = 30.0
  world_forward_velocity_below_l2 = -24.0
  world_forward_velocity_clip = 15.0
  world_forward_velocity_clip.max_velocity = 0.375
  world_heading_l2 = -90.0
  yaw_rate_l2 = -20.0

gait_step_timing:
  alternating_foot_phase = removed
  alternating_step_symmetry_l2 = -0.2
  contact_duty_symmetry_l2 = -2.0
  dense_foot_swing_speed = 5.0
  dense_swing_foot_target_location_exp = 5.0
  dense_swing_step_length = 0.0
  feet_air_time = 1.0
  feet_air_time.threshold = 0.22
  foot_retreat = -1.0
  foot_sagittal_separation_l1 = -4.0
  gait_cycle_plant_water_level = 0.25
  gait_cycle_support = 0.25
  swing_foot_overtake_l1 = -100.0
  swing_foot_overtake_l1.target_length = 0.30
  swing_sole_clearance = 1.0

lateral_centerline_width:
  foot_lateral_lane_l1 = -4.0
  foot_lateral_lane_max_l1 = -2.0
  foot_lateral_spacing_l1 = -9.0
  foot_signed_lateral_clearance_l1 = -12.0
  foot_sole_lateral_lane_max_l1 = -44.0
  lateral_away_from_center_l2 = 0.0
  lateral_velocity_l2 = -20.0
  root_lateral_position_l2 = -12.0

leg_frontal_plane:
  left_leg_frontal_plane_l1 = -4.0
  leg_frontal_plane_l1 = 0.0
  leg_frontal_sole_plane_max_l1 = -14.0
  max_leg_frontal_plane_l1 = -10.0
  right_leg_frontal_plane_l1 = -4.0

posture_survival:
  alive = 1.0
  ang_vel_xy_l2 = -0.25
  base_height_l2 = -35.0
  flat_orientation_l2 = -20.0
  lin_vel_z_l2 = -2.0
  low_body_l2 = -120.0
  root_lateral_tilt_ema_l2 = -120.0
  root_lateral_tilt_l2 = -50.0
  termination_penalty = -750.0
  undesired_contacts = -2.0
  upright_alive = 8.0

sole_foot_orientation:
  foot_flat_l2 = 0.0
  foot_parallel_l2 = 0.0
  foot_toe_in_l2 = 0.0
  foot_world_parallel_l2 = 0.0
  foot_world_parallel_max_l2 = -0.8
  stance_foot_flat_l2 = -1.2
```

Notes:

```text
1. V4 keeps the corrected Top4 USD, Top4 starter pose, scaled implicit
   actuator groups, and corrected Top4 sole lanes.
2. V4 keeps the V3 cyclic/centerline gait machinery and uses command speed,
   velocity clip max, and gait timing tied to 0.375 m/s.
3. V4 restores May 31-style feet_air_time pressure and removes
   alternating_foot_phase.
4. V4 keeps the current cyclic/dense rewards active, but small enough that
   they should not dominate before the robot has recovered non-falling forward
   locomotion.
```

## V4 Video And Evaluation Guide

V4 videos should be made as checkpoint-matched inspection artifacts. Do not
use a video render as an implicit continuation request, and do not compare
V4 `model_100` to the May 31 `model_300` when judging whether the V4
restart is on track. Compare checkpoint `n` to the corresponding checkpoint in
the May 31 lineage when that checkpoint exists:

```text
May 31 iteration 99/100 reference:
  logs/rsl_rl/kbot_forward_flat/2026-05-31_03-42-10_I8/model_99.pt

May 31 iteration 200 reference:
  logs/rsl_rl/kbot_forward_flat/2026-05-31_03-56-23_I8.1/model_200.pt

May 31 iteration 300 reference:
  logs/rsl_rl/kbot_forward_flat/2026-05-31_21-14-35_I8.1.3/model_300.pt
```

The latest accepted historical V4 checkpoint/video set is the zero-to-100
rebalance run:

```text
logs/rsl_rl/kbot_forward_flat/2026-06-22_14-48-50_L5/model_100.pt

videos/play/trailing-hud-model_100-v4_0-top4-may31-rebalance-30s.mp4
videos/play/trailing-hud-model_100-v4_0-top4-may31-rebalance-30s.json
graficos/2026-06-22_14-48-50_combined_reward_components.png
metricas/2026-06-22_14-48-50_reward_components.csv
```

The interrupted V4 `100 -> 300` continuation is not an accepted baseline
comparison artifact. It was launched while we were trying to compare the
iteration-100 lineage result, so do not use that run to decide whether V4
matched the May 31 iteration-100 behavior.

Training command pattern for an iteration-100 V4 run:

```bash
.venv/bin/python scripts/rsl_rl/train.py \
  --task Isaac-KBot-Forward-Flat-V4-Top4Starter-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 101 \
  --save_interval 25 \
  --run_name v4_top4_conservative_4096envs_zero_to_100_save25_YYYYMMDD
```

`--max_iterations 101` is intentional. RSL-RL saves by loop index, so this is
how to materialize `model_100.pt` on the first try.

Graph command pattern:

```bash
.venv/bin/python scripts/diagnostics/plot_reward_components.py \
  --run-dir logs/rsl_rl/kbot_forward_flat/<run_dir>
```

Video command pattern:

```bash
.venv/bin/python scripts/rsl_rl/play_trailing.py \
  --task Isaac-KBot-Forward-Flat-V4-Top4Starter-v0 \
  --checkpoint logs/rsl_rl/kbot_forward_flat/<run_dir>/model_<n>.pt \
  --headless \
  --video_length 1500 \
  --exact_reset \
  --prime_default_targets \
  --fall_reset_height -1000 \
  --output logs/rsl_rl/kbot_forward_flat/<run_dir>/videos/play/trailing-hud-model_<n>-v4-<label>-30s.mp4 \
  --metrics_output logs/rsl_rl/kbot_forward_flat/<run_dir>/videos/play/trailing-hud-model_<n>-v4-<label>-30s.json
```

Use the full checkpoint path, not just `--load_run` plus `--checkpoint
model_<n>.pt`, for this workflow. The output should be 30 seconds at 50 FPS:
1500 frames, 1280x720, with the trailing view on the left, the side view on
the right, and one shared HUD band.

OpenCV verification:

```bash
.venv/bin/python -c "import cv2, sys; p=sys.argv[1]; cap=cv2.VideoCapture(p); print('opened', cap.isOpened()); print('frames', int(cap.get(cv2.CAP_PROP_FRAME_COUNT))); print('fps', cap.get(cv2.CAP_PROP_FPS)); print('width', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))); print('height', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))); cap.release()" \
  logs/rsl_rl/kbot_forward_flat/<run_dir>/videos/play/trailing-hud-model_<n>-v4-<label>-30s.mp4
```

Accepted videos should have a matching metrics JSON. A render without the JSON
can still be useful visually, but it is not the artifact to use for
lineage/baseline comparison.

For HUD layout and overlay maintenance rules, keep using the shared guide:

```text
policies/box_top_v2/POLICY_VIDEO_REPORT.md
```

## V4 Reward Editing Rules

Yes, the reward editing rules affect V4. V4 inherits the V3/V3.1 reward
surface and the same `RewTerm` machinery, so the grouped editing discipline
still matters. The important distinction is where the edit belongs:

```text
Parent V3 reward surface:
  KBotForwardFlatV3HandTuned648EnvCfg

Top4 asset / starter / corrected sole-lane settings:
  KBotForwardFlatV4Top4StarterEnvCfg
```

Rules for V4 work:

```text
1. If the change should affect all V3-derived experiments, edit the parent V3
   grouped reward block.
2. If the change is specific to the corrected Top4 asset, starter pose,
   Top4 sole centers, or V4 rebalance, keep it in the V4 subclass.
3. For a V4 weight retune, edit the local grouped weight table and mirror
   the full effective block in this README.
4. For a reward function, params change, or new reward term, follow the parent
   V3 pattern: wire the function/params/new `RewTerm` explicitly in the local
   V4 tables.
5. Do not edit the parent V3 values just to get V4 behavior unless the intent
   is to change every descendant of that parent.
```

Current V4 follows this by using a local grouped weight table for the May 31
compatibility weights. The Top4 geometry and sole-lane edits still live in the
V4 subclass because they are asset-specific, not generic V3 reward changes.

## Gait-Cycle Plant Guard

Implemented inside the existing `gait_cycle_plant_water_level` reward after the
2026-06-23 model-800 microstep exploit, where `gait_cycle_support` dominated the
graph while the HUD reported zero approved steps and 0.02-0.05 s contact
windows.

The current `gait_cycle_plant_water_level` term only rewards/penalizes sole
height during the scheduled plant window. The guard extends the plant state so a
foot that has planted low must stay down until its next scheduled swing window.

Guard state:

```text
plant_min_z[foot]
plant_seen[foot]
early_rise_penalty_seen[foot]
previous_contact[foot]
swing_takeoff_seen[foot]
```

Guard behavior:

```text
During scheduled plant:
  track the lowest sole height reached
  mark plant_seen when the sole reaches the configured water level

After scheduled plant and before that foot's next scheduled swing:
  if plant_seen and the sole rises upward by more than z_epsilon:
    emit one large pulse penalty once
  if plant_seen and contact is lost before the next scheduled swing:
    emit one large pulse penalty once

During scheduled swing:
  allow the first takeoff
  penalize every later takeoff event in the same scheduled swing

On the next scheduled swing window for that foot:
  reset the plant_seen and pulse-latch state
```

Add an explicit outside-plant touchdown guard using the contact sensor:

```text
first_contact = contact_sensor.compute_first_contact(env.step_dt)

if foot touchdown occurs outside its scheduled plant window:
  emit one large pulse penalty for that touchdown event
```

This is an event-level penalty, not a framewise penalty. The intent is
to make early/out-of-phase planting and repeated tap contacts expensive without
turning the reward into another dense contact-state game. The params stay local
to `gait_cycle_plant_water_level`:

```text
extra_swing_takeoff_penalty
outside_plant_touchdown_penalty
post_plant_lift_penalty
post_plant_up_penalty
z_epsilon
```

## Gait-Cycle Support Candidate: Phase-Latched Validity

2026-06-27 fixed-sweep status: this is wired into the active V4
`gait_cycle_support_reward` path, without changing the reward name. The fixed
500/800/1200/2500/5000/10000 sweep kept `toe_off_debounce_s = 0.04` and
`precycle_single_support_reward = 0.0`,
`precycle_double_support_reward = 0.0`.

Post-sweep optimization note: the harsher precycle penalty test
(`precycle_single_support_reward = -1.0`,
`precycle_double_support_reward = -0.25`) was too abrupt from the V3.2 seed and
produced poor rollout metrics. The first cadence follow-up at weight `-0.2`
reduced high-cadence stepping but also killed useful forward progression. The
active follow-up test restores precycle `0.0/0.0` and wires the same-foot
cadence ceiling below at `-0.05`.

Problem observed in the 2026-06-25 plantguard model-800 run:

```text
gait_cycle_support stayed high because it scores instantaneous contact mode.
The policy could keep the sampled contact mask correct most of the time while
using tiny high-frequency touchdown events between otherwise correct samples.
```

The non-redundant fix is to change the payout rule inside
`gait_cycle_support`, not to add another clearance, speed, or target-location
reward. Existing rewards already shape those pieces:

```text
swing_sole_clearance        -> clearance / drag / over-kick
dense_foot_swing_speed      -> forward swing velocity in lane/height window
dense_swing_target_location -> target step x/y progress
foot_retreat                -> no backward touchdown after swing
plant_water_level           -> plant lowering and contact-event pulses
```

This candidate makes `gait_cycle_support` a stricter accounting term:

```text
Swing and shift phases:
  one wrong sampled contact mode latches the phase reward to -1
  until the next phase starts.

Plant phases:
  do not latch one mistake across the whole plant phase.
  correct single-support credit fades from +1 at plant start to 0 at plant end.
  double support is neutral during plant because landing is expected there.
```

Candidate sketch:

```python
def gait_cycle_support_reward_phase_latched_candidate(
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
    state = _gait_cycle_phase_state(
        env, period_s, swing_phase_fraction, toe_off_debounce_s, plant_phase_fraction, sensor_cfg
    )
    assigned = state["assigned"]
    phase = state["phase"]
    in_contact = state["in_contact"]
    f1_contact = state["f1_contact"]
    f2_contact = state["f2_contact"]

    double_support = f1_contact & f2_contact
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
    f1_swing_nominal = torch.where(f2_single, ones, -ones)
    f2_swing_nominal = torch.where(f1_single, ones, -ones)
    f1_shift_nominal = torch.where(double_support, ones, -ones)
    f2_shift_nominal = torch.where(double_support, ones, -ones)

    # Plant is a transition: single-support credit fades out, double support is OK.
    f1_plant_reward = torch.where(
        f2_single,
        f1_plant_single_credit,
        torch.where(double_support, zeros, -ones),
    )
    f2_plant_reward = torch.where(
        f1_single,
        f2_plant_single_credit,
        torch.where(double_support, zeros, -ones),
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
    swing_shift_reward = torch.where(latch_active & bad_phase_seen, -ones, swing_shift_reward)

    plant_reward = (
        state["f1_plant"].float() * f1_plant_reward
        + state["f2_plant"].float() * f2_plant_reward
    )
    reward = swing_shift_reward + plant_reward

    setattr(env, previous_phase_name, phase_id)
    setattr(env, bad_phase_name, bad_phase_seen)
    return torch.where(assigned, reward, precycle_reward)
```

Expected effect:

```text
The policy can no longer recover a high average support score by being correct
for most samples and briefly wrong during contact chatter. A wrong sampled mode
spoils the current swing/shift phase. Plant remains a landing transition, so
late plant double support is not punished as a full support failure.
```

Post-sweep optimization candidates if microstepping survives after longer
training:

```text
Keep candidate runs named separately from the fixed same-code/weight sweep.
```

1. Existing same-foot cadence ceiling. Active follow-up test uses weight
`-0.05`; the older candidate values `-0.2` and `-1.0` are too harsh for a
V3.2-seeded run.

```python
self.rewards.walking_cycle_cadence_above_l2 = RewTerm(
    func=mdp.walking_cycle_cadence_above_l2,
    weight=-0.05,
    params={
        "max_cycle_hz": target_step_hz,
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
    },
)
```

Math:

```text
duration = now - last_touchdown_time_for_same_foot
cycle_hz = 1 / duration
penalty = sum(max(cycle_hz - max_cycle_hz, 0)^2)
```

For current V4, `target_root_speed_mps = 0.375` and
`target_step_length_m = 0.60`, so the target same-foot cycle rate is
`0.625 Hz`; using `target_step_hz = 1.25 Hz` as the ceiling is already a
tolerant 2x cap.

2. Existing contact-chatter penalty

```python
self.rewards.contact_chatter_l1 = RewTerm(
    func=mdp.contact_chatter_l1,
    weight=-6.0,
    params={
        "min_air_time": 0.10,
        "command_name": "base_velocity",
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot1", "foot3"]),
    },
)
```

Math:

```text
penalty = sum(max(min_air_time - last_air_time, 0) * first_contact)
```

This is a direct event-level penalty for touchdown after a too-short air phase.
It is safer than a broad gait rewrite, but it can suppress stepping if the
weight or `min_air_time` is too abrupt from the parent policy.

3. Proportional short-cycle penalty using the last 5 cycles or EMA

Candidate shape:

```text
cycle_duration_ema = EMA(last same-foot touchdown duration, alpha = 1 / 5)
shortfall = max(target_cycle_period_s - cycle_duration_ema, 0)
penalty = (shortfall / target_cycle_period_s)^2
```

Equivalent last-5 window:

```text
cycle_duration_avg5 = mean(last 5 same-foot touchdown durations)
shortfall = max(target_cycle_period_s - cycle_duration_avg5, 0)
penalty = (shortfall / target_cycle_period_s)^2
```

Important issue: the existing shared `_full_cycle_duration_ema` helper clamps
measured durations with `min_cycle_duration_s = 0.25` by default. That is useful
for stabilizing long-horizon EMA rewards, but it hides very fast microcycles
below 0.25 s. If this candidate is tested, use raw touchdown duration or lower
that clamp for this penalty only. EMA/window lag also means this is slower than
`contact_chatter_l1` at punishing a single bad touchdown.

## Observation Candidate: Proprioceptive Sole Position

Candidate only. Do not add this to the fixed-code sweep unless it is tested as a
separate observation branch.

Problem:

```text
The current V4 policy observes the phase clock and joint/base state, but it is
not explicitly told where each sole center is relative to the robot body. The
target-location and sole-clearance rewards compute that geometry internally, so
the policy has to infer sole position from joint state and dynamics.
```

Preferred candidate if the goal is to stay close to a proprioceptive walker:

```text
Add six observations:
  left sole center in root/body frame:  x_b, y_b, z_b
  right sole center in root/body frame: x_b, y_b, z_b

Current actor observation size: 44
Candidate actor observation size: 50
```

This is still derived proprioception if the same values can be computed from
joint encoders and the robot kinematic model. It is not an external foothold
target and it does not hand the reward answer to the policy.

Candidate `mdp.py` sketch:

```python
def sole_position_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
    foot_local_offsets: list[tuple[float, float, float]] | None = None,
) -> torch.Tensor:
    """Return left/right sole-center positions in root/body frame.

    Output shape is (num_envs, 6):
      left_x_b, left_y_b, left_z_b, right_x_b, right_y_b, right_z_b
    """
    asset = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]

    if foot_local_offsets is not None:
        foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
        local_offsets = torch.tensor(foot_local_offsets, dtype=foot_pos_w.dtype, device=foot_pos_w.device)
        foot_pos_w = foot_pos_w + quat_apply(
            foot_quat_w.reshape(-1, 4),
            local_offsets[None, :, :].expand(env.num_envs, -1, -1).reshape(-1, 3),
        ).reshape(env.num_envs, len(asset_cfg.body_ids), 3)

    root_pos_w = asset.data.root_pos_w[:, None, :]
    root_quat_w = asset.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    sole_pos_b = quat_apply_inverse(
        root_quat_w.reshape(-1, 4),
        (foot_pos_w - root_pos_w).reshape(-1, 3),
    ).reshape(env.num_envs, len(asset_cfg.body_ids), 3)

    return sole_pos_b.reshape(env.num_envs, -1)
```

Candidate V4 wiring sketch:

```python
self.observations.policy.sole_position_b = ObsTerm(
    func=mdp.sole_position_b,
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=["foot1", "foot3"]),
        "foot_local_offsets": sole_center_offsets,
    },
)
```

Optional normalization, only if the unnormalized values are hard for PPO with
`actor_obs_normalization=False`:

```python
scale = torch.tensor([0.60, 0.30, 0.90], dtype=sole_pos_b.dtype, device=sole_pos_b.device)
return (sole_pos_b / scale[None, None, :]).reshape(env.num_envs, -1)
```

Do not use absolute world foot position as the first test:

```text
world foot x/y leaks global translation and makes the policy learn an arbitrary
environment coordinate. Root/body-frame sole position preserves translation
invariance.
```

Task-reference alternative, not purely proprioceptive:

```text
Add target-relative signed error:
  left_dx, left_dy, right_dx, right_dy

This should be treated as a different branch. It tells the policy where the
reward target is, so it is closer to a clocked/task-reference controller than a
plain proprioceptive walker.
```

If the target-error branch is tested, prefer signed normalized errors over a
radius:

```text
dx = (foot_x_w - target_x_w) / x_scale
dy = (foot_y_w - target_y_w) / y_scale
```

Do not use only `r = sqrt(dx^2 + dy^2)` as the first test. Radius says how wrong
the foot is, but not whether it is too short, too long, inside, or outside.

## 2026-06-19 Implicit Actuator Preview Run

Training run:

```text
logs/rsl_rl/kbot_forward_flat/2026-06-19_15-19-33_L2
```

Training command:

```bash
.venv/bin/python scripts/rsl_rl/train.py \
  --task Isaac-KBot-Forward-Flat-V4-Top4Starter-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 101 \
  --save_interval 100 \
  --run_name v4_top4_starter_implicit_4096envs_zero_to_100_20260619
```

`--max_iterations 101` is intentional: RSL-RL saves checkpoints by loop index,
so this materializes `model_100.pt`.

Checkpoints:

```text
model_0.pt
model_100.pt
```

Videos:

```text
videos/play/trailing-hud-model_0-v4-top4-starter-implicit-30s.mp4
videos/play/trailing-hud-model_100-v4-top4-starter-implicit-30s.mp4
```

Both videos were rendered with:

```text
--exact_reset --prime_default_targets --fall_reset_height -1000 --video_length 1500
```

OpenCV verification:

```text
model_0:   1500 frames, 50 FPS, 30.0 s, 1280x720
model_100: 1500 frames, 50 FPS, 30.0 s, 1280x720
```

Metrics summary:

```text
model_0:
  fall_reset_count = 0
  final_x = -0.013 m
  final_y = -0.011 m
  mean_speed =  0.0027 m/s
  mean_root_height = 0.852 m
  final_hud_fsep = 0.245 m

model_100:
  fall_reset_count = 0
  final_x =  0.080 m
  final_y = -0.018 m
  mean_speed =  0.0048 m/s
  mean_root_height = 0.854 m
  final_hud_fsep = 0.247 m
```

Result: the implicit actuator rerun fixes the immediate fall/backslide seen in
the explicit comparison. It stands near the intended height, but it is not yet
walking.

## 2026-06-19 Explicit Actuator Comparison Run

This was the first V4 preview with the normal unscaled explicit actuator groups.
It is kept as a failed comparison because it fell/backslid in the diagnostic
videos.

Training run:

```text
logs/rsl_rl/kbot_forward_flat/2026-06-19_14-52-40_L1
```

Training command:

```bash
.venv/bin/python scripts/rsl_rl/train.py \
  --task Isaac-KBot-Forward-Flat-V4-Top4Starter-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 101 \
  --save_interval 100 \
  --run_name v4_top4_starter_explicit_4096envs_zero_to_100_20260619
```

`--max_iterations 101` is intentional: RSL-RL saves checkpoints by loop index,
so this materializes `model_100.pt`.

Checkpoints:

```text
model_0.pt
model_100.pt
```

Videos:

```text
videos/play/trailing-hud-model_0-v4-top4-starter-30s.mp4
videos/play/trailing-hud-model_100-v4-top4-starter-30s.mp4
```

Both videos were rendered with:

```text
--exact_reset --prime_default_targets --fall_reset_height -1000 --video_length 1500
```

OpenCV verification:

```text
model_0:   1500 frames, 50 FPS, 30.0 s, 1280x720
model_100: 1500 frames, 50 FPS, 30.0 s, 1280x720
```

Metrics summary:

```text
model_0:
  final_x = -0.430 m
  final_y =  0.085 m
  mean_speed = -0.328 m/s
  mean_root_height = 0.804 m
  final_hud_fsep = 0.286 m

model_100:
  final_x = -0.057 m
  final_y =  0.005 m
  mean_speed = -0.322 m/s
  mean_root_height = 0.810 m
  final_hud_fsep = 0.283 m
```

Result: this explicit-drive comparison is not a useful starting point. It
changed the asset, starter pose, and actuator model at the same time; the
implicit rerun isolates the actuator difference as the likely cause of the
immediate fall mode.

## Asset Inputs

Clean mirrored robot asset:

```text
assets/robot/usd/kbot_box_top4.usd
```

Pose/drive capture saved from Isaac Physics Inspector:

```text
assets/robot/usd/kbot_box_top4_test2.usd
```

`kbot_box_top4_test2.usd` should be treated as the evidence file for the
starter drive targets and settled joint state. For training, prefer wiring the
clean `kbot_box_top4.usd` asset and copying the reset pose / target pose below
into the Isaac Lab config, unless the V4 task intentionally wants to inherit
the authored drive attrs from the test file.

Top4 asset provenance:

```text
source asset = assets/robot/usd/kbot_box_top3.usd
generator = scripts/asset/mirror_top3_right_leg_to_left.py
output = assets/robot/usd/kbot_box_top4.usd
```

The Top4 generation mirrors the right lower leg and foot onto the left side,
preserves symmetric body frames, recloses the left knee and ankle joint frames,
and restores the mirrored left foot collision metadata as guide-purpose
convex-hull collision geometry.

## V4 Starter Pose

Use the settled joint state as the reset pose. These are the final dynamic
control joint positions from a raw USD playback of `kbot_box_top4_test2.usd`
after 300 sim steps.

```text
left_hip_pitch_04    +0.296467274 rad   +16.986 deg
left_hip_roll_03     +0.000730005 rad    +0.042 deg
left_hip_yaw_03      -0.000167742 rad    -0.010 deg
left_knee_04         +0.527403533 rad   +30.218 deg
left_ankle_02        -0.226248741 rad   -12.963 deg

right_hip_pitch_04   -0.296534926 rad   -16.990 deg
right_hip_roll_03    +0.000735207 rad    +0.042 deg
right_hip_yaw_03     -0.000100786 rad    -0.006 deg
right_knee_04        -0.528043807 rad   -30.255 deg
right_ankle_02       +0.226889685 rad   +13.000 deg
```

Recommended initial root height:

```text
root z = 0.8565 m
```

The raw playback settled around:

```text
final dynamic-control base z = 0.855528 m
final foot z = left 0.036523 m, right 0.036528 m
```

## V4 Starter Targets

These are the saved drive target positions from `kbot_box_top4_test2.usd`.
Isaac's Physics Inspector shows them in degrees. Use the radian values in
Isaac Lab config.

```text
left_hip_pitch_04    +0.296805908 rad   +17.006 deg
left_hip_roll_03     +0.000131899 rad    +0.008 deg
left_hip_yaw_03      -0.000074645 rad    -0.004 deg
left_knee_04         +0.516804885 rad   +29.611 deg
left_ankle_02        -0.226892803 rad   -13.000 deg

right_hip_pitch_04   -0.296826381 rad   -17.007 deg
right_hip_roll_03    +0.000136430 rad    +0.008 deg
right_hip_yaw_03     -0.000010827 rad    -0.001 deg
right_knee_04        -0.517032685 rad   -29.624 deg
right_ankle_02       +0.226892803 rad   +13.000 deg
```

The target pose and settled pose are close but not identical. The difference is
expected because the robot settles under contact and load. For V4 reset, use
the settled pose above. If a separate initial hold target is needed, use the
drive target pose.

## Drive Values Used For The GUI Pose

The Physics Inspector pose now uses the unscaled drive stiffness/damping values
that match the normal explicit actuator groups. These numbers should still be
treated as simulation tuning, not motor identification.

```text
hip pitch + knee:
  damping    = 4.0
  stiffness  = 45.0

hip roll:
  damping    = 3.0
  stiffness  = 35.0

hip yaw:
  damping    = 2.0
  stiffness  = 25.0

ankle:
  damping    = 1.0
  stiffness  = 12.0
```

The Inspector table order is `Damping`, then `Stiffness`, then target position.
Do not swap the first two columns.

Earlier posed-start tests used a `57.3` gain scale. That number is effectively
`180/pi`, a radians/degrees conversion hack used to match implicit-actuator and
GUI/raw-USD behavior. The Top4 GUI starter capture above used unscaled drive
values, but the first useful V4 training preview uses the same scaled implicit
actuator path as V3 because the explicit-drive comparison fell/backslid.

## Validation Command

The pose and target evidence above came from:

```bash
.venv/bin/python scripts/asset/probe_raw_usd_play.py \
  --headless \
  assets/robot/usd/kbot_box_top4_test2.usd \
  --steps 300
```

The relevant result was:

```text
RAW_USD final_dc_base_xyz=(-0.00179981, -0.00069814, 0.85552770)
RAW_USD final_dc_foot_xyz=[
  (0.02116740, 0.16532391, 0.03652318),
  (0.02103375, -0.16695546, 0.03652753)
]
```

## V4 Implementation Notes

Current preview implementation:

```text
1. KBOT_TOP4_CFG points at assets/robot/usd/kbot_box_top4.usd.
2. V4 has its own task registration; V3 is not renamed or reused.
3. init_state.pos.z is 0.8565.
4. init_state.joint_pos uses the settled V4 starter pose table above.
5. The reset/action default pose is the same settled starter pose.
6. The zero-joint USD pose is not used as the V4 reset pose.
7. The V4 task sets the Top4 asset actuators to the scaled implicit
   actuator configs used by the posed-start V3 lineage.
```

Start V4 from the latest useful V3 gait-cycle reward stack unless a separate
reward reset is intentionally planned. The asset and reset pose change should
be isolated first so failures can be attributed to the new robot model rather
than to simultaneous reward churn.

## Open Questions

```text
Should the reset pose use the settled joint positions exactly, or should reset
and target both use the saved drive target pose for a simpler first test?
```
