# Box Top V3 Report

Date: 2026-06-04

## Status

This is V3.

The live task registration is:

```text
task = Isaac-KBot-Forward-Flat-V3-648HandTuned-v0
env cfg = KBotForwardFlatV3HandTuned648EnvCfg
parent topology = KBotForwardFlatV25PoseGaitQuality648CompatEnvCfg
```

V3 is a parallel pipeline. It should not be mixed with the later V2.5 S4
experiments or with the V1 hybrid inference/fine-tune branch. The V3 README
describes the original neutral restart plan; the live V3 code has since moved
into hand-tuned reward experiments.

Current result: V3 has produced useful behaviors, but no keeper yet. The
latest changed-weight run raised scalar reward by exploiting early
bad-orientation termination, not by walking.

## Why V3 Exists

V2.5 found a stable posed-start policy, but the best policy around
`model_648.pt` still exploited a tiny-step / high-cadence gait. Later V2.5
experiments added step gates, dense step progress, cadence penalties, and
chatter terms, but those changes made it difficult to separate the source of
each failure.

V3 restarts from iteration 0 using the reward topology that trained
`model_648.pt`, then hand-tunes one explicit V3 block. The purpose is to keep
the useful authored-pose stability while pushing toward larger steps, better
clearance, and real forward speed.

## Current V3 Reward Direction

The current live V3 gait schedule is fixed from one central block:

```text
target_root_speed = 0.375 m/s
target_step_length = 0.60 m
first_step_fraction = 0.5
swing_phase_fraction = 0.375
target_cycle_hz = 0.375 / 0.60 = 0.625 Hz
target_cycle_period = 1.60 s
target_air_time = 0.375 * 1.60 = 0.60 s
first_step_length = 0.30 m
```

The main hand-tuned pressures are:

```text
forward speed:
  track_lin_vel_xy_exp.weight = 15.0
  world_forward_velocity_clip.weight = 15.0
  max world forward clip = 0.375 m/s

centerline and heading gates:
  forward rewards are multiplied by upright, y=0, and world-heading gates
  centerline_width_sq = 0.01
  heading_width_sq = 0.01

foot geometry:
  foot_sagittal_separation_l1.weight = -12.0
  foot_sagittal_separation_l1.target_length = 0.60 m
  swing_foot_overtake_l1.weight = -150.0
  swing_foot_overtake_l1.target_length = 0.30 m
  foot_lateral_spacing_l1.weight = -30.0
  foot_signed_lateral_clearance_l1.weight = -24.0
  foot_sole_lateral_lane_max_l1.weight = -100.0

straightness:
  root_lateral_position_l2.weight = -500.0
  lateral_away_from_center_l2.weight = -600.0
  world_heading_l2.weight = -1000.0
  yaw_rate_l2.weight = -200.0

roll and symmetry:
  root_lateral_tilt_l2.weight = -90.0
  root_lateral_tilt_ema_l2.weight = -12000.0
  signed_joint_pair_ema_symmetry_l2.weight = -200.0
  contact_duty_symmetry_l2.weight = -10.0
  alternating_step_symmetry_l2.weight = -8.0

clearance:
  swing_sole_clearance.weight = 1.0
  target_height = 0.010 m
  drag_floor = 0.002 m
  drag_weight = 3.0

target foot placement:
  dense_swing_step_length.weight = 0.0
  dense_foot_swing_speed.weight = 40.0
  dense_swing_foot_target_location_exp.weight = 500.0
  dense_swing_foot_target_location_exp.target_length = 0.60 m
  dense_swing_foot_target_location_exp.first_target_fraction = 0.5
  dense_swing_foot_target_location_exp.linear_progress_scale = 1.0
  dense_swing_foot_target_location_exp.target_left_y = +0.1582 m
  dense_swing_foot_target_location_exp.target_right_y = -0.1582 m
  dense_swing_foot_target_location_exp.x_scale = 0.15 m
  dense_swing_foot_target_location_exp.y_scale = 0.08 m
  foot_retreat.weight = -12.0
  foot_retreat.retreat_epsilon = 0.002 m

swing speed and sole track:
  dense_foot_swing_speed.target_air_time = 0.60 s
  dense_foot_swing_speed.target_avg_speed = 2.0 m/s
  dense_foot_swing_speed.max_step_credit = 2.0
  dense_foot_swing_speed.target_left_y = +0.1582 m
  dense_foot_swing_speed.target_right_y = -0.1582 m
  dense_foot_swing_speed.y_scale = 0.04 m
  dense_foot_swing_speed.y_linear_radius = 0.12 m
  dense_foot_swing_speed.min_height = 0.005 m
  dense_foot_swing_speed.max_height = 0.020 m
  dense_foot_swing_speed.z_scale = 0.005 m
```

## Forward Reward Gate Math

The current forward rewards are not just velocity rewards. They are multiplied
by three gates:

```text
upright_gate = 1 if root_height > 0.76 and root_tilt < 0.45, else 0

centerline_gate = exp(-((world_y - 0)^2) / centerline_width_sq)

heading_gate = exp(-(forward_y^2 + max(-forward_x, 0)^2) / heading_width_sq)
```

So the velocity-tracking reward is effectively:

```text
R_v = w * exp(-||v_cmd_xy - v_body_xy||^2 / std^2)
          * upright_gate
          * centerline_gate
          * heading_gate
```

The clipped world-forward reward is effectively:

```text
R_x = w * clip(v_world_x, 0, 0.375)
          * upright_gate
          * centerline_gate
          * heading_gate
```

This fixed the worst exploit where the policy could earn forward reward while
falling or drifting far from `y=0`. The downside is that the tight gates can
also remove too much forward-learning signal.

## Target Foot Location Math

The June 4 V3 branch replaced the dense swing-step-length reward with a dense
target-location reward for the swing foot. It now uses a half-length first
target, then full-length targets after the first touchdown:

```text
target_length_first = 0.5 * 0.60 = 0.30 m
target_length_after_first = 0.60 m

target_x_i = takeoff_x_i + active_target_length
target_y_L = env_origin_y + 0.1582
target_y_R = env_origin_y - 0.1582

e_x = (foot_x_i - target_x_i) / 0.15
e_y = (foot_y_i - target_y_i) / 0.08

gaussian_i = exp(-(e_x^2 + e_y^2))

stance_x_i = other_foot_x
progress_span_i = target_x_i - stance_x_i
linear_progress_i = clamp((foot_x_i - stance_x_i) / progress_span_i, 0, 1)
                    if progress_span_i > 0
                    else 0

R_target = 500 * sum_i 1[swing_i and has_takeoff_i]
              * (gaussian_i + linear_progress_i)
              * moving_gate
              * upright_gate
```

Positive body/world `y` is the left-foot lane in the current task convention.
This matches the existing lateral-lane rewards.

The foot-retreat term is a one-shot fixed penalty after contact. It anchors the
stance foot to the maximum `x` reached during the preceding air phase:

```text
P_retreat = -12 * sum_i 1[
    in_contact_i
    and foot_x_i < air_peak_x_i - 0.002
    and not already penalized for this stance
]
```

This is meant to punish the exploit where the swing foot reaches forward,
touches, then drags or retreats backward instead of accepting weight transfer.
The implementation returns `penalty / dt`, so after Isaac Lab's reward `dt`
scaling the cost is fixed at about `-12` per retreating foot, not proportional
to retreat distance.

## Dense Swing Speed Math

The current dense swing-speed reward is an early-swing speed watermark. It is
also the sole-track reward: the speed credit is multiplied by lateral track and
low-clearance gates.

```text
target_avg_speed = target_length / (0.5 * target_air_time)
                 = 0.60 / (0.5 * 0.60)
                 = 2.0 m/s

active_i = swing_i
           and air_time_i <= 0.5 * target_air_time
           and upright_gate
           and moving_gate

wm_vx_i = max positive world-foot-vx seen so far in this active early swing
speed_credit_i = clamp(wm_vx_i / 2.0, 0, 2.0)

y_error_i = abs(sole_y_i - target_y_i)
y_score_i = max(
    exp(-(y_error_i / 0.04)^2),
    clamp(1 - y_error_i / 0.12, 0, 1)
)

z_error_i = max(0, 0.005 - sole_z_i) + max(0, sole_z_i - 0.020)
z_score_i = exp(-(z_error_i / 0.005)^2)

R_speed = 40 * sum_i active_i * speed_credit_i * y_score_i * z_score_i
```

The reward is dense only during the first half of the target swing. It ends
after `0.30 s` of air time for that foot under the current schedule.

## Run Evidence

| Run | Parent | Checkpoint | Key change | Result |
|---|---|---:|---|---|
| `2026-05-31_21-14-35_v3_speed075_step060_clearance_overtake030_w100_from_200_to_300_save25` | earlier V3 speed/step branch | 300 | Useful parent with clearance and overtake pressure | Selected as the current V3 branching point |
| `2026-06-02_03-04-18_v3_speed075_step060_centerline_heading_gate_from_20260531_211435_m300_to_800_save25_20260602` | model 300 | 800 | Added centerline and heading gates | Stable, speed mean 0.369 m/s, final x 10.95 m, final y -1.20 m, APV 100%, J/m 95.3 |
| `2026-06-02_04-10-47_v3_centerline_heading_gate_continue_m800_to_2000_save25_20260602` | model 800 | 2000 | Continued the wider-gate branch | Stable and fast, speed mean 0.664 m/s, final x 18.80 m, but final y -5.34 m, so not acceptable |
| `2026-06-02_05-00-34_v3_tight_gauss_from_20260531_211435_m300_to_800_save25_20260602` | model 300 | 800 | Tightened Gaussian gates to `0.0025` and `0.01` | Stable and centered, final y -0.052 m and fsep 0.312 m, but speed collapsed to 0.0105 m/s mean and J/m rose to 1975 |
| `2026-06-04_01-15-49_v3_dense_target_location_from_300_to_800_save25_20260604` | model 300 | killed | First target-location launch used the wrong iteration count and would have continued past 800 | Killed; keep only as provenance |
| `2026-06-04_01-20-16_v3_dense_target_location_from_300_to_800_save25_corrected_20260604` | model 300 | 799 | Disabled dense step length, added target foot location and stance retreat penalty | Stable training, but `dense_swing_foot_target_location_exp` stayed near zero; final training log only reached about 0.0005 |
| `2026-06-04_01-33-56_v3_dense_target_location_m799_to_800_20260604` | corrected model 799 | 800 | Tiny continuation to produce exact `model_800.pt` | Stable video, no falls, final x 10.22 m, final y 0.44 m, speed mean 0.344 m/s, APV 60%, J/m 85.6 |
| `2026-06-04_09-05-58_v3_weights_vclip15_swing40_target500_4096envs_from_20260531_211435_m300_to_800_save25_20260604` | model 300 | 799 | 4096 envs; lowered forward weights to 15, raised swing speed to 40, raised target location to 500, added swing-speed sole track and half first target | Failed. Scalar reward rose, but final training had mean episode length 36.36, `track_lin_vel_xy_exp=0`, `bad_orientation=0.9985`, and target-location reward about 100.44 |
| `2026-06-04_09-18-59_v3_weights_vclip15_swing40_target500_4096envs_from_20260604_090558_m799_to_800_materialize` | 09:05 model 799 | 800 | Tiny continuation to produce exact `model_800.pt` | Video confirms no walking: final x 0.064 m, final y 0.245 m, speed mean 0.068 m/s, APV 10%, J/m 9475 |

## Latest Tight-Gate Metrics

For `2026-06-02_05-00-34.../model_800.pt`:

```text
fall_reset_count = 0
speed_mean = 0.0105 m/s
speed_final = -0.2337 m/s
command_speed = 0.75 m/s
final_x = 0.343 m
final_y = -0.052 m
root_height_mean = 0.852 m
hud_fsep = 0.312 m
hud_ksep = 0.323 m
approved_step_fraction = 40%
J/m = 1975
left swing clearance = 2.8 mm
right swing clearance = 1.5 mm
cycle length = 0.027 m
cycle cadence = 0.381 Hz
```

This is not a walking keeper. It is useful evidence that the tight Gaussian
centerline/heading gate suppresses drift but also suppresses useful forward
locomotion.

## Latest Target-Location Metrics

For `2026-06-04_01-33-56.../model_800.pt`:

```text
fall_reset_count = 0
speed_mean = 0.344 m/s
speed_final = 0.278 m/s
command_speed = 0.75 m/s
final_x = 10.223 m
final_y = 0.440 m
root_height_mean = 0.855 m
hud_fsep = 0.289 m
hud_ksep = 0.318 m
approved_step_fraction = 60%
J/m = 85.6
left swing clearance = 7.6 mm final HUD, 6.6 mm mean
right swing clearance = 4.6 mm final HUD, 5.1 mm mean
L step cadence = 12.5 Hz
R step cadence = 13.2 Hz
cycle cadence = 7.8 Hz
cycle length = 0.042 m
```

This branch is a better moving policy than the tight-gate branch, but it is
still high-cadence and below commanded speed. It also confirms that a pure
Gaussian foot-target reward is too sparse from the current `model_300` state:
the target-location scalar was effectively zero for most of training.

## Latest Changed-Weight Metrics

For `2026-06-04_09-18-59.../model_800.pt`, rendered from the changed-weight
4096-env run:

```text
fall_reset_count = 0
speed_mean = 0.068 m/s
speed_final = 0.432 m/s
command_speed = 0.375 m/s
final_x = 0.064 m
final_y = 0.245 m
root_height_mean = 0.833 m
hud_fsep = 0.298 m
hud_ksep = 0.327 m
approved_step_fraction = 10%
J/m = 9475
left swing clearance = 68 mm final HUD, 68 mm mean
right swing clearance = 22 mm final HUD, 22 mm mean
L step length = 0.0009 m
R step length = -0.0049 m
cycle length = -0.015 m
cycle cadence = 3.09 Hz
```

Training evidence from the main 09:05 run is worse than the video summary:
by iteration 799/800, `track_lin_vel_xy_exp` was exactly zero, mean episode
length was only 36.36 steps, `bad_orientation` was 0.9985, and
`dense_swing_foot_target_location_exp` had climbed to 100.44. That means the
weight change made the target-location scalar dominant enough to reward an
early-fall behavior.

## Current Conclusion

The V3 branch is real and currently active. The wider centerline-heading branch
walked, but drifted in world `y`. The tight-gate branch stayed centered, but
stopped walking. The June 4 target-location branch walked without falls, but
the pure Gaussian target-location reward was almost invisible to PPO from the
current `model_300` swing path.

The latest linear-plus-Gaussian target run did provide a visible target scalar,
but at the current weights it created a worse exploit: the policy learned to
collect target-location reward while terminating almost immediately by bad
orientation. The problem is no longer sparsity alone. The target-foot stack is
now too dominant relative to survival and useful forward locomotion.

The next useful V3 change should not continue the latest changed-weight
`model_800`. It should either roll back to the stable target-location
`2026-06-04_01-33-56` checkpoint or restart again from the original model 300
with much lower/scheduled target-foot weights.

## Recommended Next Experiments

1. Do not continue the 09:05/09:18 changed-weight run as a walking candidate.
It is an early-fall target-reward exploit.

2. Keep the target-foot math, but reduce or schedule its weight:

```text
dense_swing_foot_target_location_exp.weight: 500 -> much lower, or ramp up
dense_foot_swing_speed.weight: 40 -> lower until survival/speed survives

start from model_300 or the stable 01:33 model_800, not from the collapsed
09:18 model_800
```

The latest run proves that a visible target scalar is not enough; it has to be
kept below the pressure to stay upright and move the base forward.

3. Split forward reward into base and bonus parts:

```text
base forward reward: upright-gated only, small weight
centerline-heading bonus: upright + y + heading gated, larger weight
```

This prevents the policy from losing all forward signal once it drifts outside
the Gaussian.

4. Ramp the Gaussian widths instead of jumping directly to tight gates:

```text
start centerline_width_sq around 0.01
start heading_width_sq around 0.04
then tighten gradually toward 0.0025 and 0.01 only after speed survives
```

5. Consider randomized initial `y` offsets as a separate return-to-center
experiment:

```text
spawn y in a small band around 0
reward reducing |y| without rewarding S-shaped slalom
keep lateral_velocity_l2 and lateral_away_from_center_l2 active
```

6. Review intermediate checkpoints from the 09:05 run if a salvage point is
needed. The run collapsed after the early mid-training window, so `model_500`
or `model_525` may be more informative than the final checkpoint.

## Artifact Index

```text
V3 report:
policies/box_top_v3/V3_REPORT.md

latest tight-gate run:
logs/rsl_rl/kbot_forward_flat/2026-06-02_05-00-34_v3_tight_gauss_from_20260531_211435_m300_to_800_save25_20260602

tight-gate video JSON:
logs/rsl_rl/kbot_forward_flat/2026-06-02_05-00-34_v3_tight_gauss_from_20260531_211435_m300_to_800_save25_20260602/videos/play/trailing-hud-model_800-v3-tight-gauss-from-m300-to-800-30s.json

fast but drifting continuation:
logs/rsl_rl/kbot_forward_flat/2026-06-02_04-10-47_v3_centerline_heading_gate_continue_m800_to_2000_save25_20260602

fast but drifting video JSON:
logs/rsl_rl/kbot_forward_flat/2026-06-02_04-10-47_v3_centerline_heading_gate_continue_m800_to_2000_save25_20260602/videos/play/trailing-hud-model_2000-v3-centerline-heading-gate-from-m800-to-2000-30s.json

target-location corrected run:
logs/rsl_rl/kbot_forward_flat/2026-06-04_01-20-16_v3_dense_target_location_from_300_to_800_save25_corrected_20260604

target-location exact model_800 run:
logs/rsl_rl/kbot_forward_flat/2026-06-04_01-33-56_v3_dense_target_location_m799_to_800_20260604

target-location model_800 video:
logs/rsl_rl/kbot_forward_flat/2026-06-04_01-33-56_v3_dense_target_location_m799_to_800_20260604/videos/play/trailing-hud-model_800-v3-dense-target-location-30s.mp4

target-location model_800 video JSON:
logs/rsl_rl/kbot_forward_flat/2026-06-04_01-33-56_v3_dense_target_location_m799_to_800_20260604/videos/play/trailing-hud-model_800-v3-dense-target-location-30s.json

target-location corrected reward graph:
logs/rsl_rl/kbot_forward_flat/2026-06-04_01-20-16_v3_dense_target_location_from_300_to_800_save25_corrected_20260604/graficos/2026-06-04_01-20-16_combined_reward_components.png

target-location exact model_800 reward graph:
logs/rsl_rl/kbot_forward_flat/2026-06-04_01-33-56_v3_dense_target_location_m799_to_800_20260604/graficos/2026-06-04_01-33-56_combined_reward_components.png

changed-weight 4096-env main run:
logs/rsl_rl/kbot_forward_flat/2026-06-04_09-05-58_v3_weights_vclip15_swing40_target500_4096envs_from_20260531_211435_m300_to_800_save25_20260604

changed-weight exact model_800 run:
logs/rsl_rl/kbot_forward_flat/2026-06-04_09-18-59_v3_weights_vclip15_swing40_target500_4096envs_from_20260604_090558_m799_to_800_materialize

changed-weight model_800:
logs/rsl_rl/kbot_forward_flat/2026-06-04_09-18-59_v3_weights_vclip15_swing40_target500_4096envs_from_20260604_090558_m799_to_800_materialize/model_800.pt

changed-weight model_800 video:
logs/rsl_rl/kbot_forward_flat/2026-06-04_09-05-58_v3_weights_vclip15_swing40_target500_4096envs_from_20260531_211435_m300_to_800_save25_20260604/videos/play/trailing-hud-model_800-v3-weights-vclip15-swing40-target500-4096envs-30s.mp4

changed-weight model_800 video JSON:
logs/rsl_rl/kbot_forward_flat/2026-06-04_09-05-58_v3_weights_vclip15_swing40_target500_4096envs_from_20260531_211435_m300_to_800_save25_20260604/videos/play/trailing-hud-model_800-v3-weights-vclip15-swing40-target500-4096envs-30s.json

changed-weight reward graph:
logs/rsl_rl/kbot_forward_flat/2026-06-04_09-05-58_v3_weights_vclip15_swing40_target500_4096envs_from_20260531_211435_m300_to_800_save25_20260604/graficos/2026-06-04_09-05-58_combined_reward_components.png

changed-weight reward CSV:
logs/rsl_rl/kbot_forward_flat/2026-06-04_09-05-58_v3_weights_vclip15_swing40_target500_4096envs_from_20260531_211435_m300_to_800_save25_20260604/metricas/2026-06-04_09-05-58_reward_components.csv
```
