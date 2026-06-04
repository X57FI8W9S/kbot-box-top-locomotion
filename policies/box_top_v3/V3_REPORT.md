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

Current result: V3 has produced useful behaviors, but no keeper yet.

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

The current live V3 target command is fixed:

```text
vx command = 0.75 m/s
```

The main hand-tuned pressures are:

```text
forward speed:
  track_lin_vel_xy_exp.weight = 30.0
  world_forward_velocity_clip.weight = 30.0
  max world forward clip = 0.75 m/s

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
  signed_joint_pair_ema_symmetry_l2.weight = -90.0
  contact_duty_symmetry_l2.weight = -10.0
  alternating_step_symmetry_l2.weight = -8.0

clearance:
  swing_sole_clearance.weight = 1.0
  target_height = 0.010 m
  drag_floor = 0.002 m
  drag_weight = 3.0

target foot placement:
  dense_swing_step_length.weight = 0.0
  dense_foot_swing_speed.weight = 20.0
  dense_swing_foot_target_location_exp.weight = 30.0
  dense_swing_foot_target_location_exp.target_length = 0.60 m
  dense_swing_foot_target_location_exp.target_left_y = +0.1582 m
  dense_swing_foot_target_location_exp.target_right_y = -0.1582 m
  dense_swing_foot_target_location_exp.x_scale = 0.15 m
  dense_swing_foot_target_location_exp.y_scale = 0.08 m
  stance_foot_retreat_l1.weight = -12.0
  stance_foot_retreat_l1.retreat_epsilon = 0.002 m
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
R_x = w * clip(v_world_x, 0, 0.75)
          * upright_gate
          * centerline_gate
          * heading_gate
```

This fixed the worst exploit where the policy could earn forward reward while
falling or drifting far from `y=0`. The downside is that the tight gates can
also remove too much forward-learning signal.

## Target Foot Location Math

The June 4 V3 branch replaced the dense swing-step-length reward with a dense
target-location reward for the swing foot. It uses the same foot's takeoff
position as the sagittal reference:

```text
target_x_i = takeoff_x_i + 0.60
target_y_L = env_origin_y + 0.1582
target_y_R = env_origin_y - 0.1582

e_x = (foot_x_i - target_x_i) / 0.15
e_y = (foot_y_i - target_y_i) / 0.08

R_target = 30 * sum_i 1[swing_i and has_takeoff_i]
              * exp(-(e_x^2 + e_y^2))
              * moving_gate
              * upright_gate
```

Positive body/world `y` is the left-foot lane in the current task convention.
This matches the existing lateral-lane rewards.

The stance retreat term is a one-shot penalty after touchdown:

```text
P_retreat = -12 * sum_i 1[
    in_contact_i
    and foot_x_i < touchdown_x_i - 0.002
    and not already penalized for this stance
]
```

This is meant to punish the exploit where the swing foot reaches forward,
touches, then drags or retreats backward instead of accepting weight transfer.

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

## Current Conclusion

The V3 branch is real and currently active. The wider centerline-heading branch
walked, but drifted in world `y`. The tight-gate branch stayed centered, but
stopped walking. The June 4 target-location branch walked without falls, but
the pure Gaussian target-location reward was almost invisible to PPO from the
current `model_300` swing path.

The next useful V3 change should not be another long continuation from the
tight-gate `model_800` or another pure-Gaussian target-location continuation.
The target-foot idea should be kept, but the reward shape needs a nonzero
far-field gradient so the policy can see the direction before it is already
near the landing target.

## Recommended Next Experiments

1. Replace pure Gaussian target-location with a linear-plus-local bonus shape:

```text
distance_to_target = sqrt(e_x^2 + e_y^2)

R_target = w_linear * progress_toward_target
         + w_bonus * exp(-distance_to_target^2)
```

This keeps gradient when the foot is far from the desired landing point, then
adds extra reward near the target. Keep the one-shot stance retreat penalty.

2. Split forward reward into base and bonus parts:

```text
base forward reward: upright-gated only, small weight
centerline-heading bonus: upright + y + heading gated, larger weight
```

This prevents the policy from losing all forward signal once it drifts outside
the Gaussian.

3. Ramp the Gaussian widths instead of jumping directly to tight gates:

```text
start centerline_width_sq around 0.01
start heading_width_sq around 0.04
then tighten gradually toward 0.0025 and 0.01 only after speed survives
```

4. Consider randomized initial `y` offsets as a separate return-to-center
experiment:

```text
spawn y in a small band around 0
reward reducing |y| without rewarding S-shaped slalom
keep lateral_velocity_l2 and lateral_away_from_center_l2 active
```

5. Review intermediate checkpoints from the tight run and target-location run.
The final checkpoint is not necessarily the best one.

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
```
