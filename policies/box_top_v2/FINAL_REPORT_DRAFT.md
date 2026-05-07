# Box-Top Policy V2 Final Report Draft

This file should be updated during v2, not reconstructed at the end. It is intentionally a draft. Do not fill result sections with guesses.

## 1. Purpose

V2 restarts the flat-ground box-top locomotion policy with a cleaner training procedure.

Primary product:

```text
a reusable locomotion training and evaluation procedure
```

Secondary product:

```text
a better simplified box-top walking policy
```

The box top is a simplification that removes torso/arm training from scope. The long-term target is to adapt the procedure to a fuller humanoid and later to obstacle avoidance, vision-conditioned navigation, path planning, and fall recovery.

## 2. Baseline From V1

Reference checkpoints to compare against:

```text
V1 stable/gait-quality baseline:
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/model_11791.pt

V1 final sole-contact push:
logs/rsl_rl/kbot_forward_flat/2026-04-29_07-58-47/model_11990.pt
```

Baseline videos:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/videos/play/trailing-side-hud-model_11791-final.mp4
logs/rsl_rl/kbot_forward_flat/2026-04-29_07-58-47/videos/play/trailing-side-hud-model_11990-30s.mp4

Current headless re-test videos:
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/videos/play/trailing-side-hud-model_11791-headless-v2eval.mp4
logs/rsl_rl/kbot_forward_flat/2026-05-04_05-31-04/videos/play/trailing-side-hud-model_999-headless-v2.mp4
```

Current checkpoint labels:

```text
best all-time: logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/model_11791.pt
best all-time alternate: logs/rsl_rl/kbot_forward_flat/2026-04-29_07-58-47/model_11990.pt
best V2 so far: logs/rsl_rl/kbot_forward_flat/2026-05-04_19-18-03/model_2795.pt, rejected by evaluator but best current speed/yaw/lateral/root-roll compromise
standard testing mode from 2026-05-04 onward: headless diagnostics plus headless side-by-side trailing/side HUD video
standard output video length from 2026-05-04 onward: 30 s / 1500 control steps
playback reset policy from 2026-05-04 onward: no normal policy reset during video; only reset the rollout/policy if root height falls to or below 0.0 m
HUD averaging from 2026-05-04 onward: overlay averages use the latest 5 full gait cycles once enough same-side touchdowns exist; the 3.0 s value is only a warmup fallback.
```

V1 known remaining problems:

```text
tiptoe / weak sole contact
persistent roll bias
persistent L/R rolling-average joint asymmetry
knees probably too bent
steps too short
slight under-speed
```

## 3. V2 Design Commitments

- Diagnostics are separate from rewards.
- Checkpoint selection uses hard gates plus scorecards, not scalar reward alone.
- Rolling windows for gait metrics are based on 5 full gait cycles where possible.
- Bias and oscillation are reported separately.
- L/R joint symmetry is computed after mirrored sign normalization.
- Yaw/heading, contact quality, roll bias, symmetry, crouch, and step quality are hard gates.
- Reward terms are kept fewer and less overlapping than V1 unless a diagnostic failure proves a new term is needed.

## 4. V2 Task Configuration

```text
task id: Isaac-KBot-Forward-Flat-V2-v0
play task id: Isaac-KBot-Forward-Flat-V2-Play-v0
source config file: source/kbot_loco/kbot_loco/tasks/locomotion/env_cfg.py
robot asset: original box-top asset through KBOT_CFG; V2 no longer depends on generated pad bodies
episode length: 8 s for training, 60 s for play/evaluation
decimation: 4
num envs: 2048 default, overridden by train/evaluation CLI
command ranges: forward velocity x = 0.15 to 0.30 m/s, lateral velocity y = 0.0 m/s, yaw rate z = 0.0 rad/s, heading = 0 rad
domain randomization: friction, base mass, base COM, reset pose, reset velocity, reset joint position scaling
termination settings: time-out only in current V2 training; base_contact, bad_orientation, low_body, and locked_knees are disabled
```

Current contact sensor status:

```text
The original box-top asset exposes whole-foot contact bodies named foot1 and foot3.
A first generated pad asset exists at assets/robot/usd/kbot_box_top3_pads.usda, but V2 has moved away from using it for training.
The generated pads were useful for investigation, but they added enough asset/debugging cost that the current plan is pad-free diagnostics first.
Basic foot contact comes from foot1/foot3 contact time.
Sole quality is estimated from each foot body's local plane vectors:
  sole_normal_w  = quat_apply(foot_quat_w, [0, 0, 1])
  sole_forward_w = quat_apply(foot_quat_w, [1, 0, 0])
  sole_lateral_w = quat_apply(foot_quat_w, [0, 1, 0])
Full support, toe-down, heel-down, and edge-walk states are currently diagnostic proxies, not true sub-foot contacts.
```

## 5. V2 Reward Function

V2 should keep fewer reward pressures than V1. The current direction is:

```text
many diagnostics, fewer rewards
promote a diagnostic to a reward only after repeated scorecard failure proves it is needed
do not add heel/toe/edge rewards just because the indicators exist
```

Current active V2 reward weights relative to the V1 final reward:

```text
track_lin_vel_xy_exp                 +2.0
track_ang_vel_z_exp                  +3.0
feet_air_time                        +0.75
alternating_foot_phase               +0.25
upright_alive                        +8.0, minimum_height = 0.70, max_tilt = 0.35
flat_orientation_l2                  -15.0
lateral_velocity_l2                  -5.0
yaw_rate_l2                          -5.0
world_heading_l2                     -20.0
root_lateral_tilt_l2                 -70.0
root_lateral_tilt_ema_l2             -350.0, tau_s = 2.5
forward_velocity_below_l2            -8.0, minimum_velocity = 0.12
foot_lateral_spacing_l1              -5.0
foot_signed_lateral_clearance_l1     -20.0
foot_lateral_lane_l1                 -5.0, tolerance = 0.04
foot_lateral_lane_max_l1              0.0
leg_frontal_plane_l1                 -4.0, tolerance = 0.04
left_leg_frontal_plane_l1             0.0
right_leg_frontal_plane_l1            0.0
max_leg_frontal_plane_l1              0.0
foot_sagittal_separation_l1          -3.0, target_length = 0.22
swing_foot_overtake_l1               -10.0, target_length = 0.18
foot_parallel_l2                     -1.0
foot_toe_in_l2                       -6.0
foot_flat_l2                          0.0
stance_foot_flat_l2                  -2.0, uses single_stance_foot_flat_l2 in V2
hip_roll_yaw_position_l2             -8.0
hip_roll_yaw_position_ema_l2         -24.0, tau_s = 2.5
hip_roll_position_ema_5cycle_l2      -90.0, tau_s = 5.0
low_body_l2                          -120.0, minimum_height = 0.70
base_height_l2                       -35.0, target_height = 0.88
knee_extension_l1                    -18.0, min_bend = 0.35
```

Inherited base reward terms that remain active should be confirmed from the resolved Isaac Lab manager printout before calling this equation final.

### Branch Template

```text
branch name:
run directory:
warm start:
checkpoint range:
purpose:
reward changes:
expected failure mode:
```

Exact reward equation:

```text
R =
  ...
```

Term explanations:

```text
term:
  units:
  formula:
  intent:
  risk:
```

## 6. Diagnostics Module

Implementation location:

```text
scripts/diagnostics/evaluate_checkpoint.py
```

Required outputs:

```text
diagnostics/<checkpoint>/metrics.json
diagnostics/<checkpoint>/gait_cycles.csv
diagnostics/<checkpoint>/steps.csv
diagnostics/<checkpoint>/step_events.csv
diagnostics/<checkpoint>/summary.md
diagnostics/<checkpoint>/dashboard.html
```

Decision outputs:

```text
APPROVE
REJECT
REVIEW_VIDEO
```

Current implemented scorecard coverage:

```text
speed_mean_mps
command_speed_mean_mps
speed_tracking_ratio
yaw_drift_rad_per_m
lateral_drift_m_per_m
root_roll_mean_5cycle
root_roll_rms_centered_5cycle
hip_roll_mean_abs_5cycle_rad
hip_roll_rms_centered_5cycle_rad
hip_yaw_mean_abs_5cycle_rad
knee_abs_mean_5cycle_rad
root_height_mean_m
root_height_p05_m
double_support_fraction
airborne_fraction
full_support_fraction_left/right
toe_only_fraction_left/right
heel_only_fraction_left/right
sole_normal_z_mean_left/right
stance_sole_tilt_l2_mean
toe_down_proxy_fraction_left/right
heel_down_proxy_fraction_left/right
edge_walk_proxy_fraction_left/right
inner_edge_proxy_fraction_left/right
outer_edge_proxy_fraction_left/right
stance_slip_mean_mps
step_count
left_step_count
right_step_count
step_length_mean_m
step_duration_mean_s
left_step_length_mean_m
right_step_length_mean_m
left_step_root_advance_mean_m
right_step_root_advance_mean_m
left_step_duration_mean_s
right_step_duration_mean_s
left_step_length_last5_mean_m
right_step_length_last5_mean_m
left_step_root_advance_last5_mean_m
right_step_root_advance_last5_mean_m
left_step_duration_last5_mean_s
right_step_duration_last5_mean_s
step_duration_std_s
left_right_step_duration_error_mean_s
left_right_step_duration_error_last5_s
step_double_support_ratio_mean
left_step_double_support_ratio_mean
right_step_double_support_ratio_mean
step_full_support_ratio_mean
left_step_full_support_ratio_mean
right_step_full_support_ratio_mean
left_step_full_support_ratio_last5_mean
right_step_full_support_ratio_last5_mean
left_landing_to_opposite_toe_off_last5_mean_s
right_landing_to_opposite_toe_off_last5_mean_s
cycle_count
left_cycle_count
right_cycle_count
cycle_length_mean_m
cycle_root_advance_mean_m
cycle_duration_mean_s
cycle_duration_std_s
cycle_cadence_hz
left_cycle_length_mean_m
right_cycle_length_mean_m
left_cycle_root_advance_mean_m
right_cycle_root_advance_mean_m
left_cycle_duration_mean_s
right_cycle_duration_mean_s
left_cycle_length_last5_mean_m
right_cycle_length_last5_mean_m
left_cycle_root_advance_last5_mean_m
right_cycle_root_advance_last5_mean_m
left_cycle_duration_last5_mean_s
right_cycle_duration_last5_mean_s
left_right_cycle_duration_error_mean_s
left_right_cycle_duration_error_last5_s
cycle_double_support_ratio_mean
cycle_full_support_ratio_mean
left_stance_duration_mean_s
right_stance_duration_mean_s
left_swing_duration_mean_s
right_swing_duration_mean_s
left_duty_factor_mean
right_duty_factor_mean
left_swing_ratio_mean
right_swing_ratio_mean
left_stance_duration_last5_mean_s
right_stance_duration_last5_mean_s
left_swing_duration_last5_mean_s
right_swing_duration_last5_mean_s
left_duty_factor_last5_mean
right_duty_factor_last5_mean
left_cycle_full_support_ratio_last5_mean
right_cycle_full_support_ratio_last5_mean
paired joint mean symmetry errors where left/right names can be matched
```

Step and cycle length definitions:

```text
step_length_m:
  foot placement distance from one touchdown foot x to the opposite touchdown foot x.
  L step = R touchdown x - L touchdown x.
  R step = next L touchdown x - R touchdown x.

cycle_length_m:
  same-foot placement distance from one touchdown to the next touchdown of that same foot.

root_advance_m:
  robot/root forward displacement over the same time interval.

Use root_advance_m when asking how far the robot moved.
Use step_length_m and cycle_length_m when asking how the feet were placed.

stance_duration_s / swing_duration_s / duty_factor:
  computed per same-foot full cycle.
  stance duration is time that cycle foot is in whole-foot contact.
  swing duration is cycle duration minus stance duration.
  duty factor is stance duration / cycle duration.
```

Known diagnostic gaps versus the V1 final report, V1 gait plan, and V2 diagnostics plan:

```text
per-side stance_slip_L/R
contact force / impulse metrics
contact force heel/toe distribution
true heel_contact/toe_contact/forefoot_contact from sub-foot bodies
full_sole_ratio_per_stance_L/R
full_sole_ratio_per_step_L/R
full_sole_support_duration
single_support_duration and single_support_ratio_L/R
duty_factor_L/R
stride_length
cadence
step_length_per_velocity
step duration/length symmetry by event pairs as explicit error terms
stance duration symmetry
full sole support symmetry error
contact force symmetry error
knee_angle_min/max/range_L/R
ankle_angle_mean_L/R
hip_pitch_mean_L/R
box/top roll if exposed separately from root
lateral COM offset and support center offset
timeout fraction and termination cause summary
```

## 7. Hard Gates

A checkpoint is rejected if any gate fails.

Safety:

```text
timeout fraction:
termination causes:
non-foot body contact:
root height:
knee crouch/lock:
```

Direction:

```text
yaw drift:
lateral drift:
heading error:
path curvature:
```

Contact:

```text
toe-only ratio:
heel contact ratio:
full-sole support ratio:
stance slip:
contact force distribution:
```

Roll bias:

```text
base/root roll mean:
box/top roll mean:
roll RMS centered:
hip roll/yaw mean:
```

Symmetry:

```text
normalized L/R joint average errors:
step length symmetry:
step duration symmetry:
stance duration symmetry:
full-sole support symmetry:
```

Gait quality:

```text
step length:
stride length:
cadence:
double support:
single support:
airborne fraction:
```

## 8. Timeline

Add every meaningful v2 run here as soon as it is made.

```text
date: 2026-05-02
run: logs/rsl_rl/kbot_forward_flat/2026-05-02_04-18-19
checkpoint: model_399.pt
code/config change: first V2 task with cleaner reward weights and a 5 s hip-roll EMA penalty; no hard low-body or bad-orientation termination
why it was tried: restart from a simpler reward while keeping the v1 lessons about roll bias, step symmetry, foot placement, and toe-in
result: rejected; training reached full 8 s time-outs but exploited a low/crouched posture with poor speed tracking
decision: add hard bootstrap terminations for low root height and bad orientation before the next run
```

```text
date: 2026-05-02
run: V2 bootstrap guardrail config
checkpoint: pending
code/config change: added low root-height termination below 0.42 m, bad-orientation termination above 0.95 rad, and upright_alive reward requiring root height above 0.55 m and tilt below 0.45
why it was tried: prevent the collapsed time-out exploit observed in the first V2 run
result: rejected after probe; low_body termination saturated at 1.0 almost immediately and prevented useful learning
decision: disable hard low_body and bad_orientation terminations again; keep low_body_l2, upright_alive, termination_penalty, and evaluator gates
```

```text
date: 2026-05-02
run: asset validation, not PPO training
checkpoint: none
code/config change: generated kbot_box_top3_pads.usda with four heel/toe pad rigid bodies; diagnostics prefer heel/toe pad contacts if those bodies are present; heel pads use a 0.04 m lower offset to compensate the simplified toe-low foot posture
why it was tried: whole-foot foot1/foot3 contact cannot truthfully measure full support
result: partial pass; the pads appear as distinct rigid bodies and contact sensor body ids; held-pose validation gives clean air, toe-only, and full-support states, but clean symmetric heel-only remains weak with simple box pads
decision: usable as a first diagnostic asset only; do not make V2 training depend on pads for now
```

```text
date: 2026-05-04
run: diagnostics/config update, not PPO training
checkpoint: none
code/config change: V2 switched back from KBOT_PADS_CFG to KBOT_CFG; evaluator added pad-free sole-plane indicators from foot normal, forward, and lateral vectors
why it was tried: after two days of pad debugging, the lower-risk path is to use whole-foot contact plus vector analysis for sole-quality indicators
result: compile check passed for env_cfg.py and evaluate_checkpoint.py
decision: keep the new sole-plane measurements as diagnostics only; add rewards from them only if repeated evaluations show a specific toe/heel/edge exploit
```

```text
date: 2026-05-04
run: prompt-PDF step/cycle metrics update, not PPO training
checkpoint: none
code/config change: evaluator now records L/R step duration and length from touchdown-to-opposite-touchdown events, same-foot full cycle duration and length, root advance over those intervals, support ratios, stance/swing duty factors, touchdown-to-opposite-toe-off transition time, and last-five rolling means for each side
why it was tried: the prompt PDF and hand diagram define step as foot landing to opposite foot landing, and full cycle as same-foot landing to next same-foot landing
result: compile check passed for evaluate_checkpoint.py
decision: this is the first implemented slice of the prompt-PDF metrics; remaining support/contact-force/posture/symmetry metrics still need follow-up
```

```text
date: 2026-05-04
run: reward cleanup, not PPO training
checkpoint: none
code/config change: V2 disables always-on foot_flat_l2 and points stance_foot_flat_l2 at single_stance_foot_flat_l2, so the foot-horizontal penalty is applied only to the single support foot
why it was tried: double support is a weight-transfer phase and swing feet should not be forced horizontal; the flatness pressure should target the leg actually carrying one-leg support
result: compile check passed for mdp.py and env_cfg.py
decision: retry V2 training with this reward cleanup after resolving the low-body early-collapse issue
```

```text
date: 2026-05-04
run: V2 hard termination rollback, not PPO training
checkpoint: none
code/config change: disabled V2 low_body and bad_orientation hard terminations after the probe run; training now uses timeout-only episodes again
why it was tried: the first V2 restart reached PPO but low_body termination saturated at 1.0 by early iterations, so the policy was not getting useful bad-posture gradients
result: compile check passed for env_cfg.py and mdp.py
decision: rely on soft posture rewards during optimization and reject crouch/fall behavior with evaluator gates instead of ending every episode early
```

```text
date: 2026-05-04
run: V2 timeout-only training after prompt-PDF indicators and stance-foot flatness cleanup
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_05-31-04/model_999.pt
code/config change: trained KBotForwardFlatV2 with KBOT_CFG, whole-foot contacts, timeout-only termination, single-support stance foot flatness, RSL-RL actor/critic compatibility, cycle-based camera smoothing, and prompt-PDF step/cycle metrics in the evaluator
why it was tried: verify whether timeout-only V2 with soft posture penalties can learn a usable gait once the evaluator has enough indicators to reject bad survival strategies
training result: PPO completed 1000 iterations on cuda:0 / NVIDIA GeForce RTX 4060; training scalar reward improved from about -1130 to about -383 and episode length reached the 400-step timeout
diagnostic result: evaluator rejected model_999; speed_tracking_ratio 0.081, root_height_p05_m -0.044, alternating_steps failed with only 3 detected steps, full_support_fraction_left/right 0.0, sole_normal_z_mean_left/right 0.102/0.053, edge_walk_proxy_fraction_left/right 0.990/0.991
headless confirmation: reran the same checkpoint with --headless into diagnostics/model_999_headless and obtained the same rejection metrics, so the apparent fall-through/low-root behavior is not a viewport rendering artifact
headless video: logs/rsl_rl/kbot_forward_flat/2026-05-04_05-31-04/videos/play/trailing-side-hud-model_999-headless-v2.mp4
decision: keep the run as evidence, but do not promote this checkpoint; timeout-only training is mechanically working but the policy is exploiting a low edge/heel contact strategy instead of walking
next action: tune soft posture/contact rewards or reintroduce carefully delayed/gated termination, then retrain and compare with the same evaluator gates
```

```text
date: 2026-05-04
run: V1 best-checkpoint re-test with V2 evaluator, headless
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/model_11791.pt
comparison checkpoint: logs/rsl_rl/kbot_forward_flat/2026-04-29_07-58-47/model_11990.pt
why it was tried: establish a best all-time baseline under the same new step/cycle, sole-plane, and support diagnostics used for V2
result model_11791: REVIEW_VIDEO; speed_tracking_ratio 0.957, root_height_p05_m 0.723, step_count 272, cycle_duration_mean_s 0.220, yaw_drift_rad_per_m 0.006, lateral_drift_m_per_m 0.067, edge_walk_proxy_left/right 0.342/0.406, airborne_fraction 0.255
result model_11990: REVIEW_VIDEO; speed_tracking_ratio 0.969, root_height_p05_m 0.722, step_count 266, cycle_duration_mean_s 0.226, yaw_drift_rad_per_m -0.019, lateral_drift_m_per_m 0.009, edge_walk_proxy_left/right 0.325/0.419, airborne_fraction 0.259
decision: keep model_11791 as best all-time by default because V1 already flagged it as visually safer unless 11990 clearly improves sole contact; model_11990 remains a close alternate with slightly better speed/lateral tracking
headless video: logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/videos/play/trailing-side-hud-model_11791-headless-v2eval.mp4
```

```text
date: 2026-05-04
run: V2 scratch-only posture-first probe
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_06-43-32, stopped early
warm start: none
code/config change: reduced V2 command range to 0.15-0.30 m/s, increased low-body/upright penalties, added upright_alive reward, and kept hard low_body/bad_orientation disabled
why it was tried: restart V2 from scratch while discouraging the low-body timeout exploit without immediately ending every episode
result: rejected by training scalars before evaluation; timeout fraction reached 1.0 while base_height_l2 stayed near -85, low_body_l2 near -230, and upright_alive near 0.39
decision: plain scratch V2 still finds collapsed timeout survival; branch short runs should isolate whether hard termination, standing-first curriculum, conservative action/noise, or reset pose changes can stop the start fall
```

```text
date: 2026-05-04
run: V2 scratch hard-termination short branch
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_06-50-54/model_99.pt
warm start: none
task id: Isaac-KBot-Forward-Flat-V2-Scratch-Hard-v0
code/config change: low-speed command 0.05-0.15 m/s, hard low_body termination at 0.55 m, hard bad_orientation termination at 0.95 rad
why it was tried: prevent collapsed policies from receiving full timeout episodes
result: rejected; low_body termination saturated at 1.0 by the end, timeout fraction stayed 0.0, mean episode length stayed around 26-27 steps
decision: hard cutoffs stop the timeout exploit but do not yet provide a recoverable scratch learning signal
```

```text
date: 2026-05-04
run: V2 scratch standing-first short branch
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_06-52-07/model_99.pt
warm start: none
task id: Isaac-KBot-Forward-Flat-V2-Scratch-Stand-v0
code/config change: command range 0.0-0.05 m/s, disabled feet_air_time, alternating_foot_phase, sagittal separation, overtake, and forward-velocity floor; increased base-height, low-body, and upright rewards
why it was tried: learn tall quiet support before asking for walking
result: rejected; timeout fraction reached 1.0, but base_height_l2 remained around -109, low_body_l2 around -306, and upright_alive around 0.49
decision: standing-first without hard fall cutoffs still learns low collapsed timeout survival
```

```text
date: 2026-05-04
run: V2 scratch standing-first conservative short branch
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_06-53-21/model_99.pt
warm start: none
task id: Isaac-KBot-Forward-Flat-V2-Scratch-Stand-Conservative-v0
code/config change: standing-first branch plus action scale 0.10, PPO init_noise_std 0.05, tight reset pose/velocity noise, and reset joint scale 0.99-1.01
why it was tried: reduce early random action damage while preserving scratch learning
result: rejected; same low collapsed timeout survivor, with base_height_l2 around -108, low_body_l2 around -304, upright_alive around 0.49, and timeout fraction 1.0
decision: conservative action/noise alone does not fix the scratch start-fall problem
```

```text
date: 2026-05-04
run: passive reset stability probes, not PPO training
checkpoint: none
code/config change: probe_kbot_stability.py now supports V2 config selection and explicit hip-pitch, knee, hip-roll, ankle, and root-height reset knobs
why it was tried: determine whether scratch PPO is fighting an unstable initial pose before it has any useful controller
result: default V2 zero-action reset falls over passively. Root heights 0.78 m and 0.88 m both ended after 400 steps with root z around -0.80 m and max_abs_gravity_xy about 1.0. Hip-pitch +/-0.25 rad, lower bent-knee reset, and ankle-compensated bent-knee reset also fell passively.
decision: the current zero-action/default pose is not a passive standing pose. A from-scratch V2 policy likely needs either a balance-only curriculum with hard early fall rejection, a better reset/default posture, or a non-V1 bootstrap method before gait rewards are meaningful.
```

```text
date: 2026-05-04
run: V2 scratch balance-only short branch
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_07-00-30/model_99.pt
warm start: none
task id: Isaac-KBot-Forward-Flat-V2-Scratch-Balance-v0
code/config change: zero velocity command, action scale 0.08, PPO init_noise_std 0.05, 4 s episodes, gait/foot progression rewards disabled, strong upright/base-height/flat-orientation rewards, hard low_body and bad_orientation terminations
why it was tried: test whether an explicit balance-only scratch stage can stop the start-fall before any walking objective is introduced
result: rejected; at iteration 99 timeout fraction was 0.0, bad_orientation termination was about 0.934, low_body termination about 0.880, and mean episode length stayed about 27 steps
decision: balance-only reward shaping alone still does not solve the from-scratch start-fall. The next useful V2-scratch bootstrap should change the initial/default pose, add a curriculum that starts from a known balanced controller/pose without V1 weights, or pretrain a standing policy with a supervised/hand-authored stabilizing action prior.
headless video: logs/rsl_rl/kbot_forward_flat/2026-05-04_07-00-30/videos/play/trailing-side-hud-model_99-headless-scratch-balance.mp4
```

```text
date: 2026-05-04
run: V2 scratch V1-bootstrap reproduction, first attempt
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_07-22-33/model_299.pt
warm start: none
task id: Isaac-KBot-Forward-Flat-V2-Scratch-V1Bootstrap-v0
code/config change: reproduced V1's low-speed reward weights and timeout-only terminations, but still inherited the current 8 s episode length
why it was tried: V1 originally escaped the start-fall problem with soft fall penalties and long timeout-only training rather than hard early terminations
result: rejected; by iteration 299 timeout fraction was 1.0, mean episode length was 400, base_height_l2 was about -28.94, low_body_l2 about -34.46, and feet_air_time was near zero
decision: the V1 reward weights alone were not the missing piece
```

```text
date: 2026-05-04
run: V1 policy/action inspection and fixed-action probes
checkpoint inspected: logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/model_11791.pt
warm start: none for V2
code/config change: added scripts/diagnostics/inspect_policy_actions.py and extended probe_kbot_stability.py with explicit pose/action probes
why it was tried: determine whether V1 used a simple static standing action or a dynamic recovery sequence
result: V1 actions during the first 20 control steps were large and time-varying. Static fixed-action probes using representative V1 actions still fell, so copying one constant action is not enough.
decision: do not use V1 weights for V2 scratch; if a future action prior is used, it must be treated as a short dynamic bootstrap hint, not a deployed walking objective
```

```text
date: 2026-05-04
run: V2 scratch pose-bootstrap short branch
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_07-30-33
warm start: none
task id: Isaac-KBot-Forward-Flat-V2-Scratch-PoseBootstrap-v0
code/config change: used a hand-authored V1-derived reset pose, zero command, tight reset noise, 4 s episodes, and stand_joint_position_l2
why it was tried: test the user's earlier handcrafted-standing-position idea without loading V1 weights
result: rejected and stopped early; after the ankle targets were clamped to USD limits, the run still became a low-body timeout survivor worse than the V1-style bootstrap
decision: a static pose alone is not sufficient for this asset; the early controller/curriculum still matters
```

```text
date: 2026-05-04
run: V2 scratch V1-bootstrap exact episode-length reproduction
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_07-41-32/model_1298.pt
warm start: none
task id: Isaac-KBot-Forward-Flat-V2-Scratch-V1Bootstrap-v0
code/config change: corrected KBotForwardFlatV2ScratchV1BootstrapEnvCfg to match the original V1 scratch seed's 3 s episode length; kept timeout-only terminations, command x = 0.10-0.25 m/s, alive +5, base_height_l2 -15 at 0.78 m, low_body_l2 -30 below 0.45 m, knee_extension_l1 -80, and no V1 checkpoint loading
why it was tried: the first V1 run that eventually worked used 3 s episodes, not the 8 s episodes inherited by the first V2 bootstrap attempt. Internet/locomotion references also support staged curricula with fixed/simple early conditions, dense height/orientation shaping, and delayed complexity.
result: successful as a scratch anti-fall bootstrap, rejected as a walking gait. Training recovered from low-body collapse: low_body_l2 improved from about -30 at iteration 300 to about -0.08 at iteration 1298; base_height_l2 improved to about -0.11; flat_orientation_l2 to about -0.35; mean reward became positive. Headless evaluator root_height_p05_m = 0.762 and root_height gate passed.
evaluation result: REJECT as gait; speed_tracking_ratio 0.049, distance 0.308 m in 30 s, double_support_fraction 0.987, step_length_mean 0.034 m, yaw_drift_rad_per_m 0.802, lateral_drift_m_per_m 0.454, hip_roll_mean_abs_5cycle 0.282
decision: this is the new from-scratch V2 bootstrap base. Next stage should branch from this checkpoint only if accepted as a staged scratch curriculum, then add V2 gait rewards gradually. It is not yet a final V2 walking policy.
diagnostics: logs/rsl_rl/kbot_forward_flat/2026-05-04_07-41-32/diagnostics/model_1298_headless
headless video: logs/rsl_rl/kbot_forward_flat/2026-05-04_07-41-32/videos/play/trailing-side-hud-model_1298-headless-v2-scratch-bootstrap.mp4
playback note: future output videos should be 30 s and continuous, with no policy reset except root-height fall reset at z <= 0.0 m
30s continuous no-reset video: logs/rsl_rl/kbot_forward_flat/2026-05-04_07-41-32/videos/play/trailing-side-hud-model_1298-headless-v2-scratch-bootstrap-30s-continuous.mp4
30s continuous no-reset metrics: video_length_steps = 1500, fall_reset_count = 0, policy_reset_mode = fall_reset_only
```

```text
date: 2026-05-04
run: V2 scratch bootstrap continued into full V2 rewards with centered-foot posture fix
checkpoint: in progress, logs/rsl_rl/kbot_forward_flat/2026-05-04_15-59-46
warm start: staged scratch only; resumed from logs/rsl_rl/kbot_forward_flat/2026-05-04_07-41-32/model_1298.pt, not from the rejected model_1797 checkpoint
task id: Isaac-KBot-Forward-Flat-V2-v0
code/config change: added mirrored_joint_position_l2 for hip pitch/roll/yaw, knee, and ankle after mirror-sign normalization; re-enabled max foot lateral lane and per-side/max frontal-plane penalties; centered the scratch pose-bootstrap seed so it no longer starts from the asymmetric orange-frame posture
why it was tried: the 2026-05-04 orange screenshot showed both feet biased inward from the start, with the right foot substantially too far left. Continuing the rejected model_1797 would reinforce that habit, so this branch restarts the gait stage from the anti-fall bootstrap model_1298 with explicit foot centering and L/R joint symmetry pressure.
status: training started
```

```text
date: 2026-05-04
run: V2 centered-foot posture fix, reduced symmetry continuation
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_16-40-24/model_1797.pt
warm start: staged scratch only; resumed from logs/rsl_rl/kbot_forward_flat/2026-05-04_07-41-32/model_1298.pt
task id: Isaac-KBot-Forward-Flat-V2-v0
code/config change: mirrored_joint_position_l2 kept with evaluator-consistent signs, but weight reduced to -3.0 so the anti-fall bootstrap is not shocked into a low-body/roll failure
training result: recovered after the initial transition; final scalar reward about +21, timeout fraction 1.0, upright_alive about 7.99, low_body_l2 near zero, mirrored_joint_position_l2 about -0.90
diagnostic result: evaluator rejected model_1797; speed_tracking_ratio 1.134 and yaw_drift_rad_per_m -0.188 passed, but lateral_drift_m_per_m -0.697 failed, hip_roll_mean_abs_5cycle_rad 0.044 failed, airborne_fraction 0.023 failed, and full_support_fraction_left/right remained 0.0
decision: useful progression checkpoint, not keeper; continue from it to see if the improved speed/yaw can consolidate while symmetry and lane penalties keep improving posture.
diagnostics: logs/rsl_rl/kbot_forward_flat/2026-05-04_16-40-24/diagnostics/model_1797_headless
```

```text
date: 2026-05-04
run: V2 centered-foot posture fix, second reduced-symmetry continuation
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_16-46-50/model_2296.pt
warm start: continued from logs/rsl_rl/kbot_forward_flat/2026-05-04_16-40-24/model_1797.pt
task id: Isaac-KBot-Forward-Flat-V2-v0
training result: stable continuation; final scalar reward about +22, timeout fraction 1.0, track_lin_vel_xy_exp about 1.29, low_body_l2 small, mirrored_joint_position_l2 about -0.83, foot_lateral_lane_max_l1 about -0.096, max_leg_frontal_plane_l1 about -0.303
diagnostic result: evaluator rejected model_2296; speed_tracking_ratio 0.804 and yaw_drift_rad_per_m -0.246 passed, lateral_drift_m_per_m -0.585 failed, root_roll_mean_5cycle 0.030 failed, hip_roll_mean_abs_5cycle_rad 0.076 failed, full_support_fraction_left/right remained 0.0, edge_walk_proxy_fraction_left/right 0.973/0.975, and hip_pitch_mean_abs_error improved to 0.435 but is still too high
video: logs/rsl_rl/kbot_forward_flat/2026-05-04_16-46-50/videos/play/trailing-side-hud-model_2296-headless-v2-centered-30s-continuous.mp4
video metrics: fall_reset_count 0, policy_reset_mode fall_reset_only, speed_mean_mps 0.162, command_speed_mean_mps 0.201, hip_roll_yaw_window_mean_abs final 0.061
decision: better than the direct full-V2 branch for walking speed and yaw, but not acceptable. Next change should target lateral drift and roll/edge contact directly, not simply keep extending this reward stack unchanged.
diagnostics: logs/rsl_rl/kbot_forward_flat/2026-05-04_16-46-50/diagnostics/model_2296_headless
```

```text
date: 2026-05-04
run: V2 sole-center lane and flat-foot continuation from moving checkpoint
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_19-18-03/model_2795.pt
warm start: continued from logs/rsl_rl/kbot_forward_flat/2026-05-04_16-46-50/model_2296.pt
task id: Isaac-KBot-Forward-Flat-V2-v0
code/config change: added sole-center lane and hip-to-shin/sole frontal-column rewards using the heel/toe pad midpoint offsets, reduced foot_lateral_spacing_l1 from -5 to -2 so it cannot strongly pull feet together, weakened centered_joint_target_position_l2 from -2.0 to -0.5, and restored all-contact foot-flat pressure with foot_flat_l2 -1.0 and stance_foot_flat_l2 -6.0
diagnostic result: evaluator returned REVIEW_VIDEO. Speed, yaw, lateral drift, root roll, alternating steps, and root height passed. speed_tracking_ratio 1.183, yaw_drift_rad_per_m -0.016, lateral_drift_m_per_m -0.023, root_roll_mean_5cycle 0.0027, root_height_p05_m 0.780. Hip-roll mean still failed at 0.098 rad and airborne failed at 0.235. Edge-walk proxy improved materially versus model_2296: left/right 0.343/0.423 instead of 0.973/0.975, but full-support proxies remain 0.0 because the default asset still has only whole-foot contact bodies.
video: logs/rsl_rl/kbot_forward_flat/2026-05-04_19-18-03/videos/play/trailing-side-hud-model_2795-v2-sole-centered-flat-30s.mp4
video metrics: logs/rsl_rl/kbot_forward_flat/2026-05-04_19-18-03/videos/play/trailing-side-hud-model_2795-v2-sole-centered-flat-30s.json
decision: not approved, but this is the best V2 continuation so far for speed/yaw/lateral/root-roll while reducing edge walking. Next change should target hip-roll/airborne cadence without increasing standing/static posture pressure.
diagnostics: logs/rsl_rl/kbot_forward_flat/2026-05-04_19-18-03/diagnostics/model_2795_headless
```

```text
date: 2026-05-04
run: V2 extra continuation from model_2795 after orange3 review
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_20-45-25/model_3294.pt
warm start: continued from logs/rsl_rl/kbot_forward_flat/2026-05-04_19-18-03/model_2795.pt
task id: Isaac-KBot-Forward-Flat-V2-v0
why it was tried: orange3 showed the policy improving but still not vertically aligned through the hip/foot columns. Continue training to test whether the current sole-center/frontal-column rewards would keep improving without another config change.
diagnostic result: evaluator returned REVIEW_VIDEO. The continuation became faster and hip-roll improved slightly, but it did not replace model_2795. speed_tracking_ratio 1.345 versus 1.183, speed_mean_mps 0.271 versus 0.238, and hip_roll_mean_abs_5cycle_rad 0.095 versus 0.098. Regressions: lateral_drift_m_per_m -0.055 versus -0.023, root_roll_mean_5cycle -0.010 versus 0.0027, hip_yaw_mean_abs_5cycle_rad 0.014 versus 0.009, and right edge-walk proxy 0.438 versus 0.423. Airborne remains high at 0.228 and double support remains 0.0013.
video: logs/rsl_rl/kbot_forward_flat/2026-05-04_20-45-25/videos/play/trailing-side-hud-model_3294-v2-5cycle-sep-30s.mp4
video metrics: logs/rsl_rl/kbot_forward_flat/2026-05-04_20-45-25/videos/play/trailing-side-hud-model_3294-v2-5cycle-sep-30s.json
HUD change verified: the overlay no longer labels the main averages as a fixed 3.0 s window after gait is established. It uses the latest 5 full gait cycles, adds fixed-width L/R/full-cycle step time, step length, step rate, and adds sep = sole-center L/R y separation divided by the hip-roll body-origin y separation. leg0_shell and leg0_shell_2 are used as the hip-roll joint-axis proxy until exact joint-frame positions are exposed.
decision: keep model_2795 as the selected V2 checkpoint. model_3294 is useful evidence that simply continuing this reward stack tends to trade alignment for speed and yaw/roll drift. Next change should adjust reward conflict rather than only adding iterations.
diagnostics: logs/rsl_rl/kbot_forward_flat/2026-05-04_20-45-25/diagnostics/model_3294_headless
```

```text
date: 2026-05-04
run: V2 sole-center lane from anti-fall checkpoint, target pose too strong
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_19-07-44/model_1797.pt
warm start: continued from logs/rsl_rl/kbot_forward_flat/2026-05-04_07-41-32/model_1298.pt
task id: Isaac-KBot-Forward-Flat-V2-v0
code/config change: added sole-center lane/frontal-column rewards and centered_joint_target_position_l2 at -2.0
diagnostic result: evaluator returned REVIEW_VIDEO but the checkpoint is rejected as a walking policy. It fixed lateral drift but froze gait: speed_tracking_ratio 0.189, speed_mean_mps 0.038, double_support_fraction 0.971, step_count 12. Lateral drift passed at -0.041 and root height passed, but root roll and hip roll failed.
video: logs/rsl_rl/kbot_forward_flat/2026-05-04_19-07-44/videos/play/trailing-side-hud-model_1797-v2-sole-centered-30s.mp4
decision: useful negative result. Sole-center lanes are directionally useful, but a strong explicit centered-pose target from the anti-fall checkpoint suppresses gait. Resume from a moving checkpoint and keep the target-pose term weak.
diagnostics: logs/rsl_rl/kbot_forward_flat/2026-05-04_19-07-44/diagnostics/model_1797_headless
```

```text
date: 2026-05-05
run: V2 scratch reboot, anti-fall bootstrap with corrected hip-axis lane code staged for gait
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-05_23-43-35_v2_reboot_bootstrap_hip_axis_offsets/model_1299.pt
warm start: none
task id: Isaac-KBot-Forward-Flat-V2-Scratch-V1Bootstrap-v0
code/config change: kept the original scratch V1-bootstrap-style anti-fall curriculum; corrected hip-axis sole-lane targets are not active in this bootstrap task and are reserved for the following full V2 gait stage
training result: completed 1300 iterations cleanly. The policy recovered upright support similarly to the first V2 bootstrap: final timeout fraction 1.0, base_height_l2 about -0.13, low_body_l2 about -0.06, flat_orientation_l2 about -0.21, and mean reward about +11.7.
decision: usable staged scratch bootstrap only. Continue into full V2 gait rewards from this checkpoint; do not treat it as a walking policy.
```

```text
date: 2026-05-05
run: V2 reboot first gait-stage continuation with corrected hip-axis sole-lane targets
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-05_23-54-37_v2_reboot_gait_hip_axis_offsets_from_1299/model_1750.pt
warm start: staged scratch only; resumed from logs/rsl_rl/kbot_forward_flat/2026-05-05_23-43-35_v2_reboot_bootstrap_hip_axis_offsets/model_1299.pt
task id: Isaac-KBot-Forward-Flat-V2-v0
code/config change: foot_sole_lateral_lane_max_l1 now targets the actual hip-roll joint-axis lateral positions, left +0.15835 m and right -0.15805 m, using sole-center offsets; old body-origin lane and spacing terms are disabled so they no longer pull feet toward +/-0.12 m
training result: intended 500-iteration branch crashed at iteration 1773 with PPO value loss becoming NaN and torch rejecting a negative/invalid Normal distribution std. Latest clean checkpoint before the crash is model_1750.pt.
diagnostic result: evaluator rejected model_1750. It preserved root height but failed gait: speed_tracking_ratio 0.008, distance 0.118 m in 30 s, double_support_fraction 0.985, step_count 4, yaw_drift_rad_per_m -0.610, lateral_drift_m_per_m 0.999, edge_walk_proxy_fraction_left/right 0.992/0.991, hip_roll_mean_abs_5cycle_rad 0.086, hip_yaw_mean_abs_5cycle_rad 0.067.
video: logs/rsl_rl/kbot_forward_flat/2026-05-05_23-54-37_v2_reboot_gait_hip_axis_offsets_from_1299/videos/play/trailing-side-hud-model_1750-v2-reboot-hip-axis-offsets-30s.mp4
video metrics: logs/rsl_rl/kbot_forward_flat/2026-05-05_23-54-37_v2_reboot_gait_hip_axis_offsets_from_1299/videos/play/trailing-side-hud-model_1750-v2-reboot-hip-axis-offsets-30s.json
decision: reject. Correcting the lane target removes the known geometric error, but the full V2 gait-stage reward still shocks the scratch bootstrap into a mostly stationary edge-contact strategy and PPO instability. Next restart should keep the corrected offsets but soften the gait-stage transition or lower PPO/action noise before adding another 500-iteration branch.
```

```text
date: 2026-05-06
run: V2.1 scratch reboot after equalizing L/R frontal-plane weights
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-06_23-25-25_v2_1_bootstrap_equal_lr_frontal/model_1299.pt
warm start: none
task id: Isaac-KBot-Forward-Flat-V2-Scratch-V1Bootstrap-v0
code/config change: equalized full-V2 left_leg_frontal_plane_l1 and right_leg_frontal_plane_l1 from -3.0/-5.0 to -4.0/-4.0; the scratch bootstrap task still keeps these terms disabled
why it was tried: orange5 suggested a persistent lean, and the unequal per-side full-V2 frontal-plane weights were a likely reward asymmetry
training result: completed 1300 iterations cleanly. The run reproduced the successful V1-style anti-fall bootstrap pattern: early low-body timeout survival recovered by the end, with timeout fraction 1.0, base_height_l2 about -0.36, low_body_l2 about -0.20, and mean reward about +9.0 at iteration 1299.
geometry check: one-step V2 probe confirmed foot/body order is sane: foot1 is left/positive-y, foot3 is right/negative-y, leg0_shell/leg0_shell_2 are the hip proxy pair, and leg3_shell1/leg3_shell11 are the lower-leg proxy pair. The full V2 reward table confirmed left/right frontal-plane weights are both -4.0.
decision: usable staged scratch bootstrap only. Continue into full V2 gait rewards to test whether the equalized per-side frontal-plane pressure changes the lean/gait failure.
```

```text
date: 2026-05-06
run: V2.1 gait continuation from equal-L/R scratch bootstrap
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-06_23-36-57_v2_1_gait_equal_lr_from_1299/model_1798.pt
warm start: staged scratch only; resumed from logs/rsl_rl/kbot_forward_flat/2026-05-06_23-25-25_v2_1_bootstrap_equal_lr_frontal/model_1299.pt
task id: Isaac-KBot-Forward-Flat-V2-v0
code/config change: full V2 gait stage with equalized left/right frontal-plane weights active at -4.0/-4.0
training result: completed 500 continuation iterations without the PPO NaN seen in the previous reboot branch. The transition still shocked the bootstrap into poor gait/posture terms early, then recovered root height and timeout survival. Final training scalars included low_body_l2 about 0.0, upright_alive about 8.0, but yaw/heading and action-rate penalties remained large.
diagnostic result: evaluator rejected model_1798. It stayed upright but barely moved: speed_tracking_ratio 0.029, speed_mean_mps 0.0058 versus command_speed_mean_mps 0.201, yaw_drift_rad_per_m -0.796, lateral_drift_m_per_m -0.274, double_support_fraction 0.979, step_count 10, hip_roll_mean_abs_5cycle_rad 0.052, edge_walk_proxy_fraction_left/right 0.987/0.986, root_height_p05_m 0.789.
decision: reject as a gait policy. Equalizing the per-side frontal-plane weights removed a real suspicious asymmetry and avoided PPO NaN in this branch, but it did not solve the main failure. The current full-V2 gait stage still produces a mostly stationary double-support edge-contact strategy. Next useful change should soften or split the gait transition rather than keep extending the same full reward stack.
diagnostics: logs/rsl_rl/kbot_forward_flat/2026-05-06_23-36-57_v2_1_gait_equal_lr_from_1299/diagnostics/model_1798
```

```text
date: 2026-05-06
run: V2.1 full-V2 training from policy iteration zero with equal L/R frontal-plane weights
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-06_23-49-21_v2_1_full_v2_from_zero_equal_lr/model_350.pt
warm start: none
task id: Isaac-KBot-Forward-Flat-V2-v0
code/config change: same equalized full-V2 left/right frontal-plane weights, no resume, no scratch bootstrap checkpoint
why it was tried: user requested a true policy restart from learning iteration 0 rather than a continuation from the anti-fall bootstrap
training result: failed. The run started at Learning iteration 0/1000 and did not load a checkpoint. It quickly reached 400-step timeout episodes while staying in a low/collapsed posture. By iteration 349-363, low_body_l2 was still about -130 to -117 and base_height_l2 about -51 to -47. PPO value loss exploded, became NaN at iteration 363, and training crashed with torch rejecting an invalid Normal distribution std. Latest saved checkpoint before crash is model_350.pt.
decision: reject and do not continue this branch. Equalizing L/R frontal-plane weights does not make direct full-V2-from-zero training viable. This reinforces the staged curriculum lesson: the anti-fall bootstrap is not optional unless the full-V2 reward/noise/termination setup is redesigned.
```

```text
date: 2026-05-04
run: V2 centered-foot posture fix, corrected hip-pitch sign too strong
checkpoint: stopped early, logs/rsl_rl/kbot_forward_flat/2026-05-04_16-37-58
warm start: staged scratch only; resumed from logs/rsl_rl/kbot_forward_flat/2026-05-04_07-41-32/model_1298.pt
task id: Isaac-KBot-Forward-Flat-V2-v0
code/config change: corrected hip_pitch mirror_sign to +1.0 and corrected the pose-bootstrap hip-pitch seed
result: stopped early; mirrored_joint_position_l2 at weight -18 shocked the anti-fall bootstrap, quickly driving mirrored-joint penalty into roughly -60, low_body_l2 into large negative values, and reward near -900 despite timeout survival
decision: keep the correct hip-pitch sign, but reduce mirrored_joint_position_l2 to -3.0 so it acts as a regularizer while foot_lateral_lane_max_l1 and per-side/max frontal-plane terms do the direct centering.
```

```text
date: 2026-05-04
run: V2 centered-foot posture fix, first attempt rejected
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_15-59-46/model_1797.pt
warm start: staged scratch only; resumed from logs/rsl_rl/kbot_forward_flat/2026-05-04_07-41-32/model_1298.pt
task id: Isaac-KBot-Forward-Flat-V2-v0
diagnostic result: evaluator rejected model_1797; speed_tracking_ratio 0.097, yaw_drift_rad_per_m 0.409, hip_roll_mean_abs_5cycle_rad 0.065, double_support_fraction 0.977, step_count 11, step_length_mean_m 0.041, root height passed, lateral drift passed
important correction: this attempt used the wrong mirror sign for hip_pitch in mirrored_joint_position_l2. The evaluator's established sign convention is hip_pitch +1.0, hip_roll/hip_yaw/knee/ankle -1.0, so the branch is not a valid keeper. The centered pose-bootstrap seed was also corrected to use matching left/right hip-pitch signs.
decision: reject and restart the gait-stage continuation from model_1298 after fixing hip_pitch mirror_sign to +1.0 and the pose-bootstrap hip-pitch sign.
diagnostics: logs/rsl_rl/kbot_forward_flat/2026-05-04_15-59-46/diagnostics/model_1797_headless
```

```text
date: 2026-05-04
run: V2 scratch bootstrap continued into full V2 rewards
checkpoint: logs/rsl_rl/kbot_forward_flat/2026-05-04_14-20-49/model_1797.pt
warm start: staged scratch only; resumed from logs/rsl_rl/kbot_forward_flat/2026-05-04_07-41-32/model_1298.pt, which itself was trained from scratch without any V1 checkpoint
task id: Isaac-KBot-Forward-Flat-V2-v0
why it was tried: test whether the scratch anti-fall bootstrap can survive the full V2 indicator/reward set without immediately collapsing.
training result: survived timeout-only episodes and recovered after an initially bad transition. Final training scalar signs were standing-stable: mean episode length 400, time_out 1.0, track_lin_vel_xy_exp 1.053, base_height_l2 -0.583, low_body_l2 -0.216, upright_alive 7.952. Yaw/heading and gait terms remained weak.
evaluation result: REJECT as gait. Root height passed, yaw drift passed, and roll was small, but speed_tracking_ratio = 0.010, distance = 0.152 m in 30 s, double_support_fraction = 0.994, step_count = 2, step_length_mean = -0.019 m, lateral_drift_m_per_m = -0.524, hip_roll_mean_abs_5cycle = 0.059, edge_walk proxies were about 0.996 on both sides.
decision: useful as evidence that the bootstrap can survive full V2 posture pressure, but not useful as a walking checkpoint. The next branch should not add more standing/posture pressure; it should add a gentler gait-progression stage that rewards real root advance, alternating heel-strike events, and longer step length before returning to the full V2 reward set.
diagnostics: logs/rsl_rl/kbot_forward_flat/2026-05-04_14-20-49/diagnostics/model_1797_headless
30s continuous no-reset video: logs/rsl_rl/kbot_forward_flat/2026-05-04_14-20-49/videos/play/trailing-side-hud-model_1797-headless-v2-scratch-to-full-v2-30s-continuous.mp4
30s continuous no-reset metrics: video_length_steps = 1500, fall_reset_count = 0, policy_reset_mode = fall_reset_only
```

## 9. Checkpoint Comparisons

Use fixed evaluations. Do not compare checkpoints using training scalar reward alone.

### Comparison Template

```text
old checkpoint:
new checkpoint:
eval duration:
survival:
speed tracking:
yaw/lateral drift:
roll bias:
hip roll/yaw bias:
contact quality:
L/R symmetry:
crouch/knee posture:
step length/cadence:
decision:
reason:
```

## 10. Lessons Learned During V2

Add notes immediately when they become clear.

```text
lesson: Contact-quality measurement should start as diagnostics, not extra reward pressure.
evidence: V1 became hard to reason about because many overlapping reward terms pushed on related gait properties. V2 documents explicitly call for more indicators and fewer rewards.
what changed because of it: Sole normal, toe/heel-down, and inner/outer-edge proxies were added to the evaluator scorecard without adding new reward terms.
```

```text
lesson: Generated heel/toe pads are useful for investigation but too expensive to make the default path right now.
evidence: The generated pad asset could expose separate bodies, but validation and integration consumed multiple days and still left ambiguity around heel-only behavior.
what changed because of it: KBotForwardFlatV2EnvCfg now uses KBOT_CFG again. Pad assets/scripts remain historical tools, while V2 diagnostics use foot1/foot3 contact plus sole-plane vectors.
```

```text
lesson: V1 escaped the start-fall problem through a short-episode scratch curriculum, not through hard fall terminations.
evidence: The original V1 seed used 3 s episodes, low-speed commands, timeout-only terminations, alive/base-height/low-body/knee shaping, and enough iterations. Reproducing the reward weights with 8 s episodes stayed collapsed; changing the V2 scratch bootstrap to 3 s episodes recovered root height by iteration 1298.
what changed because of it: KBotForwardFlatV2ScratchV1BootstrapEnvCfg now explicitly sets episode_length_s = 3.0 and is recorded as the scratch anti-fall bootstrap base.
```

```text
lesson: Standing up is not the same as walking.
evidence: model_1298 passes root height after the exact V1-bootstrap reproduction, but evaluator rejects it for near-zero speed tracking, excessive double support, side drift, roll/hip-roll bias, and poor sole-plane contact quality.
what changed because of it: Treat model_1298 as a staged bootstrap checkpoint only. The next branch should add gait progression and V2 posture indicators gradually instead of calling this a walking policy.
```

```text
lesson: Full V2 rewards can preserve standing but still suppress gait from the scratch bootstrap.
evidence: Continuing model_1298 into Isaac-KBot-Forward-Flat-V2-v0 reached stable root height and timeout-only episodes, but model_1797 still had speed_tracking_ratio 0.010, double_support_fraction 0.994, and only two step events in 30 s.
what changed because of it: Do not jump directly from anti-fall bootstrap to full V2 as the main path. Insert a gait-progression stage focused on root advance, alternating heel-strike events, and practical step length before applying the full posture/alignment stack.
```

## 11. Open Questions

- Where did the current actuator parameters originate: Robstride datasheet, KBot repo, hand tuning, or earlier manual guess?
- Is the current initial pose mechanically appropriate, or only a pragmatic standing pose from early fall debugging?
- Should heel/toe/edge contact bodies or sensors be added to the asset before reward tuning?
  - Current status: not for the next iteration. Use whole-foot contact plus sole-plane vector diagnostics first. Revisit CAD/Blender five-piece soles only if vector indicators cannot distinguish a repeated failure mode.
- What mirrored sign map should be used for left/right joint symmetry?
- What thresholds should define excessive crouch for this simplified body?
- What is the target step length at each command speed?
