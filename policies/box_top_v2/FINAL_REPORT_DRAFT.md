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

2026-05-08 pose-bootstrap asset finding:

```text
The handcrafted standing pose now has a headless Isaac Lab zero-action validation path.
The pose stands only when the task does not override the KBot articulation root props.
The removed override was ArticulationRootPropertiesCfg(enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=2).
Standalone probes showed the rigid-body props alone were not the issue; the articulation root props reproduced the fall.
Pose-bootstrap uses the raw-USD settled joint pose, root z = 0.8565, settled base-height target = 0.856, and scaled implicit actuator gains.
Validation: Isaac-KBot-Forward-Flat-V2-Scratch-PoseBootstrap-v0, default asset, zero action/no policy, 1000 env steps = 20 s sim time, min_z = 0.8559, final_z = 0.8565, max_abs_gravity_xy about 0.0739.
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

The active weights are stage-dependent. Do not treat one printed table as the
global V2 equation unless it comes from the exact run's `params/env.yaml` or
resolved reward-manager printout. The reward component graphs generated by
`scripts/diagnostics/plot_reward_components.py` use TensorBoard
`Episode_Reward/*` scalars and therefore show the weighted contribution that
actually reached PPO for that run.

For V2.5 graph review, the generated files are:

```text
<run>/graficos/<timestamp>_combined_reward_components.png
<run>/graficos/<timestamp>_reward_components_legend.png
<run>/metricas/<timestamp>_reward_components.csv
```

### V2/V2.5 Reward Term Glossary

This glossary covers the 59 reward component names seen in the current V2.5
graphs. Terms may be present with weight `0.0`, inactive because their condition
never fires, or inherited from a parent config. A zero plotted contribution does
not always mean the implementation is wrong; it can also mean the current
checkpoint never enters the state where the term applies.

```text
action_rate_l2
  Source: Isaac Lab base reward.
  Meaning: Penalizes squared changes between consecutive policy actions.
  Intent: Reduce twitchy actuator targets and high-frequency control chatter.

alive
  Source: Isaac Lab base reward.
  Meaning: Constant positive reward while the episode is not terminated.
  Intent: Keep early training from immediately preferring termination.

alternating_foot_phase
  Source: custom KBot term, mdp.alternating_foot_phase_reward.
  Meaning: Rewards scheduled left-only/right-only support over a nominal period; gives partial credit for double support and penalizes airborne state.
  Intent: Light gait rhythm hint, not a hard step-length guarantee.

ang_vel_xy_l2
  Source: Isaac Lab base reward.
  Meaning: Penalizes squared root roll/pitch angular velocity.
  Intent: Reduce tumbling and violent torso rotation.

backward_velocity_l2
  Source: custom KBot term, mdp.backward_velocity_l2.
  Meaning: Penalizes negative body-frame forward velocity.
  Intent: Stop backward motion while still allowing forward-only low-speed curricula.

base_height_l2
  Source: Isaac Lab base reward.
  Meaning: Penalizes squared root-height error from a configured target.
  Intent: Keep the posed-start policy tall.

centered_joint_target_position_l2
  Source: custom KBot term, mdp.joint_target_position_l2.
  Meaning: Penalizes deviation from explicit joint targets supplied by config.
  Intent: Optional posture centering. In recent V2.5 graphs this is present but does no work.

contact_chatter_l1
  Source: custom KBot term, mdp.contact_chatter_l1.
  Meaning: On touchdown, penalizes feet whose preceding air time was shorter than a configured minimum.
  Intent: Suppress rapid contact flipping / shuffling.

dense_single_support_step_progress
  Source: custom KBot term, mdp.dense_single_support_step_progress_reward.
  Meaning: During single support, rewards clipped root +X progress since the last touchdown, normalized by command speed and target cycle frequency.
  Formula shape: reward = progress * is_single_support * is_moving.
  Intent: Dense helper for step progress when completed-step gates are too sparse.

dof_acc_l2
  Source: Isaac Lab base reward.
  Meaning: Penalizes squared joint acceleration.
  Intent: Smooth joint motion and reduce vibration.

dof_pos_limits
  Source: Isaac Lab base reward.
  Meaning: Penalizes joint positions near or beyond configured limits.
  Intent: Avoid limit-hitting gaits.

dof_torques_l2
  Source: Isaac Lab base reward.
  Meaning: Penalizes squared actuator torque.
  Intent: Reduce wasteful or saturating actuation.

feet_air_time
  Source: Isaac Lab base reward.
  Meaning: Rewards foot air time on `foot1` and `foot3` contacts above a threshold.
  Intent: Encourage stepping. Risk: can reward light/tiptoe contact if overweighted.

flat_orientation_l2
  Source: Isaac Lab base reward.
  Meaning: Penalizes root tilt using projected gravity.
  Intent: Keep torso/root upright.

foot_flat_l2
  Source: custom KBot term, mdp.foot_flat_l2.
  Meaning: Penalizes both feet when their local up vector is not close to world +Z.
  Intent: Tiptoe/edge-contact proxy. It is not a true contact-patch-area reward.

foot_lateral_lane_l1
  Source: custom KBot term, mdp.foot_lateral_lane_l1.
  Meaning: Penalizes summed left/right foot body-frame lateral lane error after tolerance.
  Intent: Keep feet in side lanes and avoid collapse toward center.

foot_lateral_lane_max_l1
  Source: custom KBot term, mdp.foot_lateral_lane_max_l1.
  Meaning: Penalizes the worst single foot's lateral lane error after tolerance.
  Intent: Prevent one foot from hiding a large lane error behind the other foot.

foot_lateral_spacing_l1
  Source: custom KBot term, mdp.foot_lateral_spacing_l1.
  Meaning: Penalizes absolute error from target left-right foot spacing in the root frame.
  Intent: Preserve support width. V2.5 target width is around 0.3164 m in the posed-start work.

foot_parallel_l2
  Source: custom KBot term, mdp.foot_parallel_l2.
  Meaning: Penalizes feet whose horizontal forward axes are not parallel to root forward.
  Intent: Reduce foot yaw relative to the robot.

foot_sagittal_separation_l1
  Source: custom KBot term, mdp.foot_sagittal_separation_l1.
  Meaning: During single stance, penalizes fore-aft foot separation below target.
  Intent: Discourage in-place stepping and short fore-aft stance geometry.

foot_signed_lateral_clearance_l1
  Source: custom KBot term, mdp.foot_signed_lateral_clearance_l1.
  Meaning: Penalizes signed left-right foot separation below a minimum.
  Intent: Prevent foot crossing by preserving left/right ordering.

foot_sole_lateral_lane_max_l1
  Source: custom KBot term, mdp.foot_sole_lateral_lane_max_l1.
  Meaning: Like `foot_lateral_lane_max_l1`, but uses authored local sole-center offsets instead of foot body origins.
  Intent: Lane control using the actual sole proxy rather than the link origin.

foot_toe_in_l2
  Source: custom KBot term, mdp.foot_toe_in_l2.
  Meaning: Penalizes toes yawing inward toward the centerline after tolerance.
  Intent: Avoid toe-in gaits and self-crossing tendencies.

foot_world_parallel_l2
  Source: custom KBot term, mdp.foot_world_parallel_l2.
  Meaning: Penalizes both feet yawing away from world +X.
  Intent: Optional absolute foot heading pressure. Often disabled or zero because root-relative terms are preferred.

foot_world_parallel_max_l2
  Source: custom KBot term, mdp.foot_world_parallel_max_l2.
  Meaning: Worst-foot version of `foot_world_parallel_l2`.
  Intent: Optional absolute foot heading pressure for the worst offending foot.

forward_velocity_below_l2
  Source: custom KBot term, mdp.forward_velocity_below_l2.
  Meaning: Penalizes body-frame forward velocity below a configured minimum.
  Intent: Prevent standing still or backward shuffling when forward walking is expected.

hip_roll_position_ema_5cycle_l2
  Source: custom KBot term, mdp.joint_position_ema_l2.
  Meaning: EMA penalty on selected hip-roll joint offsets over a longer gait-scale window.
  Intent: Penalize persistent hip-roll bias, not ordinary step-cycle motion.

hip_roll_yaw_position_ema_l2
  Source: custom KBot term, mdp.joint_position_ema_l2.
  Meaning: EMA penalty on hip roll/yaw joint offsets from default.
  Intent: Remove persistent hip roll/yaw bias without overreacting to instant motion.

hip_roll_yaw_position_l2
  Source: custom KBot term, mdp.joint_position_l2.
  Meaning: Instant squared hip roll/yaw joint offset from default.
  Intent: Keep hip roll/yaw from solving gait by sustained side bias or yawed hips.

knee_extension_l1
  Source: custom KBot term, mdp.knee_extension_l1.
  Meaning: Penalizes knees whose bend magnitude is below a minimum.
  Intent: Avoid mechanically locked, straight-knee gaits.

lateral_velocity_l2
  Source: custom KBot term, mdp.lateral_velocity_l2.
  Meaning: Penalizes body-frame lateral root velocity.
  Intent: Discourage sideways walking.

left_leg_frontal_plane_l1
  Source: custom KBot term, mdp.leg_frontal_plane_side_l1.
  Meaning: Left shin/foot lateral deviation from the left hip-centered sagittal lane.
  Intent: Side-specific frontal-plane cleanup.

leg_frontal_plane_l1
  Source: custom KBot term, mdp.leg_frontal_plane_l1.
  Meaning: Sum of both legs' shin/foot lateral deviations from hip-centered sagittal lanes.
  Intent: Keep leg columns from leaning inward/outward in the frontal plane.

leg_frontal_sole_plane_max_l1
  Source: custom KBot term, mdp.leg_frontal_sole_plane_max_l1.
  Meaning: Worst hip-to-shin/sole lateral column error using sole-center offsets.
  Intent: Frontal column cleanup using sole proxy positions.

lin_vel_z_l2
  Source: Isaac Lab base reward.
  Meaning: Penalizes squared vertical root velocity.
  Intent: Discourage hopping and bouncing.

low_body_l2
  Source: custom KBot term, mdp.root_height_below_l2.
  Meaning: Penalizes squared shortfall below a configured minimum root height.
  Intent: Soft anti-crouch / anti-collapse pressure.

max_leg_frontal_plane_l1
  Source: custom KBot term, mdp.leg_frontal_plane_max_l1.
  Meaning: Worst individual shin/foot lateral deviation from its hip-centered sagittal lane.
  Intent: Prevent one bad segment from being averaged away.

mirrored_joint_position_l2
  Source: custom KBot term, mdp.mirrored_joint_position_l2.
  Meaning: Penalizes left/right joint asymmetry after configured mirror-sign normalization.
  Intent: Keep bilateral posture symmetric where symmetry is expected.

right_leg_frontal_plane_l1
  Source: custom KBot term, mdp.leg_frontal_plane_side_l1.
  Meaning: Right shin/foot lateral deviation from the right hip-centered sagittal lane.
  Intent: Side-specific frontal-plane cleanup.

root_lateral_position_l2
  Source: custom KBot term, mdp.root_lateral_position_l2.
  Meaning: Penalizes squared world/root lateral displacement from the environment origin.
  Intent: Encourage returning to y = 0 instead of merely reducing lateral velocity.

root_lateral_tilt_ema_l2
  Source: custom KBot term, mdp.root_lateral_tilt_ema_l2.
  Meaning: EMA of root side lean, squared.
  Intent: Penalize persistent torso roll bias.

root_lateral_tilt_l2
  Source: custom KBot term, mdp.root_lateral_tilt_l2.
  Meaning: Penalizes squared side component of projected gravity in body frame.
  Intent: Reduce instantaneous lateral torso/root lean.

stance_foot_flat_l2
  Source: custom KBot term, mdp.stance_foot_flat_l2 or mdp.single_stance_foot_flat_l2 depending on config.
  Meaning: Foot flatness proxy applied to contacting feet or to the single support foot.
  Intent: Reduce tiptoe stance while avoiding swing-foot overconstraint.

stand_joint_position_l2
  Source: custom KBot term, mdp.joint_position_l2.
  Meaning: Penalizes selected joints moving away from their default/reset pose.
  Intent: Preserve the authored posed-start posture where useful.

step_advance_margin
  Source: custom KBot term, mdp.step_advance_margin_reward.
  Meaning: Signed completed-step reward on alternating touchdowns. Steps below `min_step_advance` go negative; steps above the minimum and long enough in time go positive toward a command-scaled target.
  Intent: Make too-short steps visible to PPO while rewarding real root advance.

supported_forward_velocity
  Source: custom KBot term, mdp.supported_forward_velocity_reward.
  Meaning: Rewards world +X velocity when support is not constant double stance or flight; single support gets full credit, double support partial credit, airborne no credit.
  Intent: Prefer supported walking advance over flight or static shuffling.

swing_foot_overtake_l1
  Source: custom KBot term, mdp.swing_foot_overtake_l1.
  Meaning: Penalizes a swing foot that fails to pass the stance foot by an air-time-scaled required lead.
  Intent: Encourage real stepping instead of foot tapping.

termination_penalty
  Source: Isaac Lab base reward.
  Meaning: Large penalty on non-timeout termination.
  Intent: Make falls/collapses expensive, while noting that overly aggressive reset can hide on-ground negative reward.

track_ang_vel_z_exp
  Source: Isaac Lab base reward.
  Meaning: Exponential reward for matching commanded yaw rate.
  Intent: Track turning command; in straight-walk stages command yaw is zero.

track_lin_vel_xy_exp
  Source: Isaac Lab base reward.
  Meaning: Exponential reward for matching commanded horizontal velocity.
  Intent: Main velocity tracking signal. It can be gamed by shuffle if step/support gates are weak.

undesired_contacts
  Source: Isaac Lab base reward.
  Meaning: Penalizes contact on non-foot bodies configured as undesired.
  Intent: Reject falls, dragging, and body/leg-shell ground contact.

upright_alive
  Source: custom KBot term, mdp.upright_alive.
  Meaning: Positive reward only when root height is above a threshold and tilt is below a threshold.
  Intent: Reward tall, usable survival rather than any non-terminated state.

valid_step_root_advance
  Source: custom KBot term, mdp.valid_step_root_advance_reward.
  Meaning: Positive-only completed-step reward on alternating touchdowns that pass root-advance and air-time gates.
  Intent: Strict valid-step reward. In recent V2.5 graphs this often stays zero because the gate is too sparse for shuffle policies.

walking_cycle_cadence_above_l2
  Source: custom KBot term, mdp.walking_cycle_cadence_above_l2.
  Meaning: Penalizes same-foot cycle cadence above a configured walking ceiling.
  Intent: Suppress high-frequency shuffle cadence. This measures per-foot cycle cadence, not per-step alternation frequency.

wobble_joint_vel_l2
  Source: custom KBot term, mdp.joint_velocity_l2.
  Meaning: Penalizes selected joint velocities, typically hip yaw/roll.
  Intent: Reduce visible wobble without globally overdamping every joint.

world_forward_velocity_below_l2
  Source: custom KBot term, mdp.world_forward_velocity_below_l2.
  Meaning: Penalizes world +X velocity below a configured minimum.
  Intent: Ensure actual path progress, not just body-frame motion while yawed.

world_forward_velocity_clip
  Source: custom KBot term, mdp.world_forward_velocity_clip.
  Meaning: Rewards real world +X velocity clipped above a configured maximum.
  Intent: Encourage forward progress without pushing speed beyond starter range.

world_heading_l2
  Source: custom KBot term, mdp.world_heading_l2.
  Meaning: Penalizes root forward vector deviating from world +X and facing backward.
  Intent: Keep the robot walking straight down the intended path.

yaw_rate_l2
  Source: custom KBot term, mdp.yaw_rate_l2.
  Meaning: Penalizes squared body-frame yaw rate.
  Intent: Reduce turning/spinning in straight-walk stages.
```

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

### V2 Final Conclusions And V3 Handoff

The best V2.5 result is not a final walking controller. The useful checkpoint
is `model_648.pt` from:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-09_00-16-45_v2_5_pose_gait_quality_from_v2_5_349_fsep_ksep
```

That checkpoint proved the posed-start reset can train into a tall, straight,
support-width-preserving forward policy. It improved over the raw pose and over
the first V2.5 seed in distance, speed tracking, lateral drift, `fsep`, `ksep`,
double support, edge-walk proxies, and stance slip. It should be preserved as
the conservative V2.5 evidence checkpoint.

The reason V2.5 should stop as a curriculum branch is that the later training
objective was trying to solve too many things by continuation from a local
minimum. The policy learned a high-frequency micro-step / contact-chatter
solution. Later S4 attempts correctly diagnosed the exploit, but their reward
patches either:

```text
kept the old high-cadence shuffle,
made approved-step gates visible while losing forward root advance,
lowered cadence by suppressing real movement,
or changed the live task enough that model_648 was no longer being continued under its original pipeline.
```

The important V2 lesson is therefore not "keep training 648 under a drifting
task." It is:

```text
keep the 648 reset/task topology,
keep the 648 useful geometry pressures,
do not silently add a new gate stack to the same task ID,
restart from iteration 0 before changing the reward balance materially.
```

V3 is the parallel restart for that conclusion. It does not replace V2.5 and
does not modify the frozen `648Compat` task. Its first task ID is:

```text
Isaac-KBot-Forward-Flat-V3-648HandTuned-v0
```

V3 inherits the frozen `model_648` reward topology and keeps an explicit neutral
weight block whose values match the saved `model_648` run. It deliberately
excludes the later S4 `valid_step_root_advance`,
`step_advance_margin`, `dense_single_support_step_progress`,
`walking_cycle_cadence_above_l2`, and `contact_chatter_l1` terms. The first V3
question is whether the exact 648-compatible recipe can train from iteration 0
into a better non-shuffle policy; hand tuning can then happen by editing that
visible V3 block without modifying the V2.5 compatibility task.

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

```text
lesson: The old lateral foot rewards were geometrically too narrow for the current hip axes.
evidence: The inherited foot_lateral_spacing_l1 and foot_lateral_lane_l1 targets pulled toward about 0.24 m separation: target_width = 0.24 and target_left_y/right_y = +/-0.12. The hip-axis reference is about 2 * 0.1582 = 0.3164 m. A width-fixed branch, model_3349, moved the HUD/body-frame sep to mean 0.296 m, p95 0.319 m, and final 0.308 m, while the old step-length branch model_3399 stayed near mean sep 0.239 m because it still inherited the old narrow lateral targets.
what changed because of it: V2.3 should use the hip-axis lateral reference from the start. Use target_width = 0.3164, target_left_y = +0.1582, target_right_y = -0.1582, and keep minimum_width as a crossing/collapse floor rather than the main attractor.
```

```text
lesson: Correcting sep alone is not enough; the policy can buy wider feet with hip roll and lateral drift.
evidence: model_3349_width improved speed_tracking_ratio to 0.938 and sep to near the hip-axis target, but lateral_drift_m_per_m worsened to -0.345 and hip_roll_mean_abs_5cycle rose to 0.078 rad. The follow-up hip-axis-width stabilization branch, model_3299, reduced hip roll to 0.044 rad but still had lateral_drift_m_per_m -0.365 and did not improve step length.
what changed because of it: Do not continue either width-fix branch as a keeper lineage. Restart V2.3 from iteration zero using the known staged bootstrap, with a low-weight sep fix present early and stronger root/hip/yaw constraints introduced gradually after the anti-fall seed exists.
```

```text
lesson: orange9 confirms the width fix and the new failure mode visually.
evidence: screenshots/orange9.jpg shows HUD sep about 0.30 m and height about 0.79 m, matching the corrected hip-axis target and root-height metrics. The same frame shows hipR average about 0.071 and visible lateral leg/body offset, consistent with the model_3349 diagnostic rejection.
what changed because of it: Treat sep as fixed in the reward geometry, but gate/shape lateral drift and hip roll more carefully in the staged V2.3 continuation.
```

## 10.1 V2.2 Late Probe Summary

```text
date: 2026-05-08
run: logs/rsl_rl/kbot_forward_flat/2026-05-08_02-16-23_v2_2_step_length_cleanup_from_3200
checkpoint: model_3399.pt
warm start: policy-only resume from model_3200.pt
task id: Isaac-KBot-Forward-Flat-V2-StepLengthCleanup-v0
why it was tried: lengthen tiny rapid steps while preserving the current straight/upright gait.
result: REJECT/REVIEW_VIDEO quality, worse than model_3200 on the important metrics. speed_tracking_ratio 0.700, step_length_mean_m 0.011, cycle_length_mean_m 0.023, hip_roll_mean_abs_5cycle_rad 0.048.
decision: do not continue. This branch still inherited the old narrow lateral target, so it did not address the sep geometry problem.
```

```text
date: 2026-05-08
run: logs/rsl_rl/kbot_forward_flat/2026-05-08_02-26-16_v2_2_cadence_length_cleanup_from_3200
checkpoint: model_3349.pt
warm start: policy-only resume from model_3200.pt
task id: Isaac-KBot-Forward-Flat-V2-CadenceLengthCleanup-v0
why it was tried: correct lateral sep to hip-axis width and slow the rapid shuffle.
result: evaluator REJECT. sep geometry improved: HUD/body-frame sep mean 0.296 m, p95 0.319 m, final 0.308 m against target 0.3164 m. But lateral drift and hip roll worsened: lateral_drift_m_per_m -0.345, hip_roll_mean_abs_5cycle_rad 0.078.
decision: confirms the sep target fix is correct, but not a keeper policy.
video: logs/rsl_rl/kbot_forward_flat/2026-05-08_02-26-16_v2_2_cadence_length_cleanup_from_3200/videos/play/trailing-hud-model_3349-v2_2-cadence-length-widthfix.mp4
image: screenshots/orange9.jpg
```

```text
date: 2026-05-08
run: logs/rsl_rl/kbot_forward_flat/2026-05-08_02-30-58_v2_2_hip_axis_width_cleanup_from_3200
checkpoint: model_3299.pt
warm start: policy-only resume from model_3200.pt
task id: Isaac-KBot-Forward-Flat-V2-HipAxisWidthCleanup-v0
why it was tried: isolate the hip-axis width target while restoring stronger root/yaw/hip stabilization.
result: evaluator REJECT. speed_tracking_ratio 0.917 and hip_roll_mean_abs_5cycle_rad 0.044, but lateral_drift_m_per_m remained bad at -0.365 and step_length_mean_m stayed about 0.013.
decision: do not continue. The policy needs the corrected sep target during staged scratch training, not as a late correction bolted onto model_3200.
```

## 10.2 V2.3 Restart Plan

```text
policy label: V2.3
start: true iteration zero
main lesson carried forward: use the bootstrap report's staged recipe; do not start full V2 from zero
new correction carried forward: hip-axis lateral sep target from the first stage
stage 1 task: Isaac-KBot-Forward-Flat-V2_3-Scratch-V1Bootstrap-v0
stage 1 run name: v2_3_bootstrap_from_zero_sepfix
stage 1 purpose: reproduce the reliable 3 s anti-fall bootstrap while preventing the old 0.24 m foot-separation attractor from entering the lineage
stage 1 sep settings: target_width = 0.3164, target_left_y = +0.1582, target_right_y = -0.1582, minimum_width = 0.24
stage 1 caution: sep terms must stay weak in the anti-fall phase; this stage is still an anti-fall seed, not walking
stage 2 direction: continue only after anti-fall succeeds, using a gentle gait transition with the same hip-axis sep reference
```

## 10.3 V2.3 Restart Result

```text
date: 2026-05-08
run: logs/rsl_rl/kbot_forward_flat/2026-05-08_03-41-49_v2_3_bootstrap_from_zero_sepfix
checkpoint: model_1299.pt
task id: Isaac-KBot-Forward-Flat-V2_3-Scratch-V1Bootstrap-v0
start: true policy iteration zero, no checkpoint resume
why it was tried: restart from the bootstrap report's known 3 s anti-fall recipe while carrying the corrected hip-axis sep target from the first policy update.
sep fix: target_width = 0.3164, target_left_y = +0.1582, target_right_y = -0.1582, minimum_width = 0.24. Weights were kept weak during anti-fall bootstrap: foot_lateral_spacing_l1 -0.25, foot_signed_lateral_clearance_l1 -0.75, foot_lateral_lane_l1 -0.20, foot_lateral_lane_max_l1 -0.05.
training result: completed 1300 iterations from scratch. Final logged iteration 1299/1300 had mean reward about +12.76, mean episode length 150 steps, timeout fraction 1.0, base_height_l2 about -0.2805, low_body_l2 about -0.0447, termination penalty 0.0, velocity xy error about 0.0667, yaw error about 0.1555.
interpretation: valid anti-fall seed. This is not a keeper walking checkpoint. It should feed a V2.3 gait transition stage, not final evaluation.
```

```text
date: 2026-05-08
run: logs/rsl_rl/kbot_forward_flat/2026-05-08_03-52-45_v2_3_gait_transition_from_1299_sepfix
checkpoint: model_1798.pt
task id: Isaac-KBot-Forward-Flat-V2_3-GaitTransition-v0
warm start: model_1299.pt from v2_3_bootstrap_from_zero_sepfix
why it was tried: continue the fresh V2.3 anti-fall seed into the bootstrap report's gentle gait-transition stage while preserving the corrected hip-axis sep target.
training result: completed 500 continuation iterations. Final logged iteration 1798/1799 had mean reward about +9.39, mean episode length 200 steps, timeout fraction 1.0, base_height_l2 about -0.1301, low_body_l2 about -0.0486, termination penalty 0.0, velocity xy error about 0.0768, yaw error about 0.2382.
diagnostic result: evaluator REJECT. speed_tracking_ratio 0.142, speed_mean_mps 0.0266 against command 0.1874, yaw_drift_rad_per_m 1.122, lateral_drift_m_per_m 0.375, hip_roll_mean_abs_5cycle_rad 0.336, hip_yaw_mean_abs_5cycle_rad 0.423, step_length_mean_m 0.0167, root_height_mean_m 0.732.
interpretation: the stage survives and keeps height, but the first gait-transition continuation overuses hip roll/yaw and does not produce useful forward walking. Do not promote model_1798 as a keeper.
```

```text
date: 2026-05-08
run: logs/rsl_rl/kbot_forward_flat/2026-05-08_04-01-18_v2_3_gait_transition_continue_from_1798_sepfix
checkpoint: model_2297.pt
task id: Isaac-KBot-Forward-Flat-V2_3-GaitTransition-v0
warm start: model_1798.pt from v2_3_gait_transition_from_1299_sepfix
why it was tried: continue the same gentle gait-transition stage to see whether the initially rejected model_1798 was just too early.
training result: completed 500 more continuation iterations. Final logged iteration 2297/2298 had mean reward about +16.95, mean episode length 200 steps, timeout fraction 1.0, base_height_l2 about -0.0243, low_body_l2 0.0, termination penalty 0.0, velocity xy error about 0.0594, yaw error about 0.1673.
diagnostic result: evaluator REVIEW_VIDEO. speed_tracking_ratio 0.993, speed_mean_mps 0.186 against command 0.187, yaw_drift_rad_per_m -0.051, lateral_drift_m_per_m -0.163, root_height_mean_m 0.746, step_length_mean_m 0.0296, cycle_length_mean_m 0.0612, double_support_fraction 0.678. Hip geometry remained bad: hip_roll_mean_abs_5cycle_rad 0.348, hip_yaw_mean_abs_5cycle_rad 0.493.
video: logs/rsl_rl/kbot_forward_flat/2026-05-08_04-01-18_v2_3_gait_transition_continue_from_1798_sepfix/videos/play/trailing-hud-model_2297-v2_3-gait-transition-continue.mp4
interpretation: strong proof that the fresh V2.3 lineage can recover forward speed after bootstrap, but the gait-transition reward has no active hip roll/yaw penalty, so further training on this exact stage is likely to preserve or worsen the hip-axis abuse.
```

```text
date: 2026-05-08
run: logs/rsl_rl/kbot_forward_flat/2026-05-08_04-08-49_v2_3_hip_axis_posture_cleanup_from_2297
checkpoint: model_2596.pt
task id: Isaac-KBot-Forward-Flat-V2-HipAxisWidthCleanup-v0
warm start: model_2297.pt from v2_3_gait_transition_continue_from_1798_sepfix
why it was tried: continue into a stricter hip-axis posture stage that keeps the corrected 0.3164 m sep target while activating hip roll/yaw, root lateral tilt, yaw, and root lateral position penalties.
training result: completed 300 continuation iterations. Final logged iteration 2596/2597 had mean reward about -9.31 under the stricter penalty stack, mean episode length 200 steps, timeout fraction 1.0, termination penalty 0.0, velocity xy error about 0.0722, yaw error about 0.1639. The logged hip_roll_yaw penalties fell substantially during the run, from about -3.5/-3.1 early to about -1.46/-1.18 near the end.
diagnostic result: evaluator REVIEW_VIDEO. speed_tracking_ratio 0.944, speed_mean_mps 0.177 against command 0.187, yaw_drift_rad_per_m 0.008, lateral_drift_m_per_m -0.166, root_height_mean_m 0.726, step_length_mean_m 0.0299, cycle_length_mean_m 0.0553, double_support_fraction 0.551. Hip geometry improved but still fails: hip_roll_mean_abs_5cycle_rad 0.196 and hip_yaw_mean_abs_5cycle_rad 0.146.
video: logs/rsl_rl/kbot_forward_flat/2026-05-08_04-08-49_v2_3_hip_axis_posture_cleanup_from_2297/videos/play/trailing-hud-model_2596-v2_3-hip-axis-posture-cleanup.mp4
interpretation: this is better than model_2297 on hip roll/yaw and support timing, but it gave up support width in the HUD rollout: sep mean 0.209 m and final 5-cycle sep about 0.225 m, below the 0.3164 m target. It is not a keeper yet. The next continuation should keep hip posture pressure but restore stronger width/lane enforcement and start lengthening steps gradually.
```

## 10.4 V2.4 Settled-Pose Scratch Bootstrap Result

```text
date: 2026-05-08
run: logs/rsl_rl/kbot_forward_flat/2026-05-08_12-35-11_v2_4_pose_bootstrap_from_zero_settled_fsep_ksep
checkpoint: model_1299.pt
task id: Isaac-KBot-Forward-Flat-V2_4-Scratch-PoseBootstrap-v0
start: true policy iteration zero, no checkpoint resume
why it was tried: restart the next V2 lineage from the GUI-validated settled standing pose instead of asking the policy to recover from a fall or from the older knee-crossing bootstrap posture.
training result: completed 1300 iterations from scratch. Final logged iteration 1299/1300 had mean reward about +38.05, mean episode length 200 steps, timeout fraction 1.0, termination penalty 0.0, velocity xy error about 0.0113, yaw error about 0.0119, action std about 0.03.
playback: 30.0 s, 1500 frames at 50 FPS, no fall resets.
video: logs/rsl_rl/kbot_forward_flat/2026-05-08_12-35-11_v2_4_pose_bootstrap_from_zero_settled_fsep_ksep/videos/play/trailing-hud-model_1299-v2_4-pose-bootstrap-fsep-ksep.mp4
metrics: final_hud_fsep_m 0.184, final_hud_ksep_m 0.288, root_height_mean_m 0.847, root_height_final_m 0.847, speed_mean_mps 0.002.
interpretation: valid settled-pose anti-fall seed. This is not a walking checkpoint. It should feed the next gentle gait-transition branch, with `fsep` and `ksep` watched explicitly to catch foot or knee crossing before the policy reaches the old failure mode.
```

## 11. Open Questions

- Where did the current actuator parameters originate: Robstride datasheet, KBot repo, hand tuning, or earlier manual guess?
- Is the current initial pose mechanically appropriate, or only a pragmatic standing pose from early fall debugging?
- Should heel/toe/edge contact bodies or sensors be added to the asset before reward tuning?
  - Current status: not for the next iteration. Use whole-foot contact plus sole-plane vector diagnostics first. Revisit CAD/Blender five-piece soles only if vector indicators cannot distinguish a repeated failure mode.
- What mirrored sign map should be used for left/right joint symmetry?
- What thresholds should define excessive crouch for this simplified body?
- What is the target step length at each command speed?
