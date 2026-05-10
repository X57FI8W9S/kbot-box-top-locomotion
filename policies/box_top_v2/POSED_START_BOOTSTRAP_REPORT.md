# Posed-Start Bootstrap Report

Date: 2026-05-08

## Purpose

This report replaces the old assumption behind the first bootstrap stage.

The original V2 bootstrap procedure was designed for a hard reset problem: the robot started from a pose that could fall or collapse, so the first stage spent many iterations learning not to fall. That made sense before the GUI-authored settled pose was validated in headless Isaac Lab.

With the settled authored pose, the first problem is different:

- standing is already solved at reset;
- a no-policy zero-action hold can stand for at least 20 s in headless Isaac Lab;
- spending 1300 iterations on only "do not fall" can make the policy worse than the starting pose.

The new bootstrap should improve the useful authored pose into a better policy-controlled gait seed. Preserving the reset pose is only the floor. Training has to produce a measurable improvement, especially support width and preparation for stepping, otherwise the policy is just spending GPU time to imitate something the simulator can already hold.

## Current Evidence

### Authored pose is a valid headless starter

The final pose validation used:

```text
root z = 0.8565
left_hip_pitch_04  =  0.284315
right_hip_pitch_04 = -0.284115
left_hip_roll_03   =  0.001739
right_hip_roll_03  =  0.001906
left_hip_yaw_03    =  0.001332
right_hip_yaw_03   =  0.000435
left_knee_04       =  0.507304
right_knee_04      = -0.505952
left_ankle_02      = -0.246028
right_ankle_02     =  0.247223
```

Headless no-policy tests held this pose:

```text
standalone raw-USD articulation probe: 4000 physics steps = 20 s sim time
registered Isaac Lab task probe: 1000 env steps = 20 s sim time
min_z ~= 0.8559
final_z ~= 0.8565
max_abs_gravity_xy ~= 0.073-0.074
```

The required asset/config fixes were:

- use the settled pose as `init_state.joint_pos`;
- use settled root height around `0.8565`;
- use scaled implicit actuator gains to match the raw USD drive strength;
- remove the KBot `ArticulationRootPropertiesCfg` override, because self-collisions / solver overrides reproduced the headless fall while raw USD did not.

### V2.4 trained the wrong thing

V2.4 was:

```text
task = Isaac-KBot-Forward-Flat-V2_4-Scratch-PoseBootstrap-v0
run = logs/rsl_rl/kbot_forward_flat/2026-05-08_12-35-11_v2_4_pose_bootstrap_from_zero_settled_fsep_ksep
checkpoint = model_1299.pt
start = true iteration zero, no checkpoint resume
```

It completed 1300 scratch iterations:

```text
mean reward ~= 38.05
mean episode length = 200
timeout fraction = 1.0
termination_penalty = 0.0
velocity xy error ~= 0.0113
yaw error ~= 0.0119
```

The 30 s playback did not fall:

```text
fall_reset_count = 0
root_height_mean_m = 0.847
root_height_final_m = 0.847
speed_mean_mps = 0.002
```

But it gave up support width:

```text
target support width = 0.3164 m
final_hud_fsep_m = 0.184
fsep_m mean = 0.185
fsep_m p95 = 0.186
fsep_m max = 0.285
final_hud_ksep_m = 0.288
ksep_m mean = 0.289
ksep_m p95 = 0.289
```

Interpretation:

- `ksep` stayed near useful hip/knee width.
- `fsep` collapsed much lower than the desired 0.3164 m support width.
- The feet moved inward relative to the knees.
- The policy learned a narrow-foot standing solution that is worse than the intended starter pose for gait.

This is a negative result, not a file to delete. It tells us the old anti-fall bootstrap objective is now obsolete for posed starts.

## Why V2.4 Was Allowed To Happen

The old acceptance logic was incomplete for posed-start bootstrapping.

The actual checkpoint evaluator is:

```text
scripts/diagnostics/evaluate_checkpoint.py
```

It computes gait diagnostics and gates speed, yaw drift, lateral drift, root roll, hip roll, alternating steps, airborne fraction, and root height. It does not currently compute or gate:

- `fsep`: sole-center lateral foot separation in root/body coordinates;
- `ksep`: knee-proxy lateral separation in root/body coordinates;
- support-width target error from `0.3164 m`;
- minimum standing support width.

The playback script had the only `sep` measurement:

```text
scripts/rsl_rl/play_trailing.py
```

That script now displays:

- `fsep`: foot/sole-center lateral separation;
- `ksep`: knee-proxy lateral separation;
- `final_hud_sep_m`: compatibility alias for `final_hud_fsep_m`.

The missing process step was treating `fsep`/`ksep` as acceptance metrics before continuing to gait. V2.4 was accepted only as a no-fall standing seed, but for posed-start bootstrap that is too weak.

## What To Keep From The Old Bootstrap Procedure

Keep these ideas:

- staged learning is still better than full V2 from iteration zero;
- do not start full gait with every strict foot/contact/hip term enabled;
- keep short, cheap first-stage rollouts;
- evaluate using fixed rollouts and video, not scalar reward alone;
- continue to a gentle gait stage only after the first stage passes its own acceptance gates;
- preserve the staged report trail so failed branches explain what not to repeat.

Keep these specific technical fixes:

- left/right frontal-plane reward weights should remain equal unless there is measured hardware evidence;
- hip-axis support width target is still `0.3164 m`;
- lane targets are approximately `left_y = +0.1582`, `right_y = -0.1582`;
- foot/sole vector diagnostics are preferred before adding more asset contacts;
- fall-only playback reset is still the right video behavior;
- the authored pose and root height should remain the reset state for posed-start branches.

## What To Drop Or Change

Drop this old assumption:

```text
Stage 1 goal = learn not to fall for 1300 iterations.
```

For posed starts, standing is already provided by the initial condition. The replacement rule is not "make Stage 1 look better" as an end in itself. The formal rule should be:

```text
Stage N goal = produce a checkpoint that satisfies the promotion contract for Stage N+1.
```

Local stage metrics only matter when they make the next stage easier. For example, keeping the pose tall, keeping `fsep` near `0.3164 m`, and avoiding knee/foot crossing are useful in the posed-start bootstrap because they should make first-step training easier. If a checkpoint improves a local score but makes the next stage harder, the stage failed.

Specific V2.4 mistakes to avoid:

- do not leave support-width metrics only in video playback;
- do not zero the body-foot width/lane terms while relying on a weak inherited sole-lane term;
- do not keep a strong all-joint default-pose penalty if it fights support width;
- do not spend 1300 iterations on a zero-command stage unless it is improving required geometry;
- do not accept timeout-only standing without `fsep` and `ksep` gates.

## Formal Stage Structure

Each stage should be documented as a promotion contract, not just a run name. Use this exact structure:

```text
Stage ID:
  Parent:
  Task:
  Training budget:
  Purpose:
  Promotion gates:
  Reject gates:
  Next stage:
  Do-not-optimize warning:
```

For now, the stage contract values remain `UNKNOWN` until we fill them from code, evaluator gates, and measured training outcomes. The only committed part below is the proposed stage ladder.

## Proposed Stage Ladder

The proposed ladder from pre-training validation to optimized walking is:

```text
Stage ID:
  S0_PRETRAIN_ASSET_SIM_VALIDATION
  Parent: UNKNOWN
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: UNKNOWN
  Promotion gates: UNKNOWN
  Reject gates: UNKNOWN
  Next stage: UNKNOWN
  Do-not-optimize warning: UNKNOWN

Stage ID:
  S1_AUTHORED_RESET_POSE_VALIDATION
  Parent: UNKNOWN
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: UNKNOWN
  Promotion gates: UNKNOWN
  Reject gates: UNKNOWN
  Next stage: UNKNOWN
  Do-not-optimize warning: UNKNOWN

Stage ID:
  S2_POSED_GEOMETRY_POLICY_SEED
  Parent: UNKNOWN
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: UNKNOWN
  Promotion gates: UNKNOWN
  Reject gates: UNKNOWN
  Next stage: UNKNOWN
  Do-not-optimize warning: UNKNOWN

Stage ID:
  S3_FIRST_SUPPORTED_STEPS
  Parent: UNKNOWN
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: UNKNOWN
  Promotion gates: UNKNOWN
  Reject gates: UNKNOWN
  Next stage: UNKNOWN
  Do-not-optimize warning: UNKNOWN

Stage ID:
  S4_ANTI_SHUFFLE_WALK
  Parent: UNKNOWN
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: UNKNOWN
  Promotion gates: UNKNOWN
  Reject gates: UNKNOWN
  Next stage: UNKNOWN
  Do-not-optimize warning: UNKNOWN

Stage ID:
  S5_STRAIGHT_CONTACT_QUALITY_WALK
  Parent: UNKNOWN
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: UNKNOWN
  Promotion gates: UNKNOWN
  Reject gates: UNKNOWN
  Next stage: UNKNOWN
  Do-not-optimize warning: UNKNOWN

Stage ID:
  S6_SPEED_RANGE_RAMP_WALK
  Parent: UNKNOWN
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: UNKNOWN
  Promotion gates: UNKNOWN
  Reject gates: UNKNOWN
  Next stage: UNKNOWN
  Do-not-optimize warning: UNKNOWN

Stage ID:
  S7_MAX_RANGE_GAIT_SEARCH
  Parent: UNKNOWN
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: UNKNOWN
  Promotion gates: UNKNOWN
  Reject gates: UNKNOWN
  Next stage: UNKNOWN
  Do-not-optimize warning: UNKNOWN

Stage ID:
  S8_MAX_SAFE_WALKING_SPEED
  Parent: UNKNOWN
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: UNKNOWN
  Promotion gates: UNKNOWN
  Reject gates: UNKNOWN
  Next stage: UNKNOWN
  Do-not-optimize warning: UNKNOWN

Stage ID:
  S9_WALKING_ROBUSTNESS_AND_FINAL_SELECTION
  Parent: UNKNOWN
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: UNKNOWN
  Promotion gates: UNKNOWN
  Reject gates: UNKNOWN
  Next stage: UNKNOWN
  Do-not-optimize warning: UNKNOWN
```

Stage naming intent:

- `S0_PRETRAIN_ASSET_SIM_VALIDATION`: verify USD, actuator configuration, reset height, headless/GUI consistency, and no-policy pose behavior before policy training starts.
- `S1_AUTHORED_RESET_POSE_VALIDATION`: confirm the reset pose is a valid initial condition for the training task and not just a raw USD GUI behavior.
- `S2_POSED_GEOMETRY_POLICY_SEED`: produce a policy-controlled seed that preserves or improves reset geometry without optimizing standing for its own sake.
- `S3_FIRST_SUPPORTED_STEPS`: make supported alternating root advance non-optional.
- `S4_ANTI_SHUFFLE_WALK`: remove high-frequency contact chatter and micro-step exploits.
- `S5_STRAIGHT_CONTACT_QUALITY_WALK`: improve yaw, lateral drift, foot flatness, support fractions, and contact quality after real steps exist.
- `S6_SPEED_RANGE_RAMP_WALK`: expand the commanded walking speed range while preserving the Stage 5 quality gates.
- `S7_MAX_RANGE_GAIT_SEARCH`: find the lowest energy per meter over the useful walking range. The best point in this stage is the `max range gait`.
- `S8_MAX_SAFE_WALKING_SPEED`: identify the fastest speed that remains walking, safe, and gate-compliant; this is not necessarily energy-optimal.
- `S9_WALKING_ROBUSTNESS_AND_FINAL_SELECTION`: select final policies across max range gait, general walking, and maximum safe walking speed after robustness checks.

`max range gait` is the name for the gait that should travel the farthest for a fixed energy budget. For now, the simulation proxy is lowest positive joint mechanical work per meter of forward advance. Later this should include estimated baseline system energy consumption, because very slow walking can look efficient in joint work while wasting real battery energy on compute, sensors, fans, and motor-controller overhead.

## Proposed Posed-Start Bootstrap Gates

A posed-start checkpoint should be marked usable for gait only if all of these hold in a fixed 30 s headless rollout:

```text
fall_reset_count = 0
root_height_p05_m >= 0.82
root_height_final_m >= 0.82
final_hud_fsep_m >= 0.28
fsep_m mean >= 0.28
fsep_m p05 >= 0.24
final_hud_ksep_m >= 0.26
ksep_m mean >= 0.26
fsep_target_error_mean_m <= 0.06
abs(fsep_mean - 0.3164) is tracked, even if not hard-gated initially
```

The emergency hard floor is `0.24 m`; below that the checkpoint should be rejected outright. But passing V2.5 should require meaningful improvement toward the target, not just avoiding collapse. A final or mean `fsep` around `0.24 m` is still too low to call progress.

For gait-stage checkpoints, `fsep` should move toward `0.3164 m`, and any checkpoint with good speed but collapsed `fsep` should be rejected as a support-width exploit.

## V2.5 Proposed Restart

V2.5 should start from iteration zero again. It should not resume V2.4.

Recommended first-stage task behavior:

- use the same settled authored root pose and joint pose;
- do not keep zero command for a long static stage;
- introduce a very small first-step command immediately, around `0.06-0.14 m/s`;
- keep the corrected implicit actuator groups;
- keep no custom articulation root override;
- keep short episodes around 4 s;
- reduce the all-joint stand-pose penalty so the policy can widen support and improve over reset;
- explicitly enable support-width rewards from the beginning;
- enable weak first-step terms from iteration zero, not after hundreds of standing iterations;
- add evaluator/playback gates for `fsep` and `ksep`.

Initial V2.5 reward direction:

```text
foot_lateral_spacing_l1 target_width = 0.3164, weight around -3.0
foot_signed_lateral_clearance_l1 minimum_width = 0.24, weight around -4.0
foot_lateral_lane_l1 left/right = +/-0.1582, weight around -1.5
foot_lateral_lane_max_l1 left/right = +/-0.1582, weight around -0.8
foot_sole_lateral_lane_max_l1 left/right ~= +/-0.1582, keep active and stronger than V2.4
stand_joint_position_l2 weight reduce from -4.0 to about -1.0, or disable if it continues to fight width
centered_joint_target_position_l2 disable for this stage if inherited
root height/upright/flatness stay active
track_lin_vel_xy_exp active immediately, with a narrow enough std that zero speed is not rewarded as good tracking
forward_velocity_below_l2 active immediately, minimum_velocity at or near the command floor
feet_air_time active immediately, threshold about 0.18
alternating_foot_phase weak but nonzero
foot_sagittal_separation_l1 target_length about 0.08, weak
swing_foot_overtake_l1 target_length about 0.06, weak
```

This should be a shorter bootstrap than 1300 iterations and should not spend a separate long phase learning only to stand. Suggested first run:

```text
max_iterations = 400-600
stop early if fsep remains below 0.24 after the policy stabilizes
continue only if no-fall and support-width improvement gates pass
```

If V2.5 passes, the next stage should add gentle gait from that checkpoint. If it fails, the failure should be classified by which geometry broke:

- `fsep` low, `ksep` ok: feet collapsing inward under knees;
- `fsep` ok, `ksep` low: knee crossing / hip geometry problem;
- both low: whole support collapse;
- both ok, no speed: safe pose stage, ready for gait pressure;
- speed ok but fsep low: gait exploit, reject.

## V2.5 First-Step Attempt

The first V2.5 attempt did improve the support geometry, but it still did not produce useful stepping:

```text
task = Isaac-KBot-Forward-Flat-V2_5-Scratch-PoseWidthBootstrap-v0
run = logs/rsl_rl/kbot_forward_flat/2026-05-08_23-20-24_v2_5_pose_first_steps_from_zero_fsep_ksep
checkpoint = model_599.pt
decision = REJECT
```

The 30 s evaluator result:

```text
speed_mean_mps = 0.0018
command_speed_mean_mps = 0.0940
speed_tracking_ratio = 0.019
step_count = 1
double_support_fraction = 0.999
fsep_mean_m = 0.275
fsep_p05_m = 0.274
ksep_mean_m = 0.311
root_height_mean_m = 0.854
fall_reset_count = 0
```

Interpretation:

- V2.5 fixed most of the V2.4 foot-collapse problem: `fsep` rose from about `0.184 m` to about `0.275 m`, and `ksep` stayed healthy.
- It still failed as a policy seed because it stood nearly still under a small forward command.
- The first V2.5 reward mix still made stationary standing too profitable: command tracking was too forgiving around zero speed, and alive/upright rewards were still large enough to dominate.

The next V2.5 run should therefore keep the width terms but make first steps non-optional from iteration zero:

```text
lin_vel_x command range = 0.08-0.16 m/s
track_lin_vel_xy_exp weight = 3.0
track_lin_vel_xy_exp std = sqrt(0.01)
forward_velocity_below_l2 weight = -30.0
forward_velocity_below_l2 minimum_velocity = 0.08
feet_air_time weight = 0.8
alternating_foot_phase weight = 0.2
alive weight = 0.75
upright_alive weight = 5.0
```

This is not a return to "train standing first." It is a direct posed-start first-step stage whose acceptance requires both no fall and measurable movement.

That stronger version was also tested:

```text
run = logs/rsl_rl/kbot_forward_flat/2026-05-08_23-31-19_v2_5_pose_first_steps_stronger_from_zero_fsep_ksep
checkpoint = model_349.pt
decision = REJECT
```

Evaluator summary:

```text
speed_mean_mps = -0.0002
command_speed_mean_mps = 0.1140
distance_m = 0.000001
step_count = 42
fsep_mean_m = 0.335
ksep_mean_m = 0.328
edge_walk_proxy_fraction_left = 0.96
edge_walk_proxy_fraction_right = 0.971
yaw_drift_rad_per_m = very high because net distance was near zero
```

Interpretation:

- Stronger first-step pressure produced contact transitions and preserved width.
- It did not produce forward travel; it produced in-place/sideways stepping with edge-walk contacts.
- The failure is now a fake-step exploit, not a standing exploit.

The next patch should therefore be more specific, not just stronger:

```text
add world_forward_velocity_below_l2 so reward pressure matches evaluator distance;
moderate the overly aggressive speed/air-time weights;
keep fsep/ksep terms active;
raise foot-flat and heading/yaw penalties enough to reject edge-walk in-place steps.
```

That world-forward version was tested next:

```text
run = logs/rsl_rl/kbot_forward_flat/2026-05-08_23-38-01_v2_5_pose_world_forward_first_steps_from_zero_fsep_ksep
checkpoint = model_349.pt
decision = REJECT
```

Evaluator summary:

```text
distance_m = 1.048
speed_mean_mps = -0.106
command_speed_mean_mps = 0.094
step_count = 315
root_height_mean_m = -0.525
fsep_mean_m = 0.267
fsep_p05_m = 0.226
ksep_mean_m = 0.313
```

Interpretation:

- Adding world-forward pressure made the policy move, but it did so by falling/crawling.
- The task still had only `time_out` termination, so the policy could continue collecting episode data after the robot was no longer standing.
- Earlier first-step pressure requires a fall guard. Otherwise it turns into a collapse exploit instead of gait.

The V2.5 task should therefore terminate low-body and bad-orientation states during this first-step phase:

```text
low_body termination minimum_height = 0.76
bad_orientation limit_angle = 0.75
upright_alive minimum_height = 0.76
```

The fall-guarded version was then tested:

```text
run = logs/rsl_rl/kbot_forward_flat/2026-05-08_23-48-51_v2_5_pose_world_forward_fall_guard_from_zero_fsep_ksep
checkpoint = model_349.pt
decision = REJECT
```

Evaluator summary:

```text
distance_m = 0.144
speed_mean_mps = 0.014
command_speed_mean_mps = 0.089
speed_tracking_ratio = 0.160
step_count = 3
double_support_fraction = 0.998
root_height_mean_m = 0.853
root_height_p05_m = 0.847
fsep_mean_m = 0.280
fsep_p05_m = 0.279
ksep_mean_m = 0.314
```

Interpretation:

- The fall guard worked: the collapse/crawl exploit disappeared and root height stayed valid.
- Support geometry remained much better than V2.4 and close to the acceptance threshold.
- It still did not become gait. The policy found a guarded safe-hold solution with tiny forward drift, almost no swing, and near-total double support.
- This proves the next change should not be another anti-fall bootstrap. The next training pressure needs to make actual alternating swing and forward root advance profitable while keeping the fall guard.

Next direction:

```text
keep low_body / bad_orientation termination;
keep fsep/ksep gates;
add positive forward-progress reward or a short command-progress curriculum;
increase the reward for real alternating swing only when root advance is positive;
do not accept distance from collapse, edge walking, or double-support drift.
```

The first positive-forward run was tested:

```text
run = logs/rsl_rl/kbot_forward_flat/2026-05-08_23-55-48_v2_5_pose_forward_reward_fall_guard_from_zero_fsep_ksep
checkpoint = model_349.pt
decision = REVIEW_VIDEO
```

Evaluator summary:

```text
distance_m = 4.774
speed_mean_mps = 0.161
command_speed_mean_mps = 0.094
speed_tracking_ratio = 1.714
step_count = 291
double_support_fraction = 0.294
root_height_mean_m = 0.862
fsep_mean_m = 0.212
fsep_p05_m = 0.205
fsep_target_error_mean_m = 0.105
ksep_mean_m = 0.296
lateral_drift_m_per_m = 0.109
```

Interpretation:

- Positive forward progress worked too well: the policy learned forward motion quickly.
- It was not acceptable because it collapsed foot support width from the 0.3164 m target to about 0.212 m.
- This is exactly why `fsep` has to be an evaluator gate. Without it, this run would look like a breakthrough from distance and step count alone.

The next run reduced the raw forward bonus and made support width much more expensive:

```text
run = logs/rsl_rl/kbot_forward_flat/2026-05-09_00-01-36_v2_5_pose_forward_width_guard_from_zero_fsep_ksep
checkpoint = model_349.pt
decision = REJECT
```

Evaluator summary:

```text
distance_m = 4.306
speed_mean_mps = 0.150
command_speed_mean_mps = 0.094
speed_tracking_ratio = 1.590
step_count = 306
double_support_fraction = 0.416
root_height_mean_m = 0.859
fsep_mean_m = 0.307
fsep_p05_m = 0.304
fsep_target_error_mean_m = 0.009
ksep_mean_m = 0.320
yaw_drift_rad_per_m = -0.099
lateral_drift_m_per_m = -0.221
```

Interpretation:

- The width guard worked. `fsep` and `ksep` both passed, and the feet were close to the 0.3164 m target.
- It was still rejected because the policy walked diagonally / laterally too much.
- The next patch needed to target direction, not more support width.

The successful V2.5 posed-start run added tighter direction control:

```text
run = logs/rsl_rl/kbot_forward_flat/2026-05-09_00-07-27_v2_5_pose_forward_width_heading_guard_from_zero_fsep_ksep
checkpoint = model_349.pt
decision = APPROVE
```

Key config changes versus the previous rejected run:

```text
track_lin_vel_xy_exp weight = 4.0
track_lin_vel_xy_exp std = sqrt(0.01)
world_forward_velocity_clip weight = 3.0
lateral_velocity_l2 weight = -18.0
yaw_rate_l2 weight = -18.0
world_heading_l2 weight = -80.0
```

Evaluator summary:

```text
distance_m = 3.239
speed_mean_mps = 0.109
command_speed_mean_mps = 0.094
speed_tracking_ratio = 1.158
step_count = 352
double_support_fraction = 0.596
root_height_mean_m = 0.860
root_height_p05_m = 0.858
fsep_mean_m = 0.304
fsep_p05_m = 0.301
fsep_final_m = 0.303
fsep_target_error_mean_m = 0.012
ksep_mean_m = 0.320
ksep_p05_m = 0.319
yaw_drift_rad_per_m = 0.018
lateral_drift_m_per_m = 0.073
```

Interpretation:

- This is the first posed-start scratch checkpoint in this series that passes all current evaluator gates.
- It improves over the raw pose by producing forward progress while preserving useful support width.
- It should be treated as the V2.5 bootstrap seed for the next staged gait run, not as a final gait policy.
- Residual risk remains: edge-walk proxy fractions are still high and feet-air-time is very low. The next stage should improve step quality, not simply push speed higher.

## V2.5 Pose Gait Quality Continuation

The gait-quality stage continues inside the V2.5 lineage. It continued from the approved V2.5 checkpoint with policy-only resume:

```text
task = Isaac-KBot-Forward-Flat-V2_5-PoseGaitQuality-v0
source checkpoint = 2026-05-09_00-07-27_v2_5_pose_forward_width_heading_guard_from_zero_fsep_ksep/model_349.pt
run = logs/rsl_rl/kbot_forward_flat/2026-05-09_00-16-45_v2_5_pose_gait_quality_from_v2_5_349_fsep_ksep
checkpoint = model_648.pt
decision = APPROVE
```

This was deliberately a continuation stage, not another scratch bootstrap or a new version number. It kept the V2.5 width and heading guards, then added pressure for cleaner gait quality:

```text
command lin_vel_x = 0.08-0.16 m/s
feet_air_time weight = 1.0, threshold = 0.22
alternating_foot_phase weight = 0.18
foot_sagittal_separation_l1 target_length = 0.10, weight = -2.0
swing_foot_overtake_l1 target_length = 0.08, target_air_time = 0.20, weight = -3.0
root_lateral_position_l2 weight = -12.0
foot_world_parallel_max_l2 weight = -0.8
stance_foot_flat_l2 weight = -1.2
```

Evaluator comparison:

```text
metric                         V2.5 approved      V2.5 gait-quality continuation
distance_m                     3.239              3.721
speed_mean_mps                 0.109              0.126
command_speed_mean_mps         0.094              0.114
speed_tracking_ratio           1.158              1.102
fsep_mean_m                    0.304              0.307
fsep_p05_m                     0.301              0.303
fsep_target_error_mean_m       0.012              0.009
ksep_mean_m                    0.320              0.321
step_count                     352                459
double_support_fraction        0.596              0.461
root_height_p05_m              0.858              0.854
yaw_drift_rad_per_m            0.018              -0.000
lateral_drift_m_per_m          0.073              -0.004
edge_walk_proxy_fraction_left  0.797              0.723
edge_walk_proxy_fraction_right 0.799              0.738
stance_slip_mean_mps           0.025              0.019
```

Interpretation:

- The gait-quality continuation is better than the first V2.5 approved seed on the current evaluator and should replace it as the active continuation seed.
- It improved direction control and reduced lateral drift almost to zero.
- It preserved support geometry: `fsep` and `ksep` remain near the intended width.
- It reduced double support and edge-walk proxy fractions, but did not eliminate them.
- Feet-air-time remains low, so the next stage should focus on true swing clearance and contact quality without breaking width or heading.

Follow-up continuation from `model_648.pt`:

```text
task = Isaac-KBot-Forward-Flat-V2_5-PoseGaitQuality-v0
run = logs/rsl_rl/kbot_forward_flat/2026-05-09_01-00-12_v2_5_gait_quality_continue_from_648
checkpoint = model_947.pt
decision = APPROVE
video = logs/rsl_rl/kbot_forward_flat/2026-05-09_01-00-12_v2_5_gait_quality_continue_from_648/videos/play/trailing-hud-model_947-v2_5-x-y-fsep-ksep.mp4
```

Key metrics:

```text
x_distance_m                  3.771
y_distance_m                  0.219
speed_mean_mps                0.128
speed_tracking_ratio          1.118
fsep_mean_m                   0.307
ksep_mean_m                   0.321
double_support_fraction       0.465
airborne_fraction             0.001
step_length_mean_m            0.007
cycle_length_mean_m           0.015
yaw_drift_rad_per_m           0.036
lateral_drift_m_per_m         0.058
stance_slip_mean_mps          0.020
```

Interpretation: `model_947.pt` passes the current gates and preserves support width, but it does not obviously solve the tiny-step/contact-chatter problem. The raw Y drift is worse than `model_648.pt`, and mean step length remains around 7 mm. Treat it as a candidate only after video review, not as an automatic keeper.

Continuation from `model_947.pt`:

```text
task = Isaac-KBot-Forward-Flat-V2_5-PoseGaitQuality-v0
run = logs/rsl_rl/kbot_forward_flat/2026-05-09_01-21-36_v2_5_gait_quality_continue_from_947
checkpoint = model_1246.pt
decision = APPROVE
keeper = no
```

Key metrics:

```text
x_distance_m                  3.666
y_distance_m                  0.518
speed_mean_mps                0.125
speed_tracking_ratio          1.100
fsep_mean_m                   0.305
fsep_p05_m                    0.302
ksep_mean_m                   0.320
double_support_fraction       0.466
airborne_fraction             0.001
step_length_mean_m            0.007
cycle_length_mean_m           0.014
yaw_drift_rad_per_m           0.063
lateral_drift_m_per_m         0.141
stance_slip_mean_mps          0.019
edge_walk_proxy_fraction_left 0.737
edge_walk_proxy_fraction_right 0.728
```

Interpretation: `model_1246.pt` passes the current evaluator gates but is worse than the earlier V2.5 candidates. It loses forward distance versus `model_947.pt`, accumulates much more lateral drift, worsens yaw drift, and still has tiny approximately 7 mm steps. Do not keep it as the next seed. This is evidence that the evaluator still needs a stronger gate for useful step length / root advance and lateral drift, because a policy can pass while mostly doing high-frequency contact chatter.

Walking-only gate update:

```text
evaluator patch = max per-foot cycle cadence, root advance per step/cycle, tighter lateral drift
diagnostic rerun = logs/rsl_rl/kbot_forward_flat/2026-05-09_01-21-36_v2_5_gait_quality_continue_from_947/diagnostics/model_1246_headless_walk_gates
decision = REJECT
```

Key failed metrics:

```text
max_cycle_cadence_hz          8.835
left_cycle_cadence_hz         8.835
right_cycle_cadence_hz        8.690
step_root_advance_mean_m      0.0069
cycle_root_advance_mean_m     0.0140
lateral_drift_m_per_m         0.141
airborne_fraction             0.001
double_support_fraction       0.466
```

Interpretation: cadence is not the definition of walking versus running. The walking-only definition should remain contact/support based: no meaningful flight phase, controlled support transitions, and root advance through supported steps. The cadence gate is an anti-chatter guard because the current loophole uses approximately 9 same-foot cycles per second, or roughly 18 total footfall events per second. A real keeper should trend toward about `0.5-1.25 Hz` per-foot cycle cadence at this slow stage, with speed raised later after support quality is solved.

Reward update:

```text
valid_step_root_advance reward:
  rewards alternating touchdowns only when swing time and root advance are sufficient
walking_cycle_cadence_above_l2:
  penalizes per-foot same-foot cadence above walking range
contact_chatter_l1:
  penalizes touchdown events with too little preceding air/swing time
swing_foot_overtake_l1:
  grace time increased so millimeter oscillation gets less useful signal
```

Training test with the walking-only reward/gates:

```text
source checkpoint = 2026-05-09_00-16-45_v2_5_pose_gait_quality_from_v2_5_349_fsep_ksep/model_648.pt
run = logs/rsl_rl/kbot_forward_flat/2026-05-09_02-36-29_v2_5_walk_only_contact_quality_from_648
resume mode = policy_only_resume
task = Isaac-KBot-Forward-Flat-V2_5-PoseGaitQuality-v0
```

Mid-run `model_800.pt`:

```text
decision = REJECT
x_distance_m                  3.430
speed_tracking_ratio          1.033
max_cycle_cadence_hz          7.525
step_root_advance_mean_m      0.0076
cycle_root_advance_mean_m     0.0154
fsep_mean_m                   0.307
ksep_mean_m                   0.321
lateral_drift_m_per_m         -0.136
```

Final `model_947.pt`:

```text
decision = REJECT
x_distance_m                  0.390
speed_tracking_ratio          0.144
max_cycle_cadence_hz          6.049
step_root_advance_mean_m      0.0032
cycle_root_advance_mean_m     0.0022
fsep_mean_m                   0.308
ksep_mean_m                   0.321
lateral_drift_m_per_m         0.104
```

Interpretation: the evaluator now catches the exploit, but the first reward patch did not produce useful walking. The policy either keeps high-cadence shuffling while tracking speed (`model_800`) or slows down into low-advance shuffling (`model_947`). `valid_step_root_advance` stayed at zero in training logs, so the reward is too sparse or too hard to discover from this seed. Do not continue this run.

## Immediate To-Do

1. Keep `fsep` and `ksep` in `evaluate_checkpoint.py`.
2. Keep hard/reporting gates for minimum support width.
3. Do not continue from `2026-05-09_01-21-36_v2_5_gait_quality_continue_from_947/model_1246.pt`.
4. Do not continue from `2026-05-09_02-36-29_v2_5_walk_only_contact_quality_from_648/model_947.pt`.
5. Use `2026-05-09_00-16-45_v2_5_pose_gait_quality_from_v2_5_349_fsep_ksep/model_648.pt` as the conservative active seed.
6. Keep the new evaluator gates for cadence, step root advance, cycle root advance, and lateral drift.
7. Revise the reward before the next run: make valid step/root advance less sparse, likely by using a shaped continuous root-advance target during swing/stance instead of only touchdown events.
8. Keep walking-only support semantics: no meaningful flight phase, controlled support transitions, and root advance through supported steps. Cadence remains an anti-chatter guard, not the definition of walking.
9. Reject standing, fake in-place stepping, edge-walk shuffling, and collapse/crawl movement even if some gait counters pass.
10. Run 30 s playback and diagnostics before any gait continuation.

The main principle is: with an authored pose, bootstrap should preserve good geometry, not relearn survival.
