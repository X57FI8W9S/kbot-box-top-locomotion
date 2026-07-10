# Posed-Start Curriculum

Date: 2026-05-10

This document defines the posed-start curriculum as a stage ladder. It is separate from the bootstrap report so the curriculum contract can stay stable while individual reports record experiments, failures, and code changes.

Core rule:

```text
Stage N goal = produce a checkpoint that satisfies the promotion contract for Stage N+1.
```

A stage is a required robot capability and promotion contract. A config class is one concrete implementation attempt for a stage. A task ID is the command-line handle that selects that config. A run is one training execution. A checkpoint is the artifact. The evaluator decides whether that artifact promotes.

## Contract Template

Use this structure for every stage:

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

Values are filled only where already known from code, diagnostics, or reports. Unknown values remain `UNKNOWN`.

## Stage Ladder

```text
Stage ID:
  S0_PRETRAIN_ASSET_SIM_VALIDATION
  Parent: NONE
  Task: No training task. Raw USD / asset probe stage.
  Training budget: NONE
  Purpose: Verify USD, actuator configuration, reset height, headless/GUI consistency, and no-policy pose behavior before policy training starts.
  Promotion gates:
    Raw USD/headless pose holds for 4000 physics steps = 20 s.
    Registered Isaac Lab pose probe holds for 1000 env steps = 20 s.
    min_z ~= 0.8559.
    final_z ~= 0.8565.
    max_abs_gravity_xy ~= 0.073-0.074.
  Reject gates: UNKNOWN as formal gates; falling/headless mismatch rejects in practice.
  Next stage: S1_AUTHORED_RESET_POSE_VALIDATION
  Do-not-optimize warning: UNKNOWN

Stage ID:
  S1_AUTHORED_RESET_POSE_VALIDATION
  Parent: S0_PRETRAIN_ASSET_SIM_VALIDATION
  Task: Isaac-KBot-Forward-Flat-V2-Scratch-PoseBootstrap-v0
  Training budget: NONE for validation. This is not policy training.
  Purpose: Confirm the reset pose is a valid initial condition for the training task, not just raw USD GUI behavior.
  Promotion gates:
    Same known pose-hold facts as S0 when loaded through Isaac Lab task path.
  Reject gates: UNKNOWN as formal gates.
  Next stage: S2_POSED_GEOMETRY_POLICY_SEED
  Do-not-optimize warning: Do not treat standing in GUI/raw USD alone as proof that the task reset is valid.

Stage ID:
  S2_POSED_GEOMETRY_POLICY_SEED
  Parent: S1_AUTHORED_RESET_POSE_VALIDATION
  Task: Isaac-KBot-Forward-Flat-V2_5-Scratch-PoseWidthBootstrap-v0
  Training budget: UNKNOWN formally.
  Purpose: Produce a policy-controlled seed that preserves or improves reset geometry without optimizing standing for its own sake.
  Promotion gates:
    fall_reset_count = 0
    root_height_p05_m >= 0.82
    root_height_final_m >= 0.82
    final_hud_fsep_m >= 0.28
    fsep_m mean >= 0.28
    fsep_m p05 >= 0.24
    final_hud_ksep_m >= 0.26
    ksep_m mean >= 0.26
    fsep_target_error_mean_m <= 0.06
  Reject gates:
    fsep below 0.24 m hard floor.
    timeout-only standing without fsep/ksep gates.
    speed ok but fsep low = gait exploit.
  Next stage: S3_FIRST_SUPPORTED_STEPS
  Do-not-optimize warning: Do not optimize static standing. The checkpoint must be easier to turn into gait.

Stage ID:
  S3_FIRST_SUPPORTED_STEPS
  Parent: S2_POSED_GEOMETRY_POLICY_SEED
  Task: Isaac-KBot-Forward-Flat-V2_5-PoseGaitQuality-v0
  Training budget: UNKNOWN formally.
  Purpose: Make supported alternating root advance non-optional.
  Promotion gates:
    Preserve fsep/ksep near target.
    Measurable forward movement.
    No fall.
    Exact formal gates: UNKNOWN.
  Reject gates:
    Fake in-place stepping.
    Collapse/crawl movement.
    Support-width exploit.
    Exact formal gates: UNKNOWN.
  Next stage: S4_ANTI_SHUFFLE_WALK
  Do-not-optimize warning: Do not accept standing plus gait counters as first steps.

Stage ID:
  S4_ANTI_SHUFFLE_WALK
  Parent: S3_FIRST_SUPPORTED_STEPS
  Task: Isaac-KBot-Forward-Flat-V2_5-PoseGaitQuality-v0 currently, but likely needs explicit S4 sub-stage configs.
  Training budget: UNKNOWN
  Purpose: Remove high-frequency contact chatter and micro-step exploits while keeping low-speed, alternating, nonzero root advance.
  Promotion gates:
    S4 sub-stage gates pass in order; see Stage 4 Anti-Shuffle Sub-Curriculum.
    Debounced cadence is lower.
    Approved steps exist on both sides.
    Step/cycle root advance are positive.
    Height and width do not regress.
  Reject gates:
    Cadence improves by crouching.
    Cadence improves by suppressing steps.
    Speed improves while cycle_root_advance_mean_m remains near zero.
    Step advance comes from sliding, yaw, or lateral drift.
    Only one side produces approved steps.
  Next stage: S5_STRAIGHT_CONTACT_QUALITY_WALK
  Do-not-optimize warning: S4 is not speed training, full contact-quality walking, or foot-flat polish.

Stage ID:
  S5_STRAIGHT_CONTACT_QUALITY_WALK
  Parent: S4_ANTI_SHUFFLE_WALK
  Task: Isaac-KBot-Forward-Flat-V2_5-PoseGaitQuality-v0 currently, but likely needs explicit sub-stage configs.
  Training budget: UNKNOWN
  Purpose: Convert the posed-start shuffle into stable straight walking before speed expansion.
  Promotion gates:
    Y distance trends toward zero.
    X distance tracks command over the rollout.
    Support percentages and edge-walk proxies are acceptable.
    Step/cycle advance inherited from S4 is preserved.
    J/m is tracked but not optimized yet.
  Reject gates:
    Straight-line metrics improve by returning to in-place shuffling.
    Support quality improves only by suppressing steps.
    Speed improves while cycle_root_advance_mean_m remains tiny.
  Next stage: S6_SPEED_RANGE_RAMP_WALK
  Do-not-optimize warning: Do not optimize speed in S5. S5 exists to make the gait real, tall, straight, and support-safe.

Stage ID:
  S6_SPEED_RANGE_RAMP_WALK
  Parent: S5_STRAIGHT_CONTACT_QUALITY_WALK
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: Expand commanded walking speed range while preserving earlier quality gates.
  Promotion gates: UNKNOWN
  Reject gates: UNKNOWN
  Next stage: S7_MAX_RANGE_GAIT_SEARCH
  Do-not-optimize warning: UNKNOWN

Stage ID:
  S7_MAX_RANGE_GAIT_SEARCH
  Parent: S6_SPEED_RANGE_RAMP_WALK
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: Find the gait with lowest energy per meter over the useful walking range.
  Promotion gates: UNKNOWN
  Reject gates: UNKNOWN
  Next stage: S8_MAX_SAFE_WALKING_SPEED
  Do-not-optimize warning: Current J/m is only positive joint mechanical work per meter. Later it must include baseline system energy.

Stage ID:
  S8_MAX_SAFE_WALKING_SPEED
  Parent: S7_MAX_RANGE_GAIT_SEARCH
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: Identify fastest speed that remains walking, safe, and gate-compliant.
  Promotion gates: UNKNOWN
  Reject gates: UNKNOWN
  Next stage: S9_WALKING_ROBUSTNESS_AND_FINAL_SELECTION
  Do-not-optimize warning: Max safe walking speed is not necessarily the max range gait.

Stage ID:
  S9_WALKING_ROBUSTNESS_AND_FINAL_SELECTION
  Parent: S8_MAX_SAFE_WALKING_SPEED
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: Select final policies across max range gait, general walking, and max safe walking speed.
  Promotion gates: UNKNOWN
  Reject gates: UNKNOWN
  Next stage: NONE / deployment candidate
  Do-not-optimize warning: UNKNOWN
```

## Known Artifacts

Current approved posed-geometry seed:

```text
Stage: S2_POSED_GEOMETRY_POLICY_SEED
run = logs/rsl_rl/kbot_forward_flat/2026-05-09_00-07-27_H9
checkpoint = model_349.pt
decision = APPROVE
```

Current conservative active seed for first-step / anti-shuffle work:

```text
Stage: S3_FIRST_SUPPORTED_STEPS / S4_ANTI_SHUFFLE_WALK boundary
run = logs/rsl_rl/kbot_forward_flat/2026-05-09_00-16-45_H9.1
checkpoint = model_648.pt
decision = APPROVE
status = conservative active seed, not final gait
```

Known rejects that define S4:

```text
run = logs/rsl_rl/kbot_forward_flat/2026-05-09_01-21-36_H10.1
checkpoint = model_1246.pt
decision = REJECT after tightened walk gates
reason = high cadence, tiny root advance, lateral/yaw regression

run = logs/rsl_rl/kbot_forward_flat/2026-05-09_02-36-29_H9.1.1
checkpoint = model_947.pt
decision = REJECT
reason = high cadence, tiny root advance, poor speed tracking
```

Current S4 anti-shuffle evidence:

```text
run = logs/rsl_rl/kbot_forward_flat/2026-05-15_08-43-54_H9.1.13.1.2
checkpoint = model_885.pt
decision = REVIEW_VIDEO
summary = best current normal-speed posed-start baseline before the latest continuations
speed_tracking_ratio = 0.7647
step_root_advance_mean_m = 0.0065
cycle_root_advance_mean_m = 0.0133
cycle_cadence_hz = 6.74
root_height_p05_m = 0.8502

run = logs/rsl_rl/kbot_forward_flat/2026-05-15_14-30-16_H9.1.13.1.2.2
checkpoint = model_964.pt
decision = REVIEW_VIDEO
summary = cadence improved, but step advance and speed regressed; usable as evidence, not a promotion
speed_tracking_ratio = 0.6924
step_root_advance_mean_m = 0.0055
cycle_root_advance_mean_m = 0.0129
cycle_cadence_hz = 5.89
root_height_p05_m = 0.8479

run = logs/rsl_rl/kbot_forward_flat/2026-05-15_14-36-39_H9.1.13.1.2.2.1
checkpoint = model_1043.pt
decision = REVIEW_VIDEO
summary = cadence and step advance improved, but speed overshot and height margin degraded; not a clean continuation seed
speed_tracking_ratio = 1.5025
step_root_advance_mean_m = 0.0091
cycle_root_advance_mean_m = 0.0018
cycle_cadence_hz = 4.68
root_height_p05_m = 0.8125

run = logs/rsl_rl/kbot_forward_flat/2026-05-09_00-16-45_H9.1
checkpoint = model_648.pt
diagnostics = diagnostics/model_648_headless_s4_baseline
summary = refreshed S4 parent baseline with approved-step and clearance metrics
speed_tracking_ratio = 1.1019
approved_step_fraction = 0.1220
left_approved_step_fraction = 0.1304
right_approved_step_fraction = 0.1135
step_root_advance_mean_m = 0.0072
cycle_root_advance_mean_m = 0.0154
cycle_cadence_hz = 8.06
root_height_p05_m = 0.8545
fsep_mean_m = 0.3069
ksep_mean_m = 0.3209
swing_sole_clearance_mean_left_m = 0.0036
swing_sole_clearance_mean_right_m = 0.0029

run = logs/rsl_rl/kbot_forward_flat/2026-05-15_16-03-47_H9.1.14
checkpoint = model_727.pt
decision = REJECT
summary = first S4.2 attempt reduced cadence only trivially and collapsed the working approved-step signal
speed_tracking_ratio = 1.4363
approved_step_fraction = 0.0574
step_root_advance_mean_m = 0.0054
cycle_root_advance_mean_m = 0.0116
cycle_cadence_hz = 7.94
root_height_p05_m = 0.8508

run = logs/rsl_rl/kbot_forward_flat/2026-05-15_16-42-06_H9.1.15
checkpoint = model_697.pt
decision = REJECT
summary = recovered the 8 mm / 70 ms approved-step reward gate, but the 5 Hz cadence ceiling was still too abrupt
speed_tracking_ratio = 1.3772
approved_step_fraction = 0.0801
left_approved_step_fraction = 0.0594
right_approved_step_fraction = 0.1009
step_root_advance_mean_m = 0.0061
cycle_root_advance_mean_m = 0.0131
cycle_cadence_hz = 7.79
root_height_p05_m = 0.8517
swing_sole_clearance_mean_left_m = 0.0039
swing_sole_clearance_mean_right_m = 0.0028

run = logs/rsl_rl/kbot_forward_flat/2026-05-15_16-47-01_H9.1.16
checkpoint = model_697.pt
decision = REJECT
summary = 7.5 Hz ceiling preserved height/width/clearance but did not suppress shuffle; cadence and overspeed worsened
speed_tracking_ratio = 1.5081
approved_step_fraction = 0.1040
left_approved_step_fraction = 0.1162
right_approved_step_fraction = 0.0917
step_root_advance_mean_m = 0.0065
cycle_root_advance_mean_m = 0.0134
cycle_cadence_hz = 8.33
root_height_p05_m = 0.8533
swing_sole_clearance_mean_left_m = 0.0038
swing_sole_clearance_mean_right_m = 0.0029

run = logs/rsl_rl/kbot_forward_flat/2026-05-15_16-51-22_H9.1.17
checkpoint = model_697.pt
decision = REJECT
summary = lower speed reward preserved geometry and slightly improved right clearance/drift, but still worsened cadence and overspeed
speed_tracking_ratio = 1.5306
approved_step_fraction = 0.1106
left_approved_step_fraction = 0.1292
right_approved_step_fraction = 0.0921
step_root_advance_mean_m = 0.0066
cycle_root_advance_mean_m = 0.0138
cycle_cadence_hz = 8.33
root_height_p05_m = 0.8534
swing_sole_clearance_mean_left_m = 0.0038
swing_sole_clearance_mean_right_m = 0.0030
```

S4.2 corrections after the rejected attempts:

```text
1. The policy-visible completed-step gate must recover the known working apv% contract instead of weakening it.
   Use step_advance_margin_reward with min_step_advance = 0.008 m and min_step_duration = 0.07 s.

2. Keep the command minimum above the reward's moving-command threshold.

3. Treat the stricter air-time valid_step_root_advance term as disabled until the robot has real swing clearance.

4. Do not jump from an 8.06 Hz baseline directly to a 5.0 Hz cycle ceiling in S4.2.
   The next S4.2 target starts near the baseline at max_cycle_hz = 7.5, then S4.3 can ramp lower if approved steps and advance are preserved.

5. The 7.5 Hz ceiling alone is also insufficient; it allowed the old overspeed shuffle to continue.
   Do not continue from either S4.2 attempt above.
   The next anti-shuffle change needs to lower speed reward pressure or separate cadence/step approval from velocity tracking more strongly before another continuation.

6. The next S4.2 test is `apv`-dominant: keep the 8 mm / 70 ms approved-step gate, keep max_cycle_hz = 7.5, and reduce forward-speed reward pressure so the policy cannot improve scalar return mainly by overspeed shuffling.

7. The `apv`-dominant test also failed. Reducing positive speed rewards alone did not stop overspeed because the policy can still retain the old locomotion mode and accept the cadence penalty.
   Next S4 work should not keep stacking small S4.2 continuations from these failed branches.
   Reconsider the reward shape around signed cadence/step margin, or introduce a separate branch method that directly penalizes overspeed/chatter events while preserving the model_648 geometry.
```

## Stage 4 Anti-Shuffle Sub-Curriculum

Stage 4 should be a real anti-shuffle curriculum, not a generic gait-quality continuation. The current diagnosis is right, but direct jumps to low cadence targets are too abrupt and let the policy trade one failure mode for another.

Default parent:

```text
run = logs/rsl_rl/kbot_forward_flat/2026-05-09_00-16-45_H9.1
checkpoint = model_648.pt
role = S3/S4 boundary seed
```

Use `model_648.pt` as the default S4 parent unless a newer checkpoint is explicitly selected by video review.

Core S4 rule:

```text
S4 is not speed training.
S4 is not full contact-quality walking.
S4 is not foot-flat polish.
S4 only needs to produce debounced, low-chatter, alternating steps with nonzero root advance.
S5 handles straight contact-quality walking.
```

Evaluator metrics to add or verify:

```text
raw_touchdown_event_rate_hz_left/right
debounced_touchdown_event_rate_hz_left/right
contact_flip_rate_hz_left/right
short_air_fraction_left/right
short_stance_fraction_left/right
debounced_cycle_cadence_hz
debounced_step_root_advance_mean_m
debounced_cycle_root_advance_mean_m
approved_step_fraction_left/right
```

Each sub-stage uses this rule:

```text
Promote only when the primary metric improves by a meaningful percentage, crosses a minimum useful floor, and guard metrics do not regress.
```

Indicator names describe what the sub-stage targets. Promotion is based on measured improvement percentage plus hard floors, not on indicator category alone.

```text
Stage ID:
  S4.2_CHATTER_SUPPRESSION
  Parent: S4_ANTI_SHUFFLE_WALK boundary seed, default model_648.pt
  Task: Isaac-KBot-Forward-Flat-V2_5-S4_2-ChatterSuppression-v0
  Training budget: UNKNOWN
  Purpose: Suppress raw contact chatter while keeping low-speed supported alternating motion alive.
  Config direction:
    vx range 0.06-0.10 m/s so the reward-side moving-command gate stays active.
    Reduce speed reward pressure strongly.
    track_lin_vel_xy_exp weight about 0.5.
    world_forward_velocity_clip weight about 0.25.
    supported_forward_velocity weight about 0.25.
    Strengthen contact_chatter_l1 and action_rate_l2.
    Initial max_cycle_hz about 7.5, close enough to baseline that approved steps should not collapse.
    feet_air_time threshold low/moderate, about 0.10-0.14 s.
    Use step_advance_margin_reward as the policy-visible approved-step gate.
    step_advance_margin_reward min_step_advance = 0.008 m.
    step_advance_margin_reward min_step_duration = 0.07 s.
    valid_step_root_advance stays disabled in S4.2 because the stricter air-time gate was 0% visible from this local minimum.
  Promotion gates:
    raw_touchdown_event_rate_hz and contact_flip_rate_hz improve from parent.
    debounced cadence does not increase.
    approved_step_fraction_left/right do not regress materially from the model_648 baseline.
    root_height_p05_m regresses by <= 0.01 m from parent and remains >= 0.84 preferred / >= 0.82 hard floor.
    debounced_step_root_advance_mean_m and debounced_cycle_root_advance_mean_m do not regress by more than 10%.
    fsep/ksep remain near target and above hard floors.
  Reject gates:
    chatter improves because contact events disappear.
    cadence improves by crouching.
    speed tracking improves while debounced cycle advance remains near zero.
    low_body terminations increase materially.
  Next stage: S4.3_CADENCE_RAMP
  Do-not-optimize warning: Do not optimize speed here. The goal is debounced contacts, not fast walking.

Stage ID:
  S4.3_CADENCE_RAMP
  Parent: S4.2_CHATTER_SUPPRESSION
  Task: Isaac-KBot-Forward-Flat-V2_5-S4_3-CadenceRamp-v0
  Training budget: UNKNOWN
  Purpose: Gradually reduce cadence without suppressing steps or forcing crouch/collapse.
  Config direction:
    Ramp max_cycle_hz gradually: 4.5 -> 3.5 -> 2.75 -> 2.25.
    Do not jump directly to 1.25 Hz.
    Keep vx low, about 0.04-0.10 m/s.
    Preserve height, fsep, ksep, yaw, and lateral gates.
  Promotion gates:
    debounced_cycle_cadence_hz improves by >= 15-20% from parent at each ramp step.
    debounced_cycle_cadence_hz crosses the current ramp floor.
    debounced step/cycle root advance remain within 90% of parent.
    approved steps remain present on both sides.
    root height, fsep, ksep, yaw, and lateral drift stay within S4.2 guardrails.
  Reject gates:
    cadence improves because the robot stops stepping.
    cadence improves by losing height.
    cadence improves but left/right approved-step balance collapses.
  Next stage: S4.4_MINIMUM_REAL_STEP
  Do-not-optimize warning: Cadence reduction is not enough; it must preserve alternating root advance.

Stage ID:
  S4.4_MINIMUM_REAL_STEP
  Parent: S4.3_CADENCE_RAMP
  Task: Isaac-KBot-Forward-Flat-V2_5-S4_4-MinimumRealStep-v0
  Training budget: UNKNOWN
  Purpose: Turn debounced alternating contacts into minimum real steps with nonzero root advance.
  Config direction:
    Use step_advance_margin_reward as the main completed-step shaping term.
    Start min_step_advance = 0.006 m, then 0.008 m.
    min_step_duration = 0.08 s.
    Keep valid_step_root_advance weak or disabled until approved_step_fraction is nonzero.
    dense_single_support_step_progress is a small helper, not the main objective.
  Promotion gates:
    debounced_step_root_advance_mean_m improves by >= 25-40% from parent.
    debounced_cycle_root_advance_mean_m improves by >= 20% from parent.
    approved_step_fraction_left/right improve or remain meaningfully nonzero.
    debounced_cycle_cadence_hz does not climb back toward the shuffle band.
    height, fsep, ksep, yaw, and lateral drift stay within guardrails.
  Reject gates:
    step advance comes from sliding, yaw, lateral drift, or falling forward.
    step advance improves but cycle advance does not.
    only one side produces approved steps.
    longer steps destroy height, fsep, ksep, or L/R symmetry.
  Next stage: S4.5_ANTI_SHUFFLE_PROMOTION
  Do-not-optimize warning: Minimum real steps are still not speed training or polished walking.

Stage ID:
  S4.5_ANTI_SHUFFLE_PROMOTION
  Parent: S4.4_MINIMUM_REAL_STEP
  Task: Isaac-KBot-Forward-Flat-V2_5-S4_5-AntiShufflePromotion-v0
  Training budget: UNKNOWN
  Purpose: Freeze rewards and evaluate the S4 result before passing to S5 straight/contact-quality walking.
  Promotion gates:
    Debounced cadence is lower than model_648.pt and below the current S4 ramp floor.
    Approved steps exist on both sides.
    Debounced step/cycle root advance are positive.
    Height and fsep/ksep do not regress from the selected parent.
    Video shows low-chatter alternating steps, not contact vibration.
  Reject gates:
    Any S4 reject gate is triggered.
    Evaluator improves by suppressing steps rather than debouncing them.
  Next stage: S5_STRAIGHT_CONTACT_QUALITY_WALK
  Do-not-optimize warning: This is a promotion/evaluation stage, not another reward search.
```

## Naming

`max range gait` means the gait that should travel the farthest for a fixed energy budget. For now, the simulation proxy is lowest positive joint mechanical work per meter of forward advance. Later this must include baseline system energy consumption.
