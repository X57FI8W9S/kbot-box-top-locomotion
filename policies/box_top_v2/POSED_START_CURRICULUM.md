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
  Task: Isaac-KBot-Forward-Flat-V2_5-PoseGaitQuality-v0 currently, but likely needs a better config.
  Training budget: UNKNOWN
  Purpose: Remove high-frequency contact chatter and micro-step exploits.
  Promotion gates:
    max per-foot cycle cadence gate exists.
    step_root_advance_mean_m gate exists.
    cycle_root_advance_mean_m gate exists.
    lateral drift gate tightened.
    Current intended cadence trend: about 0.5-1.25 Hz per-foot cycle cadence at the slow stage.
  Reject gates:
    max_cycle_cadence_hz too high.
    step_root_advance_mean_m too low.
    cycle_root_advance_mean_m too low.
    lateral drift too high.
  Next stage: S5_STRAIGHT_CONTACT_QUALITY_WALK
  Do-not-optimize warning: Do not reward or accept speed tracking achieved by high-frequency shuffling.

Stage ID:
  S5_STRAIGHT_CONTACT_QUALITY_WALK
  Parent: S4_ANTI_SHUFFLE_WALK
  Task: Isaac-KBot-Forward-Flat-V2_5-PoseGaitQuality-v0 currently, but likely needs explicit sub-stage configs.
  Training budget: UNKNOWN
  Purpose: Convert the posed-start shuffle into stable straight walking before speed expansion.
  Promotion gates:
    Sub-stage gates must pass in order; see Stage 5 Sub-Curriculum.
  Reject gates:
    Do not promote if cadence improves by lowering the body into collapse.
    Do not promote if speed improves while cycle_root_advance_mean_m remains near zero.
    Do not promote if valid_step_root_advance reward remains unreachable.
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
run = logs/rsl_rl/kbot_forward_flat/2026-05-09_00-07-27_v2_5_pose_forward_width_heading_guard_from_zero_fsep_ksep
checkpoint = model_349.pt
decision = APPROVE
```

Current conservative active seed for first-step / anti-shuffle work:

```text
Stage: S3_FIRST_SUPPORTED_STEPS / S4_ANTI_SHUFFLE_WALK boundary
run = logs/rsl_rl/kbot_forward_flat/2026-05-09_00-16-45_v2_5_pose_gait_quality_from_v2_5_349_fsep_ksep
checkpoint = model_648.pt
decision = APPROVE
status = conservative active seed, not final gait
```

Known rejects that define S4:

```text
run = logs/rsl_rl/kbot_forward_flat/2026-05-09_01-21-36_v2_5_gait_quality_continue_from_947
checkpoint = model_1246.pt
decision = REJECT after tightened walk gates
reason = high cadence, tiny root advance, lateral/yaw regression

run = logs/rsl_rl/kbot_forward_flat/2026-05-09_02-36-29_v2_5_walk_only_contact_quality_from_648
checkpoint = model_947.pt
decision = REJECT
reason = high cadence, tiny root advance, poor speed tracking
```

Current Stage 5 evidence:

```text
run = logs/rsl_rl/kbot_forward_flat/2026-05-15_08-43-54_v2_5_step_gate_007_from_806
checkpoint = model_885.pt
decision = REVIEW_VIDEO
summary = best current normal-speed posed-start baseline before the latest continuations
speed_tracking_ratio = 0.7647
step_root_advance_mean_m = 0.0065
cycle_root_advance_mean_m = 0.0133
cycle_cadence_hz = 6.74
root_height_p05_m = 0.8502

run = logs/rsl_rl/kbot_forward_flat/2026-05-15_14-30-16_v2_5_step_gate_007_normal_continue_from_885
checkpoint = model_964.pt
decision = REVIEW_VIDEO
summary = cadence improved, but step advance and speed regressed; usable as evidence, not a promotion
speed_tracking_ratio = 0.6924
step_root_advance_mean_m = 0.0055
cycle_root_advance_mean_m = 0.0129
cycle_cadence_hz = 5.89
root_height_p05_m = 0.8479

run = logs/rsl_rl/kbot_forward_flat/2026-05-15_14-36-39_v2_5_step_gate_007_normal_continue_from_964
checkpoint = model_1043.pt
decision = REVIEW_VIDEO
summary = cadence and step advance improved, but speed overshot and height margin degraded; not a clean continuation seed
speed_tracking_ratio = 1.5025
step_root_advance_mean_m = 0.0091
cycle_root_advance_mean_m = 0.0018
cycle_cadence_hz = 4.68
root_height_p05_m = 0.8125
```

## Stage 5 Sub-Curriculum

Stage 5 is too large to train as one undifferentiated gait-quality phase. The current policy can trade high cadence for lower height, overspeed, or near-zero cycle advance. Split Stage 5 into explicit sub-stages before entering speed training:

```text
Stage ID:
  S5A_CADENCE_CHATTER_REDUCTION
  Parent: S4_ANTI_SHUFFLE_WALK
  Task: Isaac-KBot-Forward-Flat-V2_5-PoseGaitQuality-v0 or a derived cadence-only config.
  Training budget: UNKNOWN
  Purpose: Lower contact chatter and per-foot cycle cadence while preserving tall posed-start geometry.
  Promotion gates:
    cycle_cadence_hz trends downward.
    root_height_p05_m stays above the posed-start safety floor.
    fsep/ksep stay near target.
    low_body terminations do not increase materially.
  Reject gates:
    lower cadence caused by crouch/collapse.
    speed overshoot used to hide bad step mechanics.
    cycle_root_advance_mean_m remains near zero after cadence improves.
  Next stage: S5B_VALID_STEP_GATE_ACCESS
  Do-not-optimize warning: Do not reward speed here. The goal is slower, cleaner contact timing.

Stage ID:
  S5B_VALID_STEP_GATE_ACCESS
  Parent: S5A_CADENCE_CHATTER_REDUCTION
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: Make the policy actually see the valid-step reward often enough to learn from it.
  Promotion gates:
    approved/valid step percentage reaches a useful training signal range.
    valid_step_root_advance reward is nonzero in training diagnostics.
    step_root_advance_mean_m improves without losing height/width.
  Reject gates:
    valid-step percentage remains too sparse.
    dense progress reward improves but gated valid-step reward remains zero.
  Next stage: S5C_STEP_LENGTH_RAMP
  Do-not-optimize warning: A sparse gate that never fires is a monitor, not a training signal.

Stage ID:
  S5C_STEP_LENGTH_RAMP
  Parent: S5B_VALID_STEP_GATE_ACCESS
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: Raise root advance per step toward a speed-scaled walking target while keeping cadence in walking range.
  Promotion gates:
    step_root_advance_mean_m and cycle_root_advance_mean_m increase together.
    cycle_cadence_hz remains within the current S5 target band.
    no knee/foot crossing, fsep/ksep stay near target.
  Reject gates:
    step length comes from sliding, yawing, lateral drift, or falling forward.
    step_root_advance improves but cycle_root_advance does not.
  Next stage: S5D_STRAIGHTNESS_AND_SUPPORT_QUALITY
  Do-not-optimize warning: Do not chase long steps before the robot can repeat valid alternating contacts.

Stage ID:
  S5D_STRAIGHTNESS_AND_SUPPORT_QUALITY
  Parent: S5C_STEP_LENGTH_RAMP
  Task: UNKNOWN
  Training budget: UNKNOWN
  Purpose: Clean yaw drift, lateral drift, foot flatness, support fractions, and edge-walk proxies after real steps exist.
  Promotion gates:
    y distance trends toward zero.
    x distance tracks command over the 30 s rollout.
    support percentages and edge-walk proxies are acceptable.
    J/m is tracked but not optimized yet.
  Reject gates:
    straight-line metrics improve by returning to in-place shuffling.
    support quality improves only by suppressing steps.
  Next stage: S6_SPEED_RANGE_RAMP_WALK
  Do-not-optimize warning: Energy and speed are diagnostics in S5D, not primary objectives.
```

## Naming

`max range gait` means the gait that should travel the farthest for a fixed energy budget. For now, the simulation proxy is lowest positive joint mechanical work per meter of forward advance. Later this must include baseline system energy consumption.
