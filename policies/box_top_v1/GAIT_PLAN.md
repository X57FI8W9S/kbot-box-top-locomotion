# KBot Gait Quality Plan

## Goal

Train a forward walking policy with a visibly clean, repeatable gait. The main quality target is not just forward speed. The primary target is low persistent lateral body and hip compensation over several step cycles.

The gait should tolerate short spikes during foot exchange, but it should not settle into a permanent sideways lean or permanent hip roll/yaw offset.

## Primary Acceptance Metrics

Use rolling windows long enough to cover several steps. Start with 3 seconds, and also report 5 seconds for final candidates.

Primary metrics:

- `torso_lateral_tilt_rms`: RMS of `projected_gravity_b[:, 1]` over the rolling window.
- `hip_roll_yaw_pos_rms`: RMS of hip roll and hip yaw joint positions over the rolling window.
- `torso_lateral_tilt_mean`: signed mean of `projected_gravity_b[:, 1]` over the rolling window.
- `hip_roll_yaw_pos_mean_abs`: absolute signed mean per hip roll/yaw joint over the rolling window.

Interpretation:

- RMS catches both oscillation and permanent lean.
- Signed mean catches bias. A gait with a constant lean can have low-looking visual motion but still bad mean tilt.
- Short spikes are acceptable if the rolling mean returns near zero and the RMS does not stay elevated.

Target bands:

- Baseline 30 second torso RMS mean was `0.04046`; the target is 80-90% lower: `0.004-0.008`.
- Baseline 30 second hip roll/yaw RMS mean was `0.10818`; the target is 80-90% lower: `0.011-0.022 rad`.
- Bad: persistent `torso_lateral_tilt_mean` away from zero, even if the robot stays upright.
- Bad: one hip roll/yaw joint holding a constant offset to make the gait work.

The earlier `0.025` torso / `0.08 rad` hip numbers are now only interim milestones, not the final goal.

## Secondary Metrics

Track these for context, but do not let them override the primary gait-quality metrics:

- Forward velocity tracking error.
- Yaw rate RMS.
- World heading error.
- Foot lateral lane error.
- Foot contact duty factor per side.
- Double support fraction.
- Step frequency.
- Stride length.
- Stance foot flatness.
- Foot contact area proxy, if available.
- Action rate and torque cost.

The desired failure mode is a slower but symmetric gait, not a fast gait with permanent torso tilt.

## Current Suspect Reward Clusters

The current reward set has several overlapping constraints that may be boxing the policy into a stable but biased gait.

### Foot Yaw Cluster

Current terms:

- `foot_parallel_l2`
- `foot_world_parallel_l2`
- `foot_world_parallel_max_l2`
- `foot_toe_in_l2`
- `world_heading_l2`
- `yaw_rate_l2`
- `hip_roll_yaw_position_l2`

Risk:

The policy is being told to keep the body heading, feet, and hip yaw all close to neutral/world-forward at the same time. This can prevent natural transient yaw during swing and stance. It may also force compensation through torso lean or foot placement.

Recommendation:

- Disable `foot_world_parallel_l2`.
- Disable `foot_world_parallel_max_l2`.
- Keep `foot_toe_in_l2` only if toe-in is still visually bad.
- Keep `world_heading_l2` and `yaw_rate_l2`, but do not over-weight them while diagnosing gait quality.

### Frontal Plane Cluster

Current terms:

- `leg_frontal_plane_l1`
- `left_leg_frontal_plane_l1`
- `right_leg_frontal_plane_l1`
- `max_leg_frontal_plane_l1`
- `foot_lateral_lane_l1`
- `foot_lateral_lane_max_l1`
- `foot_signed_lateral_clearance_l1`

Risk:

This cluster tries to keep the feet and leg segments in clean lateral lanes, which is useful, but the combination may overconstrain balance. A biped may need small lateral hip/foot adjustments while still averaging to a symmetric gait.

Recommendation:

- Keep foot crossing prevention.
- Keep one foot lane term.
- Reduce or temporarily disable the duplicate leg frontal plane terms.
- Reintroduce only the terms that improve rolling torso/hip RMS without producing permanent offsets.

### Knee Bend Term

Current term:

- `knee_extension_l1 = -80.0`

Risk:

This is very strong. It may prevent knee locking, but it can also force a permanent crouch or make the policy solve stability through constant knee bend rather than clean stepping.

Recommendation:

- Reduce to `-20.0` or `-30.0` for the next gait-quality branch.
- Keep monitoring knee angles in the HUD.
- Raise it only if locked knees return.

## Proposed Reward Direction

Make persistent lateral bias expensive, but allow transient step-cycle motion.

Add or strengthen terms that penalize rolling-window mean and RMS rather than instantaneous pose only:

- Torso lateral tilt rolling mean penalty.
- Torso lateral tilt rolling RMS penalty.
- Hip roll/yaw rolling mean penalty.
- Hip roll/yaw rolling RMS penalty.

If rolling-window reward terms are awkward inside the manager reward API, start by logging these metrics during training and using them for checkpoint selection. Then add reward terms once the metric implementation is stable.

Important distinction:

- Do not punish every instantaneous lateral movement too strongly.
- Punish lateral movement that stays biased over several steps.

## Gait Phase Scaffold

Add a simple phase signal to the policy:

- `sin(phase)`
- `cos(phase)`

Use a nominal step frequency, then allow it to vary later. Start with a conservative frequency that matches slow walking.

Add light contact schedule rewards:

- Left stance while right swings for half the phase.
- Right stance while left swings for the other half.
- Penalize both feet airborne.
- Penalize permanent double support, but allow short double support during transitions.

This should help the policy discover a real alternating gait instead of solving all constraints through leaning, dragging, or shuffling.

Keep the phase scaffold light. It should guide stepping rhythm, not force a brittle animation.

## Experiment Sequence

### Branch A: Deconstrain And Measure

Purpose:

Find out whether the current wall is caused by overconstraint.

Changes:

- Disable `foot_world_parallel_l2`.
- Disable `foot_world_parallel_max_l2`.
- Reduce `knee_extension_l1` from `-80.0` to `-30.0`.
- Keep foot crossing prevention.
- Keep one foot lane term.
- Reduce duplicate leg frontal plane penalties by at least 50%.

Train from `model_10300.pt` for a short run.

Accept if:

- Torso rolling RMS improves or stays similar.
- Hip roll/yaw RMS improves.
- Foot contact looks flatter or less edge-loaded.
- No major toe-in or crossing regression.

Reject if:

- The robot uses large hip yaw/roll permanently.
- Feet cross again.
- Forward tracking collapses.

### Branch B: Slower Clean Gait

Purpose:

Get symmetry and low persistent tilt before asking for speed.

Changes:

- Command range: `0.35-0.55 m/s`.
- Keep Branch A deconstraint changes.
- Lower the forward-speed floor to match the slower command range.
- Shorten sagittal step-length and swing-overtake targets so slow walking is not punished for smaller steps.
- Train from a stable checkpoint, possibly `model_10300.pt`, but expect some adaptation.

Accept if:

- Rolling torso tilt mean returns near zero over multiple windows.
- Hip roll/yaw mean does not sit at a permanent offset.
- Alternating contacts are visible and stable.

Then gradually ramp command speed:

- `0.45-0.65`
- `0.55-0.75`
- `0.65-0.85`
- `0.75-0.95`

Only advance speed when rolling torso and hip metrics stay within acceptable bands.

### Branch C: Phase-Guided Gait

Purpose:

Introduce an explicit stepping rhythm if Branch A/B still plateau.

Changes:

- Add `sin(phase), cos(phase)` observations.
- Add light alternating contact schedule rewards.
- Keep primary acceptance metrics unchanged.

Accept if:

- Contact timing becomes more regular.
- Torso tilt RMS drops without increasing hip roll/yaw RMS.
- The policy does not become a forced marching animation with poor recovery.

### Branch D: Persistent Bias Rewards

Purpose:

The reward wall is now clear: instantaneous torso and hip penalties improved the gait, but they plateaued far above the 80-90% target. The next branch must directly punish the thing we care about: persistent rolling bias.

Changes:

- Add `root_lateral_tilt_ema_l2` with a multi-step EMA.
- Add `hip_roll_yaw_position_ema_l2` with a multi-step EMA.
- Keep instantaneous torso and hip penalties active.
- Keep the slower command range and 30 second videos.

Accept if:

- Torso RMS mean moves below the Branch D continuation value, `0.02419`, with no collapse.
- Hip roll/yaw RMS mean moves below `0.09646`.
- The signed mean terms drop, not just the oscillation.

Reject if:

- The robot stands, shuffles, or uses a brittle forced phase.
- Forward tracking collapses.
- Foot contact area gets worse because the policy avoids hip motion by tipping the feet.

## Checkpoint Selection Rules

Do not choose checkpoints by scalar reward alone.

For each candidate checkpoint:

1. Render a 30 second trailing HUD video for routine iteration.
2. Compute rolling 3 second and 5 second metrics.
3. Inspect at least 3 windows:
   - Early after startup.
   - Middle steady walking.
   - Late rollout.
4. Reject any checkpoint with persistent torso lean, even if speed tracking is better.
5. Reject any checkpoint with permanent hip roll/yaw offsets, unless it is clearly transient and recovers.

Preferred checkpoint profile:

- Slightly slower is acceptable.
- Small momentary tilt spikes are acceptable.
- Permanent lean is not acceptable.
- Permanent hip roll/yaw compensation is not acceptable.
- Feet should land with a large contact surface, not on edges or with chronic toe-in.

## Current Status And Plan Update

Best current torso checkpoint:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-28-53/model_11195.pt
```

Best balanced checkpoint:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-25-25/model_11096.pt
```

Measured 30 second results:

```text
model_11096 Branch D:
  torso RMS mean 0.02505, p95 0.03328
  torso mean-bias mean 0.01423
  hip roll/yaw RMS mean 0.09609, p95 0.09725
  hip roll/yaw mean-abs mean 0.08422

model_11195 Branch D continuation:
  torso RMS mean 0.02419, p95 0.02955
  torso mean-bias mean 0.01315
  hip roll/yaw RMS mean 0.09646, p95 0.09778
  hip roll/yaw mean-abs mean 0.08373
```

Decision:

Plain continuation is still lowering torso RMS, but hip roll/yaw RMS is plateaued around `0.096 rad`. Do not keep spending long runs on the same reward balance unless checkpoint selection shows a clear hip improvement.

Next branch:

- Keep Branch D rewards as the base.
- Keep the light phase/rhythm scaffold, but do not trust it alone.
- Add explicit EMA rewards for torso lateral tilt and hip roll/yaw joint offsets.
- Keep 30 second videos for routine iteration.
- Only return to speed ramping after hip roll/yaw RMS breaks below the current `~0.096 rad` plateau.

## Immediate Next Step

Do not continue Branch D blindly. The torso metric is still improving, but hip roll/yaw RMS has stalled.

Recommended next patch:

- Train the EMA-bias branch from `model_11294.pt` if phase rollout metrics are later good, otherwise from `model_11195_phase44.pt` or the original `model_11195.pt` with a compatible checkpoint.
- Render 30 second videos and compare against `model_11195.pt`.
- If EMA improves signed bias but not RMS, increase the EMA weights before changing foot-lane rewards again.

Accept the next branch only if it lowers both torso and hip roll/yaw metrics without losing stable forward walking.
