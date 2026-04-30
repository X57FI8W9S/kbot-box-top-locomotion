# KBot Gait Progress Report

## Current Goal

Improve gait quality with primary focus on removing persistent torso lateral tilt and persistent hip roll/yaw offsets over multi-step rolling windows. Short spikes during step exchange are acceptable; sustained bias is not.

Clarified target: the desired reduction is 80-90% lower than baseline, not just a small incremental improvement.

```text
Baseline torso RMS mean:       0.04046 -> target 0.004-0.008
Baseline hip roll/yaw RMS mean: 0.10818 -> target 0.011-0.022 rad
```

## Baseline

Current reference checkpoint:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_01-57-15/model_10300.pt
```

Reference videos:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_01-57-15/videos/play/trailing-hud-model_10300-60s.mp4
logs/rsl_rl/kbot_forward_flat/2026-04-27_01-57-15/videos/play/trailing-hud-model_10300-30s.mp4
```

The HUD now shows paired left/right joint positions by joint type, plus 3 second rolling `torso rms` and `hip ry rms`.

## Plan Document

Detailed plan:

```text
GAIT_PLAN.md
```

## Current Branch: Branch A

Purpose:

Test whether gait quality is blocked by overconstrained yaw/frontal-plane rewards.

Config changes applied:

- `foot_world_parallel_l2.weight`: `-6.0 -> 0.0`
- `foot_world_parallel_max_l2.weight`: `-3.0 -> 0.0`
- `knee_extension_l1.weight`: `-80.0 -> -30.0`
- `leg_frontal_plane_l1.weight`: `-14.0 -> -7.0`
- `left_leg_frontal_plane_l1.weight`: `-4.0 -> -2.0`
- `right_leg_frontal_plane_l1.weight`: `-4.0 -> -2.0`
- `max_leg_frontal_plane_l1.weight`: `-16.0 -> -8.0`

Files changed:

```text
source/kbot_loco/kbot_loco/tasks/locomotion/env_cfg.py
scripts/rsl_rl/play_trailing.py
GAIT_PLAN.md
PROGRESS_REPORT.md
```

Training run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_03-46-31
```

Saved checkpoints:

```text
model_10350.pt
model_10400.pt
model_10449.pt
```

First candidate video:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_03-46-31/videos/play/trailing-hud-model_10449-60s.mp4
```

Quick sampled-frame note:

- Around frame 180, HUD showed `torso rms ~= 0.040` and `hip ry rms ~= 0.111`.
- This is not an obvious improvement over the baseline sampled frame.
- Do not conclude from one frame; review several rolling windows in the full video.

30 second metric comparison:

```text
baseline model_10300:
  torso RMS mean 0.04046, p95 0.04706
  torso mean-bias mean 0.03291
  hip roll/yaw RMS mean 0.10818, p95 0.10998

Branch A model_10449:
  torso RMS mean 0.04026, p95 0.04869
  torso mean-bias mean 0.03212
  hip roll/yaw RMS mean 0.11007, p95 0.11176
```

Result:

Branch A is neutral to slightly worse for the primary target metrics. It does not justify continuing in the same direction by itself.

## Current Branch: Branch B

Purpose:

Slow the task down so symmetry and low persistent torso/hip bias can improve before ramping speed again.

Additional config changes applied after Branch A:

- `lin_vel_x`: `(0.75, 0.95) -> (0.35, 0.55)`
- `forward_velocity_below_l2.minimum_velocity`: `0.68 -> 0.30`
- `foot_sagittal_separation_l1.target_length`: `0.32 -> 0.20`
- `swing_foot_overtake_l1.target_length`: `0.24 -> 0.16`

Training run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-06-27
```

Candidate:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-06-27/model_10599.pt
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-06-27/videos/play/trailing-hud-model_10599-30s.mp4
outputs/gait_metrics/branch_b_model_10599_30s.json
```

30 second metric result:

```text
Branch B model_10599:
  torso RMS mean 0.03736, p95 0.04333
  torso mean-bias mean 0.02913
  hip roll/yaw RMS mean 0.10641, p95 0.10785
  hip roll/yaw mean-abs mean 0.09177
```

Result:

Branch B is a small improvement but not enough. It lowers speed and slightly lowers torso/hip metrics, but the persistent torso bias remains.

## Current Branch: Branch C

Purpose:

Keep the slower command range and directly push down the two primary metrics.

Additional config changes after Branch B:

- `root_lateral_tilt_l2.weight`: `-24.0 -> -60.0`
- `hip_roll_yaw_position_l2.weight`: `-1.5 -> -6.0`

Training run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-11-47
```

Candidate:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-11-47/model_10798.pt
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-11-47/videos/play/trailing-hud-model_10798-30s.mp4
outputs/gait_metrics/branch_c_model_10798_30s.json
```

30 second metric result:

```text
Branch C model_10798:
  torso RMS mean 0.02886, p95 0.03702
  torso mean-bias mean 0.01924
  hip roll/yaw RMS mean 0.10263, p95 0.10400
  hip roll/yaw mean-abs mean 0.08899
```

Result:

Branch C is the first clear improvement. Continue this direction before adding phase scaffolding.

## Current Branch: Branch C Continuation

Purpose:

Continue from `model_10798.pt` with the same slower command range and stronger torso/hip penalties to see whether the primary metrics keep falling without another reward change.

Training run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-15-45
```

Candidate:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-15-45/model_10997.pt
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-15-45/videos/play/trailing-hud-model_10997-30s.mp4
outputs/gait_metrics/branch_c_cont_model_10997_30s.json
```

30 second metric result:

```text
Branch C continuation model_10997:
  torso RMS mean 0.02677, p95 0.03726
  torso mean-bias mean 0.01218
  hip roll/yaw RMS mean 0.09561, p95 0.09727
  hip roll/yaw mean-abs mean 0.08335
```

Result:

Accepted as the first sub-0.10 hip roll/yaw RMS checkpoint. Training was noisy, so later checkpoints still need rollout-based selection.

## Branch C Continuation 2

Purpose:

Continue from `model_10997.pt` without changing rewards.

Training run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-21-18
```

Candidate:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-21-18/model_11096.pt
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-21-18/videos/play/trailing-hud-model_11096-30s.mp4
outputs/gait_metrics/branch_c_cont2_model_11096_30s.json
```

30 second metric result:

```text
Branch C continuation 2 model_11096:
  torso RMS mean 0.02583, p95 0.03222
  torso mean-bias mean 0.01381
  hip roll/yaw RMS mean 0.09861, p95 0.10036
  hip roll/yaw mean-abs mean 0.08642
```

Result:

Mixed. Torso improved, but hip roll/yaw regressed. Do not treat plain continuation as enough.

## Branch D

Purpose:

Keep the torso improvement while reducing hip roll/yaw offsets by easing lateral foot-lane pressure and increasing direct hip neutral pressure.

Additional config changes after Branch C:

- `foot_signed_lateral_clearance_l1.weight`: `-24.0 -> -20.0`
- `foot_lateral_lane_l1.weight`: `-10.0 -> -7.0`
- `foot_lateral_lane_max_l1.weight`: `-8.0 -> -5.0`
- `hip_roll_yaw_position_l2.weight`: `-6.0 -> -9.0`

Training run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-25-25
```

Candidate:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-25-25/model_11096.pt
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-25-25/videos/play/trailing-hud-model_11096-30s.mp4
outputs/gait_metrics/branch_d_model_11096_30s.json
```

30 second metric result:

```text
Branch D model_11096:
  torso RMS mean 0.02505, p95 0.03328
  torso mean-bias mean 0.01423
  hip roll/yaw RMS mean 0.09609, p95 0.09725
  hip roll/yaw mean-abs mean 0.08422
```

Result:

Best balanced checkpoint so far. It keeps the torso improvement and recovers most of the hip regression from plain continuation.

## Branch D Continuation

Purpose:

Check whether Branch D keeps improving with one more short continuation.

Training run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-28-53
```

Candidate:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-28-53/model_11195.pt
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-28-53/videos/play/trailing-hud-model_11195-30s.mp4
outputs/gait_metrics/branch_d_cont_model_11195_30s.json
```

30 second metric result:

```text
Branch D continuation model_11195:
  torso RMS mean 0.02419, p95 0.02955
  torso mean-bias mean 0.01315
  hip roll/yaw RMS mean 0.09646, p95 0.09778
  hip roll/yaw mean-abs mean 0.08373
```

Result:

Best torso checkpoint so far, but hip roll/yaw RMS is plateaued around `0.096 rad`. Next change should not be another plain continuation; use phase/rhythm scaffolding or a hip-specific rolling/bias term.

## Branch E: Phase Warm-Start

Purpose:

Add a light gait phase scaffold instead of continuing the same reward balance blindly.

Code/config changes:

- Added `gait_phase` observation: `sin(phase), cos(phase)`.
- Added `alternating_foot_phase` contact reward.
- Padded `model_11195.pt` from 42 to 44 observation inputs as `model_11195_phase44.pt`.

Runs:

```text
fresh phase run: logs/rsl_rl/kbot_forward_flat/2026-04-27_11-09-21
warm-start phase run: logs/rsl_rl/kbot_forward_flat/2026-04-27_11-12-07
```

Result:

- Fresh phase training collapsed early and was stopped.
- Warm-start training finished and produced `model_11294.pt`.
- 30 second rollout evaluation could not complete because Isaac/PhysX exhausted GPU memory, then `nvidia-smi` stopped communicating with the driver in this session.
- The partial video file from the failed render is invalid (`44` bytes) and should be ignored.
- Training reward did not collapse, but live reward terms still showed hip roll/yaw penalty near the same plateau, so this branch is not accepted until rollout metrics prove otherwise.

Candidate to evaluate after GPU recovery:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_11-12-07/model_11294.pt
outputs/gait_metrics/branch_e_phase_model_11294_30s.json
```

## Branch F: EMA Persistent-Bias Rewards

Purpose:

The plateau suggests instantaneous penalties are not enough. Add rewards that directly penalize persistent torso tilt and persistent hip roll/yaw offsets over a multi-step EMA.

Code/config changes:

- Added `mdp.root_lateral_tilt_ema_l2(tau_s=1.5)`, weight `-300.0`.
- Added `mdp.joint_position_ema_l2(tau_s=1.5)` for hip roll/yaw, weight `-24.0`.
- Kept the phase scaffold and the existing instantaneous penalties.

Status:

- Code compiles.
- Training/evaluation is blocked until the GPU driver recovers.

Next command once GPU is usable:

```text
.venv/bin/python scripts/rsl_rl/train.py --task Isaac-KBot-Forward-Flat-v0 --headless --num_envs 2048 --resume --load_run 2026-04-27_11-12-07 --checkpoint model_11294.pt --max_iterations 100
```

## Branch G: Straight-Posture Ramp 1

Purpose:

Keep training the current policy, but increase straight walking and neutral posture pressure gradually instead of restarting or applying a large one-shot reward change.

Clarification:

- `track_lin_vel_xy_exp` and `track_ang_vel_z_exp` are exponential bell rewards with `std` parameters.
- Hip roll/yaw and torso lateral-tilt posture rewards are L2 penalties, not Gaussian bells.
- Hip roll/yaw metrics are joint radians.
- Torso lateral tilt uses `projected_gravity_b[:, 1]`; near zero it is approximately roll radians.

Code/config changes:

- Increased lateral velocity penalty: `-5.0 -> -7.0`.
- Increased yaw-rate penalty: `-5.0 -> -7.0`.
- Increased root lateral tilt penalty: `-60.0 -> -90.0`.
- Increased root lateral tilt EMA penalty: `-300.0 -> -450.0`.
- Increased world heading penalty: `-24.0 -> -32.0`.
- Increased hip roll/yaw instantaneous penalty: `-9.0 -> -12.0`.
- Increased hip roll/yaw EMA penalty: `-24.0 -> -36.0`.
- Reduced lane/leg frontal-plane tolerances:
  - foot lateral lane: `0.04 -> 0.03`
  - foot lateral lane max: `0.025 -> 0.02`
  - leg frontal plane: `0.04 -> 0.03`
  - left/right leg frontal plane: `0.02 -> 0.015`
  - max leg frontal plane: `0.015 -> 0.01`

Runs:

```text
ramp 1: logs/rsl_rl/kbot_forward_flat/2026-04-27_17-42-57/model_11393.pt
ramp 1 continuation: logs/rsl_rl/kbot_forward_flat/2026-04-27_17-46-04/model_11492.pt
```

Videos and metrics:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_17-42-57/videos/play/trailing-hud-model_11393-30s.mp4
outputs/gait_metrics/branch_g_ramp1_model_11393_30s.json

logs/rsl_rl/kbot_forward_flat/2026-04-27_17-46-04/videos/play/trailing-hud-model_11492-30s.mp4
outputs/gait_metrics/branch_g_ramp1_cont_model_11492_30s.json
```

30 second metric result:

```text
Branch E warm-start model_11294:
  speed mean 0.41700, command mean 0.41824
  yaw-rate mean -0.00681, p95 0.04034, max 0.45274
  torso RMS mean 0.02054, p95 0.02668
  hip roll/yaw RMS mean 0.09683, p95 0.09878

Branch G ramp 1 model_11393:
  speed mean 0.42063, command mean 0.41824
  yaw-rate mean 0.01343, p95 0.08522, max 0.20426
  torso RMS mean 0.01959, p95 0.02584
  hip roll/yaw RMS mean 0.09193, p95 0.09361

Branch G ramp 1 continuation model_11492:
  speed mean 0.39958, command mean 0.41824
  yaw-rate mean 0.00139, p95 0.08088, max 0.43302
  torso RMS mean 0.02087, p95 0.02676
  hip roll/yaw RMS mean 0.08850, p95 0.09026
```

Result:

Ramp 1 moved hip roll/yaw in the right direction (`0.09683 -> 0.09193 -> 0.08850 rad`) without causing falls. Torso improved at `model_11393` but regressed slightly by `model_11492`. Yaw-rate mean improved by `model_11492`, but yaw p95 remains worse than `model_11294`, and speed tracking fell below command. Do not tighten further yet; inspect the videos and prefer `model_11393` if visual straightness is acceptable, otherwise use `model_11492` only if the hip improvement is worth the speed/yaw tradeoff.

## Branch H: Sole Contact Cleanup

Purpose:

Keep `model_11492.pt` as the warm start, but target the visible tiptoe/edge-contact issue from the side screenshots without changing the gait schedule or lateral lane balance.

Code/config changes:

- Changed `foot_flat_l2` and `stance_foot_flat_l2` from `square(1 - abs(up_z))` to `1 - up_z^2`.
- This keeps the same reward weights but makes moderate foot pitch/roll visible to the optimizer instead of nearly disappearing.
- Added `torso avg` to the rollout HUD alongside `torso rms`, so persistent bias and oscillation are separate.
- Added a synchronized 90-degree side-view HUD video output from `scripts/rsl_rl/play_trailing.py`.

Run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/model_11791.pt
```

Resume command:

```text
.venv/bin/python scripts/rsl_rl/train.py --task Isaac-KBot-Forward-Flat-v0 --headless --num_envs 1024 --max_iterations 300 --resume --load_run 2026-04-27_17-46-04 --checkpoint model_11492.pt
```

Videos and metrics:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/videos/play/trailing-hud-model_11791.mp4
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/videos/play/trailing-hud-model_11791-side.mp4
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/videos/play/trailing-hud-model_11791-metrics.json
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/videos/play/trailing-side-hud-model_11791.mp4
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/videos/play/trailing-side-hud-model_11791-metrics.json
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/videos/play/trailing-side-hud-model_11791-16x9.mp4
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/videos/play/trailing-side-hud-model_11791-16x9-metrics.json
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/videos/play/trailing-side-hud-model_11791-final.mp4
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/videos/play/trailing-side-hud-model_11791-final.metrics.json
```

Playback script correction:

- `play_trailing.py` now extends playback episode length to exceed the requested video length, so rollouts no longer reset every 8 seconds when using the training task id.
- The preferred inspection video is now one side-by-side file: trailing view on the left and 90-degree side view on the right.
- The corrected preferred file is `trailing-side-hud-model_11791-final.mp4`: `1280x720`, 60 seconds, one compact HUD overlay, each camera view composed into an `8:9` half-frame.
- The final HUD restores the paired L/R rolling-average joint-position columns, keeps `torso rms`, `torso avg`, and `hip ry rms`, and uses a shorter top strip with tighter line spacing.
- The final camera defaults use a 2.18 m distance and a horizontal look-at around knee/mid-body height; the side view targets the hip/root link laterally instead of looking ahead of the robot.

60 second metric result:

```text
Branch H model_11791:
  speed mean 0.39986, command mean 0.41960
  yaw-rate mean 0.00568, p95 0.12706, max 0.29180
  torso avg mean -0.00087, p95 0.00456, final 0.00131
  torso RMS mean 0.01888, p95 0.02296
  hip roll/yaw RMS mean 0.08566, p95 0.08764
```

Result:

Training remained stable through full 8 second episodes and hip roll/yaw RMS improved again (`0.08850 -> 0.08566 rad`). Torso signed bias is essentially zero on the 60 second rollout, while torso RMS remains around `0.019`, so the remaining torso motion appears more oscillatory than biased. The sole-contact reward terms are now strong in scalar value (`stance_foot_flat_l2` around `-1.13` near the end), so select this branch by visual inspection of the new side video before increasing foot-flatness further.

## Branch I: Final Sole-Contact Push

Purpose:

Last short continuation before restarting with a new policy. Keep the Branch H gait and posture settings, but bias more strongly toward full-sole stance contact.

Code/config changes:

- Reduced `feet_air_time.weight`: `2.25 -> 1.75`.
- Increased `foot_flat_l2.weight`: `-0.2 -> -0.35`.
- Increased `stance_foot_flat_l2.weight`: `-1.5 -> -2.5`.
- Left torso/hip posture rewards unchanged.

Run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-29_07-58-47/model_11990.pt
```

Resume command:

```text
.venv/bin/python scripts/rsl_rl/train.py --task Isaac-KBot-Forward-Flat-v0 --headless --num_envs 1024 --max_iterations 200 --resume --load_run 2026-04-29_06-29-05 --checkpoint model_11791.pt
```

Video and metrics:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-29_07-58-47/videos/play/trailing-side-hud-model_11990-30s.mp4
logs/rsl_rl/kbot_forward_flat/2026-04-29_07-58-47/videos/play/trailing-side-hud-model_11990-30s.metrics.json
```

30 second metric result:

```text
Branch I model_11990:
  speed mean 0.42284, command mean 0.43620
  yaw-rate mean -0.00835, p95 0.10869, max 0.28420
  torso avg mean -0.00122, p95 0.01127, final 0.00027
  torso RMS mean 0.02125, p95 0.02641
  hip roll/yaw RMS mean 0.08552, p95 0.08741
```

Result:

Training stayed stable for the full 200-iteration continuation: mean episode length remained 400, timeout stayed 1.0, and termination penalty stayed zero. Compared with Branch H's 60 second rollout, hip roll/yaw RMS is effectively unchanged, torso RMS is slightly higher on this 30 second sample, and speed tracking is a little better. Use the 30 second side video for visual selection: if sole contact is visibly improved, keep `model_11990.pt`; otherwise `model_11791.pt` remains the safer checkpoint.
