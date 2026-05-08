# Box-Top V2 Bootstrap Report

Date: 2026-05-07

## Purpose

This report answers one narrow question: across the scratch restarts and short branches in this project, what got the robot to useful early walking in the fewest iterations, and what is the fastest repeatable way to bootstrap a new run now?

The important distinction is:

- **Anti-fall bootstrap**: the policy learns to keep the body up for the episode. This is necessary but is not walking.
- **Early walking**: the policy produces credible forward motion from the upright bootstrap, even if lateral drift, contact quality, or symmetry still fail.
- **Keeper gait**: the evaluator passes speed, yaw, lateral drift, body posture, contact, and foot behavior gates.

## Executive Summary

The fastest reliable path found so far is not direct full-V2 PPO from iteration zero. It is a staged V1-style bootstrap:

1. Train a short 3 s timeout-only anti-fall bootstrap from scratch.
2. Continue from that checkpoint into a gentler gait-shaping stage.
3. Only after the robot is upright and moving, turn on the stricter full V2 gait/symmetry/lane terms.

The shortest reliable anti-fall bootstrap was:

- `logs/rsl_rl/kbot_forward_flat/2026-05-04_07-41-32/model_1298.pt`
- Scratch training, about 1298 policy iterations.
- 3 s episodes, timeout-only, `alive=+5`, base-height and low-body shaping, knee-extension shaping, low command range.
- Result: upright recovery with positive reward, but not walking. Evaluator speed ratio was only about `0.049`, distance was about `0.308 m` over 30 s, and double support was about `0.987`.

The earliest V2 walking-ish checkpoint after a scratch restart was the continuation from that anti-fall bootstrap:

- Bootstrap: `model_1298.pt`
- Gait continuation: `logs/rsl_rl/kbot_forward_flat/2026-05-04_16-40-24/model_1797.pt`
- Total from scratch lineage: about `1797` iterations.
- Result: speed ratio about `1.134` and yaw passed, but lateral drift and gait-quality gates still failed.

The best V2 checkpoint in this scratch-bootstrap lineage was later:

- `logs/rsl_rl/kbot_forward_flat/2026-05-04_19-18-03/model_2795.pt`
- Total from scratch lineage: about `2795` iterations.
- Result: speed, yaw, lateral, root-roll, and root-height gates passed. Hip-roll/contact/airborne behavior still failed, so it was not a final keeper.

Direct full V2 from iteration zero was tested after the left/right frontal-plane weights were equalized. It still failed:

- `logs/rsl_rl/kbot_forward_flat/2026-05-06_23-49-21.../model_350.pt`
- Started at learning iteration zero with no checkpoint.
- By iteration 349-363 it was still low/collapsed: low-body losses near `-130` to `-117`, base-height losses near `-51` to `-47`.
- PPO then hit NaN in the value loss / Normal std path.

Conclusion: the left/right weight asymmetry was real and should stay fixed, but it was not the only bootstrap blocker. Full V2 is too hard at iteration zero for this robot/reset/reward stack.

## Scratch Restart Results

| Attempt | Iterations | Result | Lesson |
| --- | ---: | --- | --- |
| Direct V2 scratch, posture-first | stopped early | Timeout survivor in a low body posture. Base-height and low-body losses stayed huge. | Dense full task allowed collapsed timeout survival. |
| Scratch hard low-body termination | 99 | Low-body termination saturated, mean episode length about 26-27 steps. | Hard cutoffs removed the exploit but also removed recoverable learning signal. |
| Scratch standing-first | 99 | Timeout fraction 1.0, but still collapsed. | Stronger standing terms alone did not create a stable bootstrap. |
| Scratch conservative standing-first | 99 | Smaller action scale/noise still collapsed. | Exploration size was not the main blocker. |
| Passive reset probes | diagnostic only | Default/root-height/bent-knee/ankle-compensated poses all fell passively. | The reset pose is not a passive standing solution. |
| Balance-only branch | 99 | Rejected, high terminations, short episodes. | Zero-command balance alone was still too brittle. |
| V1 reward reproduction with inherited 8 s episode | 299 | Timeout survivor, low-body still bad. | V1 weights were not sufficient if the episode length stayed too long. |
| Static V1-derived pose bootstrap | short branch | Rejected after ankle clamp; low-body timeout survivor. | A static pose by itself did not replace the dynamic bootstrap. |
| Exact V1-style 3 s bootstrap | 1298 | Successful anti-fall bootstrap. Not walking. | Short episodes plus soft survival/posture shaping were the first reliable base. |
| Anti-fall bootstrap plus gait continuation | 1797 | First V2 walking-ish checkpoint: speed/yaw good, lateral/contact bad. | The fastest walking came from continuing the anti-fall policy, not restarting full V2. |
| Later staged continuation | 2795 | Best V2 gait so far, several evaluator gates passed. | Gait quality improves when strict terms are introduced after movement exists. |
| Full V2 from zero after L/R equalization | 350-363 | Collapsed and PPO NaN. | Equalizing asymmetry helped correctness, but did not make full V2 bootstrappable. |

## Short Branch Analysis

The short branching runs were useful because they separated several possible explanations:

- **Wrong reward weights alone**: not sufficient. V1-like weights failed when the episode length was wrong.
- **Too much action noise**: not sufficient. Conservative action scale and low PPO noise still collapsed.
- **Need harsher terminations**: not sufficient. Early hard low-body termination saturated and shortened episodes before the policy found recovery.
- **Need a better static pose**: maybe helpful, but not sufficient by itself. The V1-derived pose branch still fell into a bad timeout survivor.
- **Need exact staged curriculum**: yes. The first reliable recovery came when the V1-style bootstrap was reproduced with the short 3 s episode and soft survival/posture shaping.

The branch analysis suggested the shortest path to early walking:

1. Keep the first stage short and forgiving.
2. Reward upright survival before enforcing gait aesthetics.
3. Do not use full V2 lateral/hip/foot/contact constraints from iteration zero.
4. Do not rely on hard fall termination as the first learning signal.
5. Move into gait shaping only after low-body and base-height losses are near normal and reward is positive.

## Fastest Known Bootstrap Recipe

Use this as the current least-time guide for a new restart.

### Stage 0: Sanity Checks Before Training

Before spending GPU time, verify:

- Left/right reward weights are symmetric unless intentionally asymmetric.
- Body order and foot links are correct.
- Foot/sole local offsets have the expected signs.
- The reset pose has no accidental roll/yaw bias.
- A zero-action rollout is understood. It does not need to stand passively, but if it immediately folds left/right, inspect geometry and joint signs first.

The `left_leg_frontal_plane_l1=-3.0`, `right_leg_frontal_plane_l1=-5.0` asymmetry was suspicious and should remain equalized. The corrected full-V2 weights should stay `left=-4.0`, `right=-4.0` unless there is a measured hardware reason not to.

### Stage 1: Anti-Fall Bootstrap From Iteration Zero

Use the V1-style bootstrap task:

```bash
.venv/bin/python scripts/rsl_rl/train.py \
  --task Isaac-KBot-Forward-Flat-V2-Scratch-V1Bootstrap-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 1300 \
  --run_name v2_bootstrap_from_zero
```

The key config traits are:

- `episode_length_s = 3.0`
- timeout-only survival, with low-body and bad-orientation hard terminations disabled
- command x range about `0.10` to `0.25`
- heading command disabled
- `alive = +5`
- `base_height_l2 = -15`, target height about `0.78`
- `low_body_l2 = -30`, minimum body height about `0.45`
- `knee_extension_l1 = -80`
- modest velocity tracking and action-rate penalties
- most strict V2 foot-lane, hip-bias, wobble, frontal-plane, and phase constraints disabled

Expected checkpoints:

- By about 300 iterations: if it is still behaving like the failed 8 s reproduction, check that the episode length is really 3 s.
- By about 700-900 iterations: low-body and base-height terms should be materially improving.
- By about 1298-1300 iterations: reward should be positive and low-body/base-height losses should be small enough that the robot is a usable upright seed.

Do not call this walking. Treat it as the base checkpoint.

### Stage 2: Gentle Gait Transition

Continue from the anti-fall checkpoint into a gait-shaping stage. Historically, the first walking-ish V2 result appeared after about 500 continuation iterations, around `model_1797` in the scratch lineage.

The transition stage should:

- Keep survival/posture support active.
- Increase forward velocity tracking gradually.
- Add foot alternation and air-time terms gradually.
- Keep lateral drift and yaw terms active but not dominant.
- Keep strict hip-roll, frontal-plane, lane-center, and contact aesthetics weaker until movement exists.
- Evaluate every few hundred iterations with the same headless diagnostic script and the trailing video pass.

Stop early if the run shows collapsed timeout survival again. Continuing a collapsed branch wastes more time than restarting from the known anti-fall recipe.

### Stage 3: Full V2 Gait Shaping

Only switch into full V2 or near-full V2 once the policy is already upright and advancing.

Target signs before escalating:

- forward distance clearly nonzero over 30 s
- speed tracking ratio near useful range, even if imperfect
- yaw not diverging
- lateral drift bounded
- low-body and base-height losses no longer dominating

The best current evidence says full V2 from zero is not the shortest path. Full V2 is a refinement stage, not the bootstrap stage.

## What Not To Do

Avoid these if the goal is least time to early walking:

- Do not start full V2 directly from iteration zero.
- Do not use the 8 s episode length for the first bootstrap.
- Do not make hard low-body termination the first major learning signal.
- Do not assume a static handcrafted pose replaces the anti-fall curriculum.
- Do not copy one fixed action from a good policy and expect it to stand.
- Do not keep training a branch where low-body/base-height losses remain huge after the expected bootstrap window.

## Handcrafted Starter Pose

Yes, the handcrafted starter pose is worth trying.

It should be treated as a candidate reset pose or curriculum input, not as proof that the robot can stand. The earlier static pose branch failed, but a better handcrafted pose could still shorten Stage 1 if it reduces the amount of early recovery the policy has to discover.

Please provide it as an explicit joint map in radians, plus root pose:

- root height
- root orientation, if non-neutral
- `left_hip_pitch_04`
- `right_hip_pitch_04`
- `left_hip_roll_03`
- `right_hip_roll_03`
- `left_hip_yaw_03`
- `right_hip_yaw_03`
- `left_knee_04`
- `right_knee_04`
- `left_ankle_02`
- `right_ankle_02`

Also include any intended reset noise/tolerance if you have it.

The first thing to do with the pose is not a long training run. The fastest validation is:

1. Run a zero-action/passive probe.
2. Check root roll, root height, and left/right body heights during the first second.
3. Compare sole and knee positions for left/right symmetry.
4. If it is less unstable than the current reset, use it in the 3 s anti-fall bootstrap.
5. If it has a strong left/right fold, fix the pose or geometry before training.

## Current Recommendation

For the current restart, use the equalized left/right weights, but do not restart full V2 from zero again as the main path. Restart policy training from iteration zero with the 3 s V1-style anti-fall bootstrap, save the 1298/1299 checkpoint, then branch into a gentle gait-transition stage.

If the handcrafted starter pose is available now, test it before the long bootstrap. If it passes the quick symmetry/passive checks better than the current reset, include it in the new from-zero bootstrap. If it does not, proceed with the known bootstrap recipe and keep the pose out of the main run.

## 2026-05-07 V2.2 Restart Result

The old handcrafted pose was not used in the main run. It was treated as a clue only because it came from different actuator assumptions and its hip-pitch signs did not match the current mirror convention.

### Stage 1: Known Anti-Fall Bootstrap

Run:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-07_00-36-13_v2_2_bootstrap_from_zero_known_recipe
```

Command:

```bash
.venv/bin/python scripts/rsl_rl/train.py \
  --task Isaac-KBot-Forward-Flat-V2-Scratch-V1Bootstrap-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 1300 \
  --run_name v2_2_bootstrap_from_zero_known_recipe
```

Result:

- Completed from true policy iteration zero.
- Training time about `554 s`.
- Final checkpoint: `model_1299.pt`.
- Final logged iteration `1299/1300`: mean reward about `+8.99`, base-height loss about `-0.363`, low-body loss about `-0.203`, timeout fraction `1.0`.
- This is a valid anti-fall seed, not a final walking policy.

### Stage 2: Gentle Gait Transition

A new transition task was added:

```text
Isaac-KBot-Forward-Flat-V2-GaitTransition-v0
```

It inherits the successful bootstrap task and only adds moderate gait pressure: higher forward tracking, small alternating-foot reward, gentle lateral/yaw penalties, weak foot spacing/separation/flatness terms, and continued survival/posture support. It intentionally does not enable full V2 lane, sole-plane, frontal-plane, hip-bias, or long-window roll penalties yet.

Run:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-07_01-01-30_v2_2_gait_transition_from_1299
```

Command:

```bash
.venv/bin/python scripts/rsl_rl/train.py \
  --task Isaac-KBot-Forward-Flat-V2-GaitTransition-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 500 \
  --resume \
  --load_run 2026-05-07_00-36-13_v2_2_bootstrap_from_zero_known_recipe \
  --checkpoint model_1299.pt \
  --run_name v2_2_gait_transition_from_1299
```

Result:

- Continued from bootstrap learning iteration `1299` to `1798`.
- Training time about `231 s`.
- Final logged iteration `1798/1799`: mean reward about `+8.22`, speed error about `0.076`, yaw error about `0.264`, base-height loss about `-0.135`, low-body loss about `-0.102`, timeout fraction `1.0`.

Headless diagnostic:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-07_01-01-30_v2_2_gait_transition_from_1299/diagnostics/model_1798_headless
```

Evaluator decision: `REJECT`.

Important metrics:

- speed tracking: `PASS`, ratio about `0.997`
- alternating steps: `PASS`, step count `91`
- root height: `PASS`, mean about `0.782 m`, p05 about `0.761 m`
- yaw drift: `FAIL`, about `-10.74 rad/m`
- lateral drift: `FAIL`, about `-10.55 m/m`
- root roll mean: `FAIL`, about `0.027 rad`
- hip roll mean: `FAIL`, about `0.093 rad`
- airborne: `FAIL`, airborne fraction about `0.034`
- contact quality is still poor: double support about `0.787`, full-support fractions `0.0`, edge-walk proxies above `0.85`

Interpretation:

- This reproduces the historical shortest path: anti-fall around `1299`, early walking-ish around `1798`.
- It is better than direct full V2 from zero because it preserves upright posture and learns forward speed.
- It is not ready for full keeper status. The next stage should target yaw/lateral drift and foot contact quality without destroying the anti-fall seed.

### Stage 3: Yaw/Lateral Transition

A focused continuation task was added:

```text
Isaac-KBot-Forward-Flat-V2-YawLateralTransition-v0
```

It inherits the gentle gait transition and increases yaw, lateral velocity, world-heading, root-lateral-tilt, foot-lane, and hip-roll/yaw penalties. This stage is intended to straighten the early gait without immediately switching to full V2.

First continuation:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-07_01-24-17_v2_2_yaw_lateral_transition_from_1798
```

Result:

- Continued from `model_1798.pt` to `model_2297.pt`.
- Training time about `240 s`.
- Final logged training metrics: speed error about `0.061`, yaw error about `0.172`, timeout fraction `1.0`.
- Better late training metrics appeared around `2293-2296`, but only every-50 checkpoints plus the final checkpoint were saved.

Headless diagnostic for `model_2297.pt`:

- Evaluator decision: `REJECT`.
- speed tracking: `PASS`, ratio about `1.003`
- yaw drift: `PASS`, about `0.061 rad/m`
- lateral drift: `FAIL`, about `0.255 m/m`
- root roll mean: `PASS`, about `-0.0036 rad`
- hip roll mean: `FAIL`, about `0.033 rad`
- alternating steps: `PASS`, step count `112`
- airborne: `FAIL`, airborne fraction about `0.030`
- root height: `PASS`, mean about `0.774 m`, p05 about `0.741 m`

Interpretation: this stage fixed the catastrophic yaw problem from `model_1798`, but still had too much lateral drift and toe/edge contact behavior.

Second continuation:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-07_01-30-59_v2_2_yaw_lateral_transition_from_2297
```

Result:

- Continued from `model_2297.pt` to `model_2796.pt`.
- Training time about `234 s`.
- Final logged training metrics: speed error about `0.047`, yaw error about `0.078`, timeout fraction `1.0`, mean reward about `21.46`.

Headless diagnostic for `model_2796.pt`:

- Evaluator decision: `REJECT`.
- speed tracking: `PASS`, ratio about `1.069`
- yaw drift: `PASS`, about `-0.121 rad/m`
- lateral drift: `FAIL`, about `-0.397 m/m`
- root roll mean: `PASS`, about `0.0015 rad`
- hip roll mean: `FAIL`, about `0.055 rad`
- alternating steps: `PASS`, step count `203`
- airborne: `FAIL`, airborne fraction about `0.055`
- root height: `PASS`, mean about `0.780 m`, p05 about `0.765 m`

Video:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-07_01-30-59_v2_2_yaw_lateral_transition_from_2297/videos/play/trailing-hud-model_2796-v2_2-yaw-lateral.mp4
```

Interpretation:

- The second continuation improved training yaw and produced more stepping, but the diagnostic rollout drifted laterally more than `model_2297`.
- It also reduced double support from about `0.713` to `0.551` and increased airborne fraction from about `0.030` to `0.055`, so it may be a better motion seed even though the lateral diagnostic gate worsened.
- Do not blindly keep pushing this exact stage forever. The next useful branch should inspect the video, then either add a stronger explicit lateral-position/lane correction or branch from `model_2297.pt` with a slightly more conservative hip-roll/lateral bias correction.

### Stage 4: Lateral Cleanup

After reviewing `orange6.jpg`, the failure looked less like a left/right offset and more like lateral wandering with narrow edge/toe contacts. A new cleanup task was added:

```text
Isaac-KBot-Forward-Flat-V2-LateralCleanup-v0
```

This stage inherits the yaw/lateral transition and adds an explicit root lateral-position penalty relative to each environment origin, plus modestly stronger foot-lane, foot-spacing, foot-flat, stance-flat, and hip-roll/yaw penalties.

Run:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-07_02-05-02_v2_2_lateral_cleanup_from_2796
```

Result:

- Continued from `model_2796.pt` to `model_3195.pt`.
- Training time about `191 s`.
- The reward was initially shocked negative, but recovered by about iteration `2905`.
- Final logged training metrics: speed error about `0.048`, yaw error about `0.061`, timeout fraction `1.0`, no low-body loss, no termination penalty.

Headless diagnostic for `model_3195.pt`:

- Evaluator decision: `REVIEW_VIDEO`.
- speed tracking: `PASS`, ratio about `0.941`
- yaw drift: `PASS`, about `-0.082 rad/m`
- lateral drift: `PASS`, about `-0.114 m/m`
- root roll mean: `PASS`, about `-0.010 rad`
- hip roll mean: `FAIL`, about `0.042 rad`
- alternating steps: `PASS`, step count `371`
- airborne: `FAIL`, airborne fraction about `0.131`
- root height: `PASS`, mean about `0.780 m`, p05 about `0.768 m`
- double support improved to about `0.097`
- stance sole tilt improved to about `0.481`

Video:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-07_02-05-02_v2_2_lateral_cleanup_from_2796/videos/play/trailing-hud-model_3195-v2_2-lateral-cleanup.mp4
```

Interpretation:

- This is the first V2.2 checkpoint in the current restart chain to pass the evaluator's lateral drift gate.
- The remaining failures are hip-roll mean and airborne/contact quality.
- The next branch should be visual-led. If the video confirms the gait is usable, continue from `model_3195.pt` with a softer hip-roll/contact cleanup. If the video shows very short rapid steps or edge walking, do not increase air-time blindly; reduce cadence pressure and improve foot contact first.

### Stage 5: Fine-Tune Probe From `model_3195`

Two normal resume branches from `model_3195.pt` degraded quickly because RSL-RL restored the old optimizer state along with actor/critic weights. A wrapper flag was added:

```text
--policy_only_resume
```

When used with `scripts/rsl_rl/train.py`, it loads actor and critic weights but does not restore the optimizer state. A fine-tune PPO config was also added:

```text
Isaac-KBot-Forward-Flat-V2-LateralCleanup-FineTune-v0
```

The useful fine-tune run is:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-07_02-53-20_v2_2_lateral_cleanup_policy_only_finetune_from_3195
```

Result:

- Continued from `model_3195.pt` with policy-only resume and low learning rate.
- Full `model_3344.pt` slowed too much and failed the speed gate.
- Early checkpoint `model_3200.pt` is the better fine-tune candidate.

Headless diagnostic for `model_3200.pt`:

- Evaluator decision: `REVIEW_VIDEO`.
- speed tracking: `PASS`, ratio about `0.855`
- yaw drift: `PASS`, about `-0.078 rad/m`
- lateral drift: `PASS`, about `-0.045 m/m`
- root roll mean: `PASS`, about `-0.007 rad`
- hip roll mean: `FAIL`, about `0.036 rad`
- alternating steps: `PASS`, step count `358`
- airborne: `FAIL`, airborne fraction about `0.129`
- root height: `PASS`, mean about `0.782 m`

Video with fixed gait table columns and corrected raw `sep` display:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-07_02-53-20_v2_2_lateral_cleanup_policy_only_finetune_from_3195/videos/play/trailing-hud-model_3200-v2_2-finetune-hudfix-rawsep.mp4
```

In this render, HUD `sep` is `abs(left_sole_y - right_sole_y)` in root/body coordinates, averaged over the same five-cycle HUD window used by most of the overlay. The final JSON reports `final_hud_sep_m = 0.255`.

Interpretation:

- `model_3200.pt` is slightly better than `model_3195.pt` on hip-roll mean and lateral drift, but worse on speed and still not a keeper.
- Further PPO updates from this lineage tend to slow the policy or increase hip roll. Treat `model_3200.pt` as the current fine-tune candidate, but keep `model_3195.pt` as the safer baseline.
- The next attempt should probably use a branch specifically aimed at step length/contact geometry rather than more generic continuation.
