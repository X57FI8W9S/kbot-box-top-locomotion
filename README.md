# KBot RSL-RL Forward Walking

This workspace contains an external Isaac Lab task package for training the `assets/robot/usd/kbot_box_top3.usd` robot with PPO through RSL-RL.

The current task is a flat-ground, forward-only bootstrap for the asymmetric KBot biped. It was tuned until the robot could stay upright through the full 8 second horizon, move forward continuously, reduce yaw drift, avoid the crossed-feet gait seen in earlier videos, and begin ramping from a clean `0.4 m/s` walk toward faster `1.0-1.5 m/s` walking.

## Current Best Policy

Current quality-consolidation candidate run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_01-57-15
```

Recommended checkpoint:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_01-57-15/model_10300.pt
```

This checkpoint continues from the toe-in branch and narrows the command range to `0.75-0.95 m/s` for gait-quality consolidation instead of pushing more speed. It is the best scalar tradeoff from the latest run: low-body stays at zero, forward/yaw tracking improve, root lateral tilt improves, and wobble improves, while foot-lane and sagittal-plane metrics remain in the same band.

The current reward weights include `foot_lateral_lane_l1 = -10`, `foot_lateral_lane_max_l1 = -8`, `leg_frontal_plane_l1 = -14`, `left_leg_frontal_plane_l1 = -4`, `right_leg_frontal_plane_l1 = -4`, `max_leg_frontal_plane_l1 = -16`, `root_lateral_tilt_l2 = -24`, and neutral-position pressure on hip roll and hip yaw. The foot-lane terms only look at body-frame Y, i.e. horizontal/coronal placement, using left/right targets of `+0.12 m` and `-0.12 m` with no X/Z distance component. The hip roll and hip yaw defaults are all `0.0` in `assets.py`, so the joint-position reward is anchored to neutral roll/yaw.

Around `model_10300.pt`, the scalar metrics were: forward velocity error about `0.1294`, yaw error about `0.0976`, low-body `0.0`, weighted root lateral-tilt penalty about `-0.0475`, weighted yaw-rate penalty about `-0.0780`, weighted foot-lane max penalty about `-0.0624`, weighted aggregate leg-plane penalty about `-0.0957`, weighted max leg-plane penalty about `-0.0693`, weighted toe-in penalty about `-0.0718`, wobble joint velocity about `-0.1317`, feet air-time about `0.0831`, and base-height penalty about `-0.1732`.

The later checkpoint `2026-04-27_01-57-15/model_10349.pt` improves a few individual terms, especially swing-foot overtake and wobble, but it gives back forward/yaw tracking and step-separation quality. Use `model_10300.pt` as the current default unless visual inspection strongly favors `model_10349.pt`. `2026-04-27_01-50-43/model_10050.pt` is the previous balanced fallback. `2026-04-26_20-29-08/model_9250.pt` remains the pre-toe-in foot-lane fallback.

Exported policy:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-26_20-29-08/exported/policy.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_20-29-08/exported/policy.onnx
```

Recorded headless rollout:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-26_20-29-08/videos/play/rl-video-step-0.mp4
```

Trailing HUD rollouts:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-26_20-29-08/videos/play/trailing-hud-model_9250-lowcam.mp4
logs/rsl_rl/kbot_forward_flat/2026-04-26_21-11-00/videos/play/trailing-hud-model_9450-lowcam.mp4
logs/rsl_rl/kbot_forward_flat/2026-04-26_21-22-06/videos/play/trailing-hud-model_9500-lowcam.mp4
logs/rsl_rl/kbot_forward_flat/2026-04-26_21-22-06/videos/play/trailing-hud-model_9500-wide-avg.mp4
logs/rsl_rl/kbot_forward_flat/2026-04-26_21-22-06/videos/play/trailing-hud-model_9500-wide-avg-smoothcam.mp4
logs/rsl_rl/kbot_forward_flat/2026-04-27_01-57-15/videos/play/trailing-hud-model_10300-60s.mp4
```

Previous useful checkpoints:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-26_20-29-08/model_9449.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_20-29-08/model_9250.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_20-29-08/model_9050.pt
logs/rsl_rl/kbot_forward_flat/2026-04-27_01-57-15/model_10300.pt
logs/rsl_rl/kbot_forward_flat/2026-04-27_01-50-43/model_10050.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_21-22-06/model_9500.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_21-11-00/model_9450.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_19-50-15/model_8950.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_19-50-15/model_9099.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_19-50-15/model_8900.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_19-42-47/model_8700.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_19-42-47/model_8895.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_19-34-50/model_8396.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_19-34-50/model_7950.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_19-12-04/model_7797.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_19-12-04/model_7600.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_18-35-51/model_7198.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_17-54-49/model_6599.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_04-50-38/model_6000.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_04-38-23/model_5449.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_03-53-29/model_4850.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_03-42-10/model_4300.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_03-27-17/model_4150.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_02-52-06/model_3600.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_02-45-50/model_3148.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_02-15-34/model_2649.pt
logs/rsl_rl/kbot_forward_flat/2026-04-26_01-38-53/model_2250.pt
logs/rsl_rl/kbot_forward_flat/2026-04-25_21-42-41/model_1549.pt
```

Do not use `logs/rsl_rl/kbot_forward_flat/2026-04-26_02-10-12` as the main candidate. That branch over-weighted foot-flatness and degraded yaw/height.
Do not use `logs/rsl_rl/kbot_forward_flat/2026-04-26_03-21-49` as the main candidate. That branch jumped straight to `0.36-0.44 m/s` and used a constant step-length penalty, which improved air-time but degraded heading and height.

If the play command uses `--headless`, nothing appears on screen. It records to the MP4 path above instead.

## What Changed

The main files changed during the successful training pass are:

```text
source/kbot_loco/kbot_loco/tasks/locomotion/env_cfg.py
source/kbot_loco/kbot_loco/tasks/locomotion/mdp.py
scripts/probe_kbot_stability.py
scripts/rsl_rl/play_trailing.py
```

Important task settings now in `env_cfg.py`:

- Forward command only: `lin_vel_x = (0.75, 0.95)`, `lin_vel_y = 0`, `ang_vel_z = 0`.
- Zero heading command is active: `heading_command = True`, `heading = (0.0, 0.0)`.
- Episode length is 8 seconds.
- Hard fall/body terminations are disabled for training; timeout is the only active termination.
- Low body height is punished with reward terms instead of immediately ending the episode.
- The current branch uses stronger world-heading, yaw-rate, signed foot-clearance, foot world-parallel, forward-speed-floor, action-rate, swing-only step-length, directional swing-foot overtake, stance foot-flatness, and selected-joint wobble shaping.
- The current branch also uses root lateral-tilt, aggregate leg frontal-plane, per-side leg frontal-plane, worst-segment leg frontal-plane, body-frame Y-only foot-lane placement, foot world-parallel max, direct foot toe-in, and hip-roll/hip-yaw position shaping to reduce side-lean, diagonal leg motion, and inward toe yaw.

Important helper reward functions now in `mdp.py`:

- `lateral_velocity_l2`
- `yaw_rate_l2`
- `root_lateral_tilt_l2`
- `world_heading_l2`
- `backward_velocity_l2`
- `forward_velocity_below_l2`
- `root_height_below_l2`
- `knee_extension_l1`
- `foot_lateral_spacing_l1`
- `foot_signed_lateral_clearance_l1`
- `foot_lateral_lane_l1`
- `foot_lateral_lane_max_l1`
- `leg_frontal_plane_l1`
- `leg_frontal_plane_side_l1`
- `leg_frontal_plane_max_l1`
- `foot_sagittal_separation_l1`
- `swing_foot_overtake_l1`
- `foot_parallel_l2`
- `foot_world_parallel_l2`
- `foot_world_parallel_max_l2`
- `foot_toe_in_l2`
- `foot_flat_l2`
- `stance_foot_flat_l2`
- `joint_velocity_l2`
- `joint_position_l2`

## Install

Use the local Isaac Sim/Isaac Lab Python environment:

```bash
.venv/bin/python -m pip install -e source/kbot_loco
```

## Train

Resume from the current best checkpoint:

```bash
.venv/bin/python scripts/rsl_rl/train.py \
  --task Isaac-KBot-Forward-Flat-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 300 \
  --resume \
  --load_run 2026-04-27_01-57-15 \
  --checkpoint model_10300.pt
```

Fresh training:

```bash
.venv/bin/python scripts/rsl_rl/train.py \
  --task Isaac-KBot-Forward-Flat-v0 \
  --headless \
  --num_envs 1024
```

Quick smoke run:

```bash
.venv/bin/python scripts/rsl_rl/train.py \
  --task Isaac-KBot-Forward-Flat-v0 \
  --headless \
  --num_envs 32 \
  --max_iterations 2
```

## Play And Export

Headless play, video recording, and policy export for the current quality-consolidation checkpoint:

```bash
.venv/bin/python scripts/rsl_rl/play.py \
  --task Isaac-KBot-Forward-Flat-Play-v0 \
  --headless \
  --video \
  --video_length 3000 \
  --num_envs 16 \
  --checkpoint logs/rsl_rl/kbot_forward_flat/2026-04-27_01-57-15/model_10300.pt
```

Live viewport play:

```bash
.venv/bin/python scripts/rsl_rl/play.py \
  --task Isaac-KBot-Forward-Flat-Play-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/kbot_forward_flat/2026-04-27_01-57-15/model_10300.pt
```

Trailing follow-camera playback with speed/yaw and gait-diagnostic HUD:

```bash
.venv/bin/python scripts/rsl_rl/play_trailing.py \
  --task Isaac-KBot-Forward-Flat-Play-v0 \
  --headless \
  --num_envs 1 \
  --video_length 3000 \
  --checkpoint logs/rsl_rl/kbot_forward_flat/2026-04-27_01-57-15/model_10300.pt \
  --output logs/rsl_rl/kbot_forward_flat/2026-04-27_01-57-15/videos/play/trailing-hud-model_10300-60s.mp4
```

`scripts/rsl_rl/play_trailing.py` defaults to a 60 second video at 50 Hz (`3000` steps) and a wider trailing camera (`2.25 m` behind, `0.20 m` above root) so the full robot fits in frame. The play environment also uses a 60 second episode horizon, so the recording should not reset every 8 seconds. The HUD reports 3 second rolling averages for speed, command speed, yaw rate, every joint position, torso lateral-tilt RMS, and hip roll/yaw RMS. The numeric speed/yaw columns are fixed so signs do not move the values around. The camera trailing direction is also a 3 second rolling average by default; tune it with `--camera_window_s`.

If Isaac Sim prompts for permissions on startup, accept them. The GPU must be visible to the process; `nvidia-smi` should show the RTX 4060 before launching.

## Training Handoff Notes

The successful route was:

1. Probe showed the zero-action pose falls/collapses, and long post-fall rollouts gave poor reward signal.
2. A short 3 second bootstrap episode learned to stand and move forward by about iteration 1000.
3. Training resumed from that checkpoint with an 8 second horizon.
4. Zero-heading command and straight-line penalties improved heading, producing `model_1549.pt`.
5. A continuation run from `model_1549.pt` produced `2026-04-26_01-38-53/model_2250.pt`, which solved the original straight-line request.
6. A first heavy gait-shaping branch, `2026-04-26_02-10-12`, over-weighted foot-flatness and degraded yaw/height.
7. A corrected softer gait-shaping branch, `2026-04-26_02-15-34`, produced `model_2649.pt`, but the user still observed crossed feet and 45 degree turning.
8. A heading and foot-ordering branch, `2026-04-26_02-45-50`, produced `model_3148.pt`, reducing yaw error to about `0.184` while keeping full episodes.
9. A stronger fine-tune branch, `2026-04-26_02-52-06`, produced `model_3600.pt`, reducing yaw error to about `0.138` with low-body and signed foot-crossing penalties near zero.
10. A direct high-speed branch, `2026-04-26_03-21-49`, tried `0.36-0.44 m/s` and a constant step-length penalty. It recovered but degraded heading/height, so it is not a main candidate.
11. A corrected speed branch, `2026-04-26_03-27-17`, uses `0.32-0.42 m/s` plus swing-only step-length and wobble penalties. It produced the current recommended `model_4150.pt`.
12. A higher-speed ramp, `2026-04-26_03-42-10`, uses `0.55-0.85 m/s`, more air-time pressure, a `0.28 m` swing-only step-length target, a `0.48 m/s` forward-speed floor, and stance foot-flatness shaping. It produced the current recommended `model_4300.pt`.
13. A faster ramp, `2026-04-26_03-53-29`, uses `0.8-1.1 m/s`, a `0.32 m` swing-only step-length target, a `0.72 m/s` forward-speed floor, a more realistic `0.45 s` air-time threshold, and stronger foot-parallel/stance-flat shaping. It produced the current recommended `model_4850.pt`.
14. A gait-symmetry branch, `2026-04-26_04-38-23`, adds `swing_foot_overtake_l1`, which penalizes the airborne foot if it does not pass the stance foot before landing. It produced the current recommended `model_5449.pt`.
15. A stricter gait-symmetry branch, `2026-04-26_04-50-38`, increases `swing_foot_overtake_l1` from weight `-12` to `-14` and target overtake from `0.20 m` to `0.24 m`. It produced the current recommended `model_6000.pt`.
16. A rear-view leg-plane branch, `2026-04-26_17-54-49`, adds `leg_frontal_plane_l1`, which penalizes the shin and foot when they move laterally away from their own hip in the robot body frame. It produced the current recommended `model_6599.pt`.
17. A per-leg sagittal-plane branch, `2026-04-26_18-35-51`, tightens `leg_frontal_plane_l1`, adds `root_lateral_tilt_l2`, and adds `hip_roll_position_l2`. It produced the current recommended `model_7198.pt`.
18. A strict per-leg sagittal-plane branch, `2026-04-26_19-12-04`, further tightens `leg_frontal_plane_l1`, doubles the root lateral-tilt weight, and anchors both hip roll and hip yaw to neutral. It produced the current recommended `model_7797.pt`.
19. A max sagittal-plane branch, `2026-04-26_19-34-50`, adds left-leg, right-leg, and max-segment leg-plane terms. It showed that the aggregate metric could improve while a visible worst segment remained off-plane.
20. A balanced max sagittal-plane branch, `2026-04-26_19-42-47`, keeps the new max term but restores forward tracking, heading, height, and signed foot clearance pressure. It produced the current recommended `model_8700.pt`.
21. A final max sagittal-plane continuation, `2026-04-26_19-50-15`, continues from `model_8700.pt`. It improved heading and forward tracking while keeping max leg-plane error in the same low band. It produced the current recommended `model_8950.pt`.
22. A foot-lane branch, `2026-04-26_20-29-08`, adds body-frame Y-only foot placement rewards targeting the canonical left/right lanes at `+0.12 m` and `-0.12 m`. It produced the current recommended `model_9250.pt`.
23. A strict toe-forward comparison branch, `2026-04-26_21-11-00`, adds `foot_world_parallel_max_l2`. It improved the toe-yaw scalar but worsened lane/height enough that `model_9450.pt` is only a comparison checkpoint.
24. A gentler toe-in branch, `2026-04-26_21-22-06`, adds `foot_toe_in_l2`, restores `foot_world_parallel_l2` to `-6`, and uses a gentler max-world-parallel weight of `-3`. It produced `model_9500.pt`, the best toe-in comparison so far.
25. A quality-consolidation branch, `2026-04-27_01-50-43`, narrows the command range to `0.75-0.95 m/s` and continues from `model_9500.pt`. It improved yaw, torso lateral tilt, leg-plane metrics, and wobble through `model_10050.pt`; later checkpoints mostly traded terms.
26. A short continuation, `2026-04-27_01-57-15`, resumed from `model_10050.pt` to test for further improvement. `model_10300.pt` is the best balanced checkpoint; the final `model_10349.pt` improves a few individual terms but gives back too much forward/yaw/step-separation quality.

Current quality:

- Upright through the full 8 second training horizon, with low-body at zero for the current recommended checkpoint.
- Current command band is `0.75-0.95 m/s`, intentionally centered around the already-good `0.8 m/s` gait rather than pushing speed.
- Current recommended checkpoint is `2026-04-27_01-57-15/model_10300.pt`.
- `model_10300.pt` reached forward error about `0.1294`, yaw error about `0.0976`, root lateral tilt about `-0.0475`, yaw-rate about `-0.0780`, foot-lane max about `-0.0624`, max leg-plane about `-0.0693`, toe-in about `-0.0718`, wobble about `-0.1317`, and feet air-time about `0.0831`.
- `2026-04-27_01-50-43/model_10050.pt` is the previous balanced fallback. `2026-04-26_20-29-08/model_9250.pt` is the pre-toe-in foot-lane fallback.

Recommended next steps:

- Use `2026-04-27_01-57-15/model_10300.pt` as the current default and visually compare it only if a later scalar tradeoff clearly improves the whole gait bundle.
- Continue short quality-consolidation resumes of about `200-300` iterations from `model_10300.pt`; stop when two or three saved checkpoints trade terms instead of improving forward tracking, yaw, root lateral tilt, foot-lane max, max leg-plane, toe-in, wobble, and air-time together.
- If the gait still leans or toes inward in rear-view playback, add a direct pelvis/hip-link lateral alignment term or contact-aware foot placement term instead of only increasing existing penalties.
- Do not jump straight to `1.0-1.5 m/s` from this branch; `0.75-0.95 m/s` is the current quality band. Push speed again only after the gait looks consistently natural at this range.
- Reintroduce a fall termination for evaluation or later training only after the gait is visibly acceptable.

## Latest Continuation Command

The latest quality-consolidation continuation command was:

```bash
.venv/bin/python scripts/rsl_rl/train.py \
  --task Isaac-KBot-Forward-Flat-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 300 \
  --resume \
  --load_run 2026-04-27_01-57-15 \
  --checkpoint model_10300.pt
```

The latest completed quality-consolidation run created:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_01-57-15/model_10300.pt
```

The Codex tool permission procedure is called an escalation request. The successful `play.py --video` export needed `sandbox_permissions="require_escalated"` once so Isaac Sim could access the NVIDIA GPU/display permissions. The approved prefix is:

```text
.venv/bin/python scripts/rsl_rl/play.py
```

## Notes About This Workspace

This directory currently is not a Git worktree. `git status` from this path reports that no `.git` directory exists. There are Git repos in sibling directories, but this copy itself is just a working folder.
