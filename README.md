# KBot Box-Top Locomotion

This repository is for developing a reusable Isaac Lab + RSL-RL training procedure for a simplified biped with a box-top upper body. The box replaces torso/arms so locomotion training can be developed before moving to a richer humanoid model.

## Directory Map

```text
assets/                  Shared robot and environment assets.
source/                  Shared Isaac Lab task package currently used by training scripts.
scripts/                 Shared runnable tools: train, play, video/HUD, probes.
policies/box_top_v1/     Completed first-policy reports, history, and notes.
policies/box_top_v2/     New-policy prompt, design notes, and future v2-specific work.
logs/                    Ignored local training outputs, checkpoints, videos.
outputs/                 Ignored Hydra/runtime outputs.
isaac_lab/               Ignored external Isaac Lab checkout.
```

## Current Split

`box_top_v1` is the old policy line. It should be treated as historical context and a source of lessons, not as the place to keep editing new-policy plans.

`box_top_v2` is the restart. Its first goal is to build diagnostics and evaluator tooling before adding another dense reward stack.

`scripts/` remains shared because the training/playback scripts still rely on their current repository-relative paths. In particular, `scripts/rsl_rl/play_trailing.py` is the shared side-by-side HUD video tool.

`source/kbot_loco/.../locomotion` is currently the latest v1 task implementation. When v2 starts modifying task/reward code, fork or clearly rename the task config rather than silently overwriting the v1 baseline.

## Proprioception Walker Gait Reference

This is the general walking-cycle reference for the proprioception-only walker.
It is a target pattern for diagnostics, review videos, and future light gait
shaping. It is not meant to be a hard animation script unless a specific
training stage explicitly adds gait shaping.

Phase is normalized over one full gait cycle, where `0.0` and `1.0` are the
same point in the repeating cycle. The table uses right-foot landing as the
phase-zero convention, but the gait can start from either foot. A left-foot-led
cycle is the same pattern shifted by half a cycle with left and right swapped.

```text
phase       event / support state
0.000       right foot lands
0.000-0.125 double support and weight transfer
0.125       left foot toe-off
0.125-0.500 right single support, left swing
0.500       left foot lands
0.500-0.625 double support and weight transfer
0.625       right foot toe-off
0.625-1.000 left single support, right swing
1.000       next right foot landing; cycle repeats
```

Readable cycle:

```text
right landing -> short double support -> right stance / left swing
left landing  -> short double support -> left stance / right swing
```

This is a walking reference, not a running or trotting reference. There should
be no useful flight phase. Double support is allowed as a short weight-transfer
phase, not as permanent shuffling. During single support, the swing foot should
clear the floor, move ahead of the stance foot, and land without forcing large
torso roll, yaw drift, lateral drift, foot crossing, or loss of support width.

The single-support swing window is `0.375` of a full cycle
(`0.125 -> 0.500` or `0.625 -> 1.000`). If a future gait-shaping term ties
foot advance directly to commanded forward speed, this creates an implicit
average swing-foot advance-rate target of roughly:

```text
average_swing_advance_rate ~= target_forward_speed / 0.375
```

This should be treated as a shaping implication, not as a required current
reward term.

## Important Docs

```text
policies/box_top_v1/FINAL_REPORT.md
policies/box_top_v1/PROGRESS_REPORT.md
policies/box_top_v1/GAIT_PLAN.md
policies/box_top_v2/prompts/new_policy_prompt.txt
policies/box_top_v2/design/diagnostics_plan.md
policies/box_top_v2/FINAL_REPORT_DRAFT.md
```

## What To Avoid

- Do not commit `isaac_lab/`, `.venv/`, `logs/`, `outputs/`, checkpoints, videos, or screenshots.
- Do not use scalar reward alone to select checkpoints.
- Do not let diagnostics automatically become reward terms.
- Do not compare raw left/right joint averages without applying mirrored sign conventions.
