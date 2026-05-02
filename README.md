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
