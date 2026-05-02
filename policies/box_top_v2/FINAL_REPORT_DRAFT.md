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
robot asset: assets/robot/urdf/box_top3.urdf via source/kbot_loco/kbot_loco/tasks/locomotion/assets.py
episode length: 8 s for training, 60 s for play/evaluation
decimation: 4
num envs: 2048 default, overridden by train/evaluation CLI
command ranges: forward velocity x = 0.30 to 0.50 m/s, lateral velocity y = 0.0 m/s, yaw rate z = 0.0 rad/s, heading = 0 rad
domain randomization: friction, base mass, base COM, reset pose, reset velocity, reset joint position scaling
termination settings: time-out, low root height below 0.42 m, bad orientation above 0.95 rad
```

Current contact sensor status:

```text
The current box-top asset exposes whole-foot contact bodies named foot1 and foot3.
It does not expose separate heel, toe, sole inner-edge, or sole outer-edge contact bodies.
True full support should be defined later as heel_contact && toe_contact for each foot.
Until the asset is upgraded, contact diagnostics can only report whole-foot contact and proxy foot-flat/full-sole indicators.
```

## 5. V2 Reward Function

Fill this with the exact reward equation for each v2 branch.

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

Exact reward equation:

```text
R =
  ...
```

Term explanations:

```text
term:
  units:
  formula:
  intent:
  risk:
```

## 6. Diagnostics Module

Implementation location:

```text
scripts/diagnostics/evaluate_checkpoint.py
```

Required outputs:

```text
diagnostics/<checkpoint>/metrics.json
diagnostics/<checkpoint>/gait_cycles.csv
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
run: pending
checkpoint: pending
code/config change: added low root-height termination below 0.42 m, bad-orientation termination above 0.95 rad, and upright_alive reward requiring root height above 0.55 m and tilt below 0.45
why it was tried: prevent the collapsed time-out exploit observed in the first V2 run
result: pending
decision: pending
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
lesson:
evidence:
what changed because of it:
```

## 11. Open Questions

- Where did the current actuator parameters originate: Robstride datasheet, KBot repo, hand tuning, or earlier manual guess?
- Is the current initial pose mechanically appropriate, or only a pragmatic standing pose from early fall debugging?
- Should heel/toe/edge contact bodies or sensors be added to the asset before reward tuning?
  - Current status: yes, likely. The current asset search only shows whole-foot links `foot1` and `foot3`; no heel, toe, sole-inner, or sole-outer links/sensors are present.
- What mirrored sign map should be used for left/right joint symmetry?
- What thresholds should define excessive crouch for this simplified body?
- What is the target step length at each command speed?
