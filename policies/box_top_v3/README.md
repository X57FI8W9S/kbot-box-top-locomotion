# Box Top V3

V3 is a parallel restart from iteration 0. It is not a continuation of the
later V2.5 S4 reward experiments.

## Starting Point

V3 starts from the reward topology that produced the V2.5 `model_648.pt`
checkpoint:

```text
source run = logs/rsl_rl/kbot_forward_flat/2026-05-09_00-16-45_v2_5_pose_gait_quality_from_v2_5_349_fsep_ksep
source checkpoint = model_648.pt
source-compatible task = Isaac-KBot-Forward-Flat-V2_5-PoseGaitQuality648Compat-v0
```

The V2.5 keeper was useful because it preserved height, straightness, `fsep`,
and `ksep` while moving forward. It was not accepted as final walking because it
used high-frequency tiny steps.

## V3 Task

```text
task = Isaac-KBot-Forward-Flat-V3-648HandTuned-v0
env cfg = KBotForwardFlatV3HandTuned648EnvCfg
PPO cfg = KBotForwardFlatConservativePPORunnerCfg
resume = no
start = iteration 0
```

The V3 config inherits the frozen `648Compat` code path so the reset pose,
observations, command structure, terminations, and reward term set remain
compatible with the checkpoint that produced the useful V2.5 behavior.

## What Changed

Current V3 behavior changes no reward weights. It is a clean iteration-0
restart using the frozen 648-compatible reward topology, but it keeps an
explicit neutral tuning block in code:

```text
KBotForwardFlatV3HandTuned648EnvCfg inherits KBotForwardFlatV25PoseGaitQuality648CompatEnvCfg.
KBotForwardFlatV3HandTuned648EnvCfg.__post_init__ explicitly assigns every reward weight to the saved model_648 value.
```

The point of the first V3 run is to test whether training from iteration 0 under
the exact 648-compatible recipe behaves differently from continuing a policy
that already found the tiny-step local minimum. The explicit block is there so
hand tuning can be done by changing one visible set of numbers without altering
the V2.5 compatibility task.

## What Did Not Change

V3 deliberately does not include the later S4 anti-shuffle reward stack:

```text
valid_step_root_advance
step_advance_margin
dense_single_support_step_progress
supported_forward_velocity
walking_cycle_cadence_above_l2
contact_chatter_l1
```

Those terms remain useful diagnostics and future branch options, but they are
not part of the first V3 iteration-0 restart.

## First Evaluation Question

The first V3 run should answer:

```text
Can the exact 648 reward topology train from iteration 0 into a policy that
preserves V2.5 height/width/straightness without falling into the same tiny-step
local minimum?
```

Do not promote V3 from scalar reward alone. Use the same 30 s diagnostics and
HUD video review as V2.5, with special attention to:

```text
root_height_p05_m
fsep/ksep
step_root_advance_mean_m
cycle_root_advance_mean_m
cycle_cadence_hz
y_distance_m
yaw/lateral drift
foot clearance
support percentages
J/m
```
