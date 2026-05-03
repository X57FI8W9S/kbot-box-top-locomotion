# V2 Diagnostics And Evaluator Plan

The v2 priority is a dedicated diagnostics/evaluator module. Diagnostics must stay separate from rewards by default. A large number of diagnostic metrics is good; a large number of overlapping reward pressures is what made v1 hard to reason about.

## Current Failure Modes To Target

- Tiptoe walking and weak full-sole support.
- Persistent hip/torso/box roll bias.
- Roll oscillation centered around a tilted posture instead of neutral.
- Persistent L/R rolling-average joint differences after sign normalization.
- Knees too bent / crouch walking.
- Steps too short.
- Slight under-speed.

Yaw became acceptable in later policies, so v2 should still gate yaw/heading, but the main target is stable-but-bad walking.

## Evaluator Layers

1. Reward terms: only terms used to train the policy.
2. Diagnostic metrics: broad measurement suite with no direct optimization pressure.
3. Evaluator/scorecard: hard gates plus comparison against a baseline checkpoint.

## Hard Safety Gates

Reject a checkpoint if any of these fail:

- Fall or non-foot body contact.
- Timeout fraction too low on fixed 30 s / 60 s evaluation.
- Root height too low.
- Knees too crouched or locked into a small range.
- Uncontrolled yaw drift or lateral drift.
- No consistent alternating L/R steps.

## Contact-Quality Gates

Reject if:

- Toe-only contact ratio is too high.
- Heel contact ratio is too low.
- Full-sole contact ratio is too low.
- Stance foot slip is too high.
- Landing/contact impulses are too violent.
- Left/right sole contact quality differs too much.

Required metrics:

```text
heel_contact_L/R
toe_contact_L/R
forefoot_contact_L/R
full_sole_contact_L/R
heel_only_ratio_L/R
toe_only_ratio_L/R
full_sole_ratio_per_stance_L/R
full_sole_ratio_per_step_L/R
stance_slip_L/R
contact_force_heel_toe_distribution_L/R
```

Current asset status: true heel/toe/inner-edge/outer-edge contact bodies are not present. The current robot exposes whole-foot links `foot1` and `foot3` only. Until the asset is upgraded, full-sole support can only be approximated as whole-foot contact plus a flat foot orientation proxy.

Required asset/sensor upgrade before the next serious training run:

```text
left_heel_contact
left_toe_contact
right_heel_contact
right_toe_contact
```

First code-generated implementation:

```text
asset patch script: scripts/asset/add_heel_toe_pads.py
patched asset: assets/robot/usd/kbot_box_top3_pads.usda
inspection script: scripts/asset/inspect_kbot_usd.py
validation script: scripts/asset/validate_heel_toe_pads.py
V2 asset config: KBOT_PADS_CFG in source/kbot_loco/kbot_loco/tasks/locomotion/assets.py
V2 task uses the pad asset; V1 still uses the original asset.
```

The generated pads are simple USD cube collision bodies, fixed to the original feet:

```text
left_heel_pad
left_toe_pad
right_heel_pad
right_toe_pad
```

Initial validation confirms that Isaac Lab sees these as distinct rigid bodies and the contact sensor exposes distinct body ids. A held-pose scan was added to avoid confusing sensor validation with whether the robot can balance.

Observed held-pose validation:

```text
air: passes; high/tilted poses can produce all four pad contacts false
toe-only: passes; root height around 0.74 to 0.76 m gives toe contact without heel contact
full support: passes; lower held root poses give heel && toe on both feet
heel-only: partial; heel-dominant states can be produced, but clean symmetric heel-only for both feet is not robust with the simple box pads
```

The generated pad asset uses a heel drop offset because the current simplified foot asset is toe-low in the held-pose scan:

```text
heel_drop = 0.04 m
```

This is acceptable for a first diagnostic asset, but not ideal as final foot geometry. If the first pad-trained run shows contact artifacts or asymmetric heel behavior, replace the generated boxes with a CAD/Blender sole split. A five-piece sole per foot is preferred:

```text
heel
toe
inner edge
outer edge
center/midsole
```

Definition after that upgrade:

```text
full_support_left = left_heel_contact && left_toe_contact
full_support_right = right_heel_contact && right_toe_contact
```

Recommended optional edge sensors if side-edge walking remains visible:

```text
left_inner_edge_contact
left_outer_edge_contact
right_inner_edge_contact
right_outer_edge_contact
```

The edge sensors should be diagnostics first. They should only become rewards if the scorecard shows a repeated side-edge exploit.

## Roll Bias And Oscillation

Track bias separately from oscillation:

```text
roll_mean          persistent lean
roll_rms_centered oscillation around the mean
raw_roll_rms      total motion, less useful alone
```

Apply this to:

```text
base/root roll
box/top roll
hip roll
hip yaw
lateral COM offset
```

Reject if the oscillation is centered around a nonzero mean roll. Some roll oscillation is expected because there is no joint between the hip/root and box top, but the mean should stay near neutral.

## L/R Symmetry

Compute symmetry after mirrored sign normalization. Do not compare raw left/right values blindly.

Example:

```text
normalized_right_joint = sign_map[joint_name] * raw_right_joint
symmetry_error = left_joint - normalized_right_joint
```

Track rolling averages over 5 full gait cycles:

```text
hip_pitch_symmetry_error
hip_roll_symmetry_error
hip_yaw_symmetry_error
knee_symmetry_error
ankle_symmetry_error
step_length_symmetry_error
step_duration_symmetry_error
stance_duration_symmetry_error
full_sole_support_symmetry_error
contact_force_symmetry_error
```

## Gait Events

Use event-based windows, not only fixed time windows. The default rolling window should be 5 full gait cycles, normally about 10 steps.

Per-step metrics:

```text
heel_strike_time
toe_off_time
step_duration
step_length
step_width
swing_duration
stance_duration
double_support_duration
single_support_duration
full_sole_support_duration
```

Derived metrics:

```text
cadence
stride_length
duty_factor_L/R
double_support_ratio
single_support_ratio_L/R
airborne_ratio
step_length_per_velocity
expected_step_length = measured_vx * step_duration
```

Reject or review if cadence is high but stride is tiny, step duration is irregular, airborne time appears during walking, or left/right steps diverge.

## Crouch And Posture

Track:

```text
root_height_mean
root_height_p5
knee_angle_mean_L/R
knee_angle_min_L/R
knee_angle_max_L/R
knee_range_L/R
ankle_angle_mean_L/R
hip_pitch_mean_L/R
```

A checkpoint should not pass just because it walks forward while sitting into the knees.

## Speed Tracking

Only compare speed after contact, roll, symmetry, and posture gates pass.

Metrics:

```text
vx_mean
vx_error_mean
vx_error_p95
vx_tracking_ratio = vx_mean / cmd_vx
```

Under-speed is acceptable during development, but a better checkpoint should improve it without worsening tiptoe, crouch, roll bias, or symmetry.

## Outputs

The evaluator should write:

```text
diagnostics/<checkpoint>/metrics.json
diagnostics/<checkpoint>/gait_cycles.csv
diagnostics/<checkpoint>/step_events.csv
diagnostics/<checkpoint>/summary.md
diagnostics/<checkpoint>/dashboard.html
```

Plots should include:

```text
speed/yaw
torso or box bias/RMS
hip bias/RMS
contact states
step length/duration
support ratios
sole contact quality
symmetry
root height and knee distributions
old-vs-new scorecard
```

## Checkpoint Decision

Return one of:

```text
APPROVE
REJECT
REVIEW_VIDEO
```

Approve only if:

1. It survives and walks without body contact.
2. Yaw and lateral direction are acceptable.
3. It does not tiptoe.
4. It has real full-sole support during stance.
5. Roll oscillation is centered near neutral.
6. L/R joint averages are symmetric after sign normalization.
7. Knees are not excessively bent.
8. Step length and cadence are physically reasonable.
9. Commanded forward speed is tracked reasonably.
10. Video agrees with metrics.

Reject examples:

```text
REJECT: walks straight but tiptoes.
REJECT: yaw is good but persistent left roll bias remains.
REJECT: speed improved but knees are more crouched.
REJECT: reward improved but step length collapsed.
REJECT: foot-flat reward improved but heel contact is absent.
REJECT: average torso/box roll is nonzero even though roll RMS is small.
```
