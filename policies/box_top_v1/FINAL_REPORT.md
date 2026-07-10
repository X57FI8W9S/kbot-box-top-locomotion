# KBot Box-Top Locomotion Final Report

This report is a handoff for restarting the walking policy from a cleaner design. It records what this repository does, how the reward function evolved, what each training branch showed, and what should be changed when rebuilding the task.

## 1. What This Code Does

### Goal

The project trains a simulated biped robot with a large box-like upper body to walk forward on flat ground in Isaac Lab using PPO from RSL-RL.

The box top replaces the torso and arms because arm control is out of scope for this phase. The intended product is not this exact simplified robot. The intended product is a reusable training procedure that can later be adapted to a true humanoid with arms, more actuators, richer contacts, perception, recovery, and navigation.

The task is registered as:

```text
Isaac-KBot-Forward-Flat-v0
Isaac-KBot-Forward-Flat-Play-v0
```

Any awkwardness in the current behavior is unintentional. It is likely caused by the simplified body, limited actuation, reward design, and contact modeling. The simplification is still useful because it makes the first locomotion problem easier and cheaper to train than a full humanoid while preserving the main balance problem: a top-heavy body walking on two legs.

### Main Files

```text
assets/robot/usd/kbot_box_top3.usd
assets/robot/urdf/box_top3.urdf
source/kbot_loco/kbot_loco/tasks/locomotion/assets.py
source/kbot_loco/kbot_loco/tasks/locomotion/env_cfg.py
source/kbot_loco/kbot_loco/tasks/locomotion/mdp.py
source/kbot_loco/kbot_loco/tasks/locomotion/agents/rsl_rl_ppo_cfg.py
scripts/rsl_rl/train.py
scripts/rsl_rl/play_trailing.py
scripts/probe_kbot_stability.py
GAIT_PLAN.md
PROGRESS_REPORT.md
```

### Robot Model

The robot has 10 actuated joints:

```text
left_hip_pitch_04
right_hip_pitch_04
left_hip_roll_03
right_hip_roll_03
left_hip_yaw_03
right_hip_yaw_03
left_knee_04
right_knee_04
left_ankle_02
right_ankle_02
```

Initial pose:

```text
root position z: 0.78 m
left knee: 0.75 rad
right knee: -0.75 rad
other joints: 0.0 rad
```

This initial pose is the current value in `source/kbot_loco/kbot_loco/tasks/locomotion/assets.py`. It should be treated as the current engineering baseline, not as a proven optimal standing pose.

Actuator groups:

```text
hip pitch + knee: effort 120, stiffness 45, damping 4
hip roll:         effort 60,  stiffness 35, damping 3
hip yaw:          effort 60,  stiffness 25, damping 2
ankle:            effort 17,  stiffness 12, damping 1
```

These actuator groups also come from `assets.py`. They were read from the current repository configuration for this report. They should be treated as part of the experiment setup, not as final motor-identification results. Units are Isaac Lab actuator config units: effort limits are torque limits in N*m for revolute joints, velocity limits are rad/s, stiffness is N*m/rad, and damping is N*m*s/rad.

The policy action is joint position control with:

```text
action scale: 0.25
decimation: 4
physics dt: 0.005 s
policy dt: 0.02 s
episode length: 8.0 s during training
```

### PPO Setup

RSL-RL config:

```text
num_steps_per_env: 24
save_interval: 50
actor hidden dims: 256, 128, 128
critic hidden dims: 256, 128, 128
activation: ELU
init_noise_std: 0.2
learning_rate: 1e-3, adaptive
clip_param: 0.2
entropy_coef: 0.01
gamma: 0.99
lambda: 0.95
desired_kl: 0.01
```

Most runs used `1024` or `2048` parallel environments.

### Observations

The policy observes the standard Isaac Lab velocity-task terms plus an added phase signal:

```text
base_lin_vel
base_ang_vel
projected_gravity
velocity_commands
joint_pos
joint_vel
actions
gait_phase = sin/cos phase
```

Final policy observation size is `44`.

The phase signal is:

```text
phase = episode_time / 1.0 s modulo 1.0
gait_phase = [sin(2*pi*phase), cos(2*pi*phase)]
```

This is not an animation target. It is only a light rhythm scaffold.

### Commands

Final training command distribution:

```text
forward velocity x: 0.35 to 0.55 m/s
lateral velocity y: 0.0 m/s
yaw velocity z: 0.0 rad/s
heading: 0.0 rad
resampling time: 4.0 to 8.0 s
```

The original policy attempted faster walking. The final branch deliberately slowed the command range to discover a cleaner gait.

### Randomization

The environment keeps modest robustness randomization:

```text
static friction: 0.9 to 1.2
dynamic friction: 0.7 to 1.0
floating_base_link mass perturbation: -1.0 to 1.0 kg
floating_base_link COM perturbation:
  x: -0.015 to 0.015 m
  y: -0.025 to 0.025 m
  z: -0.010 to 0.010 m
reset root pose:
  x/y: -0.1 to 0.1 m
  yaw: -0.1 to 0.1 rad
reset root velocity:
  x/y: -0.05 to 0.05 m/s
  z: -0.02 to 0.02 m/s
  roll/pitch/yaw: -0.05 to 0.05 rad/s
joint reset multiplier: 0.95 to 1.05
```

Pushes and external force randomization are disabled.

### Final Reward Function

The final reward is a weighted sum of Isaac Lab base terms and custom KBot terms. Positive terms reward tracking, aliveness, foot air time, and phase-consistent contacts. Negative terms penalize falling behaviors, unstable posture, poor foot placement, permanent hip offsets, and tiptoe contact.

Final reward terms:

```text
 0  track_lin_vel_xy_exp                 +3.0
 1  track_ang_vel_z_exp                  +3.5
 2  lin_vel_z_l2                         -2.0
 3  ang_vel_xy_l2                        -0.25
 4  dof_torques_l2                       -5e-5
 5  dof_acc_l2                           -1e-7
 6  action_rate_l2                       -0.08
 7  feet_air_time                        +1.75
 8  undesired_contacts                   -2.0
 9  flat_orientation_l2                  -20.0
10  dof_pos_limits                       -2.0
11  alive                                +2.0
12  base_height_l2                       -20.0
13  alternating_foot_phase               +0.35
14  lateral_velocity_l2                  -7.0
15  yaw_rate_l2                          -7.0
16  root_lateral_tilt_l2                 -90.0
17  root_lateral_tilt_ema_l2             -450.0
18  world_heading_l2                     -32.0
19  backward_velocity_l2                 -2.0
20  forward_velocity_below_l2            -20.0
21  foot_lateral_spacing_l1              -6.0
22  foot_signed_lateral_clearance_l1     -20.0
23  foot_lateral_lane_l1                 -7.0
24  foot_lateral_lane_max_l1             -5.0
25  leg_frontal_plane_l1                 -7.0
26  left_leg_frontal_plane_l1            -2.0
27  right_leg_frontal_plane_l1           -2.0
28  max_leg_frontal_plane_l1             -8.0
29  foot_sagittal_separation_l1          -4.0
30  swing_foot_overtake_l1               -14.0
31  foot_parallel_l2                     -1.5
32  foot_world_parallel_l2                0.0
33  foot_world_parallel_max_l2            0.0
34  foot_toe_in_l2                       -8.0
35  foot_flat_l2                         -0.35
36  stance_foot_flat_l2                  -2.5
37  wobble_joint_vel_l2                  -0.04
38  hip_roll_yaw_position_l2             -12.0
39  hip_roll_yaw_position_ema_l2         -36.0
40  low_body_l2                          -30.0
41  knee_extension_l1                    -30.0
42  termination_penalty                  -500.0
```

Term number 6 is `action_rate_l2`. It penalizes changes in consecutive actions and is intended to reduce twitchy joint target commands.

The scalar reward at each policy step is:

```text
R =
  3.0    * track_lin_vel_xy_exp
+ 3.5    * track_ang_vel_z_exp
- 2.0    * lin_vel_z_l2
- 0.25   * ang_vel_xy_l2
- 5e-5   * dof_torques_l2
- 1e-7   * dof_acc_l2
- 0.08   * action_rate_l2
+ 1.75   * feet_air_time
- 2.0    * undesired_contacts
- 20.0   * flat_orientation_l2
- 2.0    * dof_pos_limits
+ 2.0    * alive
- 20.0   * base_height_l2
+ 0.35   * alternating_foot_phase
- 7.0    * lateral_velocity_l2
- 7.0    * yaw_rate_l2
- 90.0   * root_lateral_tilt_l2
- 450.0  * root_lateral_tilt_ema_l2
- 32.0   * world_heading_l2
- 2.0    * backward_velocity_l2
- 20.0   * forward_velocity_below_l2
- 6.0    * foot_lateral_spacing_l1
- 20.0   * foot_signed_lateral_clearance_l1
- 7.0    * foot_lateral_lane_l1
- 5.0    * foot_lateral_lane_max_l1
- 7.0    * leg_frontal_plane_l1
- 2.0    * left_leg_frontal_plane_l1
- 2.0    * right_leg_frontal_plane_l1
- 8.0    * max_leg_frontal_plane_l1
- 4.0    * foot_sagittal_separation_l1
- 14.0   * swing_foot_overtake_l1
- 1.5    * foot_parallel_l2
+ 0.0    * foot_world_parallel_l2
+ 0.0    * foot_world_parallel_max_l2
- 8.0    * foot_toe_in_l2
- 0.35   * foot_flat_l2
- 2.5    * stance_foot_flat_l2
- 0.04   * wobble_joint_vel_l2
- 12.0   * hip_roll_yaw_position_l2
- 36.0   * hip_roll_yaw_position_ema_l2
- 30.0   * low_body_l2
- 30.0   * knee_extension_l1
- 500.0  * termination_penalty
```

Expanded term meanings:

- `track_lin_vel_xy_exp`: exponential reward for matching commanded horizontal root velocity in m/s. Final commands use `x = 0.35-0.55 m/s`, `y = 0.0 m/s`, with `std = sqrt(0.04) = 0.2 m/s`.
- `track_ang_vel_z_exp`: exponential reward for matching commanded yaw rate in rad/s. Final command is `0.0 rad/s`, with `std = sqrt(0.05) ~= 0.224 rad/s`.
- `lin_vel_z_l2`: squared vertical root velocity penalty in `(m/s)^2`; discourages hopping.
- `ang_vel_xy_l2`: squared roll/pitch angular velocity penalty in `(rad/s)^2`; discourages tumbling body motion.
- `dof_torques_l2`: squared actuator torque penalty. Revolute-joint torque units are N*m.
- `dof_acc_l2`: squared joint acceleration penalty in `(rad/s^2)^2`; discourages violent joint acceleration.
- `action_rate_l2`: squared difference between consecutive policy actions; discourages twitchy target changes.
- `feet_air_time`: Isaac Lab biped air-time reward using `foot1` and `foot3` contact sensors with threshold `0.45 s`. It rewards stepping, but too much weight can encourage light/tiptoe contact.
- `undesired_contacts`: penalty when non-foot bodies contact the ground, including `floating_base_link` and leg shell bodies.
- `flat_orientation_l2`: base/root non-upright penalty from projected gravity; dimensionless because it uses gravity direction components.
- `dof_pos_limits`: joint limit penalty; discourages joint positions near or beyond configured limits in rad.
- `alive`: constant positive reward while the environment is not terminated.
- `base_height_l2`: squared error from target root height `0.78 m`.
- `alternating_foot_phase`: light schedule reward with a `1.0 s` period. Rewards left-only contact in one half-cycle and right-only contact in the other; gives partial credit for double support and penalizes both feet airborne.
- `lateral_velocity_l2`: squared body-frame lateral root velocity in `(m/s)^2`; discourages sideways drift.
- `yaw_rate_l2`: squared body-frame yaw rate in `(rad/s)^2`; encourages straight walking.
- `root_lateral_tilt_l2`: squared `projected_gravity_b[:, 1]`; near upright this is approximately squared root/torso roll in `rad^2`.
- `root_lateral_tilt_ema_l2`: same lateral tilt signal after a `1.5 s` exponential moving average. This penalizes persistent lean more than brief step-cycle motion.
- `world_heading_l2`: squared heading error from world +X. It penalizes sideways heading and backward-facing orientation using the root forward vector.
- `backward_velocity_l2`: squared negative body-frame forward velocity in `(m/s)^2`; penalizes walking backward.
- `forward_velocity_below_l2`: squared shortfall below `0.30 m/s` body-frame forward velocity; prevents shuffling or standing still when a forward command is active.
- `foot_lateral_spacing_l1`: absolute error from desired left-right foot spacing `0.24 m`, computed in the root frame.
- `foot_signed_lateral_clearance_l1`: penalty if signed left-right foot separation drops below `0.16 m`; prevents crossed feet.
- `foot_lateral_lane_l1`: penalty if left and right feet leave nominal lateral lanes at `+0.12 m` and `-0.12 m`, after tolerance `0.03 m`.
- `foot_lateral_lane_max_l1`: worst-foot version of the lane penalty, after tolerance `0.02 m`.
- `leg_frontal_plane_l1`: penalty if shin and foot lateral positions deviate from their hip-centered sagittal lanes, after tolerance `0.03 m`.
- `left_leg_frontal_plane_l1`: left-side version of the leg frontal-plane penalty, after tolerance `0.015 m`.
- `right_leg_frontal_plane_l1`: right-side version of the leg frontal-plane penalty, after tolerance `0.015 m`.
- `max_leg_frontal_plane_l1`: worst individual shin/foot lateral deviation from its sagittal lane, after tolerance `0.01 m`.
- `foot_sagittal_separation_l1`: penalty if fore-aft foot separation is below `0.20 m` during single stance.
- `swing_foot_overtake_l1`: penalty if the swing foot does not pass the stance foot before landing. It uses `target_length = 0.16 m`, `grace_time = 0.10 s`, and `target_air_time = 0.45 s`.
- `foot_parallel_l2`: squared error between each foot forward direction and the root forward direction in the horizontal plane.
- `foot_world_parallel_l2`: squared error between each foot forward direction and world +X. Final weight is `0.0`, so it is disabled.
- `foot_world_parallel_max_l2`: worst-foot version of world foot-yaw alignment. Final weight is `0.0`, so it is disabled.
- `foot_toe_in_l2`: squared toe-in penalty in the root frame after tolerance `0.03`; discourages toes pointing inward toward the centerline.
- `foot_flat_l2`: foot pitch/roll flatness proxy for both feet, `sum(1 - up_z^2)`, where `up_z` is each foot link local-up vector projected onto world z. Dimensionless orientation penalty.
- `stance_foot_flat_l2`: same foot flatness proxy, but applied only to feet currently in contact.
- `wobble_joint_vel_l2`: squared velocity penalty for hip yaw and hip roll joints in `(rad/s)^2`.
- `hip_roll_yaw_position_l2`: squared hip roll/yaw joint position error from default in `rad^2`. This is not the orientation of the whole hip/root/box-top link.
- `hip_roll_yaw_position_ema_l2`: `1.5 s` EMA of hip roll/yaw joint position error, squared. This penalizes persistent joint offsets.
- `low_body_l2`: squared root-height shortfall below `0.45 m`; discourages crouching/collapse without ending the episode.
- `knee_extension_l1`: penalty when knee bend magnitude falls below `0.50 rad`; discourages mechanically locked straight knees.
- `termination_penalty`: large penalty if the environment terminates before timeout.

There is no true reward for total sole contact area on the ground. Current contact sensors provide contact timing/forces, but the reward does not directly measure contact patch area. `stance_foot_flat_l2` is only an orientation proxy for flat stance.

### Terminations

A hard termination is an environment stop condition that ends the episode immediately when a bad state is detected. Examples are base/body contact with the ground, bad orientation, body height below a threshold, or locked knees. Hard terminations are useful for rejecting clearly invalid behavior, but they can also hide gradients from the policy: instead of seeing how bad a near-fall is, the episode simply ends.

The final task intentionally disables several hard terminations:

```text
base_contact = None
bad_orientation = None
low_body = None
locked_knees = None
```

This allowed the optimizer to see costs for bad behavior rather than instantly ending episodes. Stability was monitored with the `termination_penalty` and episode length. The final successful branches reached timeout-only episodes.

### Evaluation Video/HUD

`scripts/rsl_rl/play_trailing.py` records synchronized side-by-side playback:

```text
left half: trailing view
right half: 90-degree side view
output: 1280x720, 16:9
each camera view: 640x720, effectively 8:9
```

The HUD shows:

```text
speed
command speed
yaw
torso rms
torso avg
hip ry rms
L/R rolling-average joint position columns
```

Interpretation:

- `torso rms`: rolling RMS of root/torso lateral tilt, mixes oscillation and bias.
- `torso avg`: rolling signed average of root/torso lateral tilt, mostly bias.
- `hip ry rms`: rolling RMS of hip roll/yaw joint positions, not whole-body hip link rotation.
- L/R columns: rolling average joint positions for pitch, roll, yaw, knee, ankle.

The playback script extends episode length to exceed requested video length so 30 s / 60 s videos do not reset every 8 seconds when using the training task id.

## 2. History And Timeline

### Reconstructed History Before `model_10300`

The history before `model_10300.pt` was not lost. The checkpoint and TensorBoard directories are still present under:

```text
logs/rsl_rl/kbot_forward_flat/
```

This early period is less cleanly documented than the later branch sequence, so the following reconstruction is based on saved `params/env.yaml` files, checkpoint ranges, and TensorBoard scalar summaries. It should be treated as factual for reward/config changes, but less complete for visual conclusions than the later branches.

#### Initial Walking Attempts: `0 -> 999`

Representative run:

```text
2026-04-25_19-54-27_X2, model_0.pt -> model_999.pt
```

Initial command/task setup:

```text
episode length: 12.0 s
lin_vel_x: 0.45 to 0.65 m/s
lin_vel_y: 0.0 m/s
ang_vel_z: 0.0 rad/s
```

Initial reward terms present in this representative run:

```text
track_lin_vel_xy_exp        +2.5
track_ang_vel_z_exp         +1.0
lin_vel_z_l2                -2.0
ang_vel_xy_l2               -0.25
dof_torques_l2              -5e-5
dof_acc_l2                  -1e-7
action_rate_l2              -0.025
feet_air_time               +0.75
undesired_contacts          -2.0
flat_orientation_l2         -8.0
dof_pos_limits              -2.0
lateral_velocity_l2         -2.0
yaw_rate_l2                 -0.5
backward_velocity_l2        -2.0
termination_penalty         -5.0
```

Observed scalar summary at the end of the representative run:

```text
mean episode length: 556.95 steps of 0.02 s ~= 11.14 s
timeout fraction: 0.90666
xy velocity error: 0.80611 m/s
yaw velocity error: 0.14417 rad/s
```

Interpretation:

Early policies learned to survive long enough to time out, but forward tracking was poor. The reward function was still generic and did not yet encode the specific failure modes that later dominated: low body, locked knees, lateral lean, foot crossing, and hip roll/yaw bias.

#### Anti-Collapse Terms: around `0 -> 650`

Representative run:

```text
2026-04-25_20-35-49_X10, model_0.pt -> model_650.pt
```

Reward changes relative to the initial representative run:

```text
low_body_l2          added at -20.0
knee_extension_l1    added at -4.0
```

Interpretation:

It was identified that policies could exploit crouched or mechanically poor postures. `low_body_l2` penalized root height below a threshold, and `knee_extension_l1` discouraged straight/locked knees.

#### Slower Speed And Survival Pressure: around `0 -> 1050`

Representative run:

```text
2026-04-25_21-14-30_X18, model_0.pt -> model_1050.pt
```

Task and reward changes:

```text
lin_vel_x:              0.45-0.65 -> 0.20-0.35 m/s
alive:                  added at +5.0
base_height_l2:         added at -15.0, target 0.78 m
flat_orientation_l2:    -8.0 -> -20.0
knee_extension_l1:      -4.0 -> -50.0
low_body_l2:            -20.0 -> -30.0
termination_penalty:    -5.0 -> -100.0
```

Observed scalar summary:

```text
mean episode length: 52.70 steps ~= 1.05 s
timeout fraction: 0.0
xy velocity error: 0.05339 m/s
yaw velocity error: 0.07700 rad/s
```

Interpretation:

The command was slowed sharply and posture/survival terms were strengthened. Velocity error became small because commands were slow, but the policy did not yet survive full episodes. This was a stability-building phase, not a good walking phase.

#### Short Episode / Strong Termination Phase: around `0 -> 1000`

Representative run:

```text
2026-04-25_21-30-30_X21, model_0.pt -> model_1000.pt
```

Task and reward changes:

```text
episode length:         12.0 -> 3.0 s
lin_vel_x:              0.20-0.35 -> 0.10-0.25 m/s
knee_extension_l1:      -50.0 -> -80.0
termination_penalty:    -100.0 -> -500.0
```

Observed scalar summary:

```text
mean episode length: 150 steps = 3.0 s
timeout fraction: 1.0
xy velocity error: 0.07129 m/s
yaw velocity error: 0.27508 rad/s
```

Interpretation:

Episodes were shortened to make survival achievable. This helped produce timeout-reaching policies, but yaw control remained poor and the task was too easy/short to imply robust walking.

#### Straightness Pressure: `1050 -> 2348`

Representative runs:

```text
2026-04-25_21-42-41_X21.1.1, model_1050.pt -> model_1549.pt
2026-04-26_01-38-53_X21.(1)3, model_1550.pt -> model_2348.pt
```

Task/reward changes:

```text
episode length:             3.0 -> 8.0 s
lateral_velocity_l2:        -2.0 -> -3.0
track_ang_vel_z_exp:        +1.0 -> +2.0
yaw_rate_l2:                -0.5 -> -2.0
```

Observed scalar summaries:

```text
model_1549 range:
  mean episode length: 400 steps = 8.0 s
  timeout fraction: 1.0
  xy velocity error: 0.13314 m/s
  yaw velocity error: 0.43884 rad/s

model_2348 range:
  mean episode length: 400 steps = 8.0 s
  timeout fraction: 1.0
  xy velocity error: 0.12311 m/s
  yaw velocity error: 0.32980 rad/s
```

Interpretation:

The policy could survive 8 s episodes, but straightness/yaw was still poor. This motivated stronger yaw and foot placement structure.

#### First Foot Placement Terms: `2250 -> 3148`

Representative runs:

```text
2026-04-26_02-15-34_X21.(1)3.2, model_2250.pt -> model_2649.pt
2026-04-26_02-45-50_X21.(1)3.2.1, model_2650.pt -> model_3148.pt
```

Task/reward changes:

```text
lin_vel_x:                         0.10-0.25 -> 0.18-0.30 m/s
alive:                             +5.0 -> +2.0
forward_velocity_below_l2:         added at -8.0
foot_lateral_spacing_l1:           added at -3.0
foot_parallel_l2:                  added at -1.0
foot_flat_l2:                      added at -0.5, then -0.2

action_rate_l2:                    -0.025 -> -0.035
feet_air_time:                     +0.75 -> +1.25
foot_lateral_spacing_l1:           -3.0 -> -5.0
foot_signed_lateral_clearance_l1:  added at -12.0
foot_world_parallel_l2:            added at -4.0
lateral_velocity_l2:               -3.0 -> -5.0
track_ang_vel_z_exp:               +2.0 -> +3.0
world_heading_l2:                  added at -12.0
yaw_rate_l2:                       -2.0 -> -4.0
```

Observed scalar summary near `3148`:

```text
mean episode length: 400 steps = 8.0 s
timeout fraction: 1.0
xy velocity error: 0.08892 m/s
yaw velocity error: 0.18416 rad/s
```

Interpretation:

Foot placement and heading terms materially improved yaw/straightness. This is where the reward began moving from generic survival toward gait shaping.

#### Stronger Lateral Clearance / Step Structure: `3150 -> 4199`

Representative runs:

```text
2026-04-26_02-52-06_X21.(1)3.2.1.1, model_3150.pt -> model_3647.pt
2026-04-26_03-27-17_X21.1.(1.1.2)2, model_3600.pt -> model_4199.pt
```

Task/reward changes:

```text
lin_vel_x:                         0.18-0.30 -> 0.22-0.34 -> 0.32-0.42 m/s
action_rate_l2:                    -0.035 -> -0.05 -> -0.055
foot_lateral_spacing_l1:           -5.0 -> -6.0
foot_signed_lateral_clearance_l1:  -12.0 -> -20.0
foot_world_parallel_l2:            -4.0 -> -6.0
forward_velocity_below_l2:         -8.0 -> -10.0 -> -12.0
world_heading_l2:                  -12.0 -> -16.0
yaw_rate_l2:                       -4.0 -> -5.0
foot_sagittal_separation_l1:       added at -5.0
wobble_joint_vel_l2:               added at -0.04
```

Observed scalar summaries:

```text
model_3647 range:
  xy velocity error: 0.09279 m/s
  yaw velocity error: 0.14435 rad/s

model_4199 range:
  xy velocity error: 0.10307 m/s
  yaw velocity error: 0.14405 rad/s
```

Interpretation:

Foot crossing and foot-lane discipline were reinforced. Yaw error improved compared with the earlier 8 s survival phase, but the policy was becoming increasingly constrained by foot and heading terms.

#### Fast-Walk Push And Swing Structure: `5450 -> 6599`

Representative runs:

```text
2026-04-26_04-50-38_X21.1.(1.1.2)2.(1)4, model_5450.pt -> model_6048.pt
2026-04-26_17-54-49_X21.1.(1.1.2)2.(1)5, model_6000.pt -> model_6599.pt
```

Task/reward changes:

```text
lin_vel_x:                         0.32-0.42 -> 0.80-1.10 m/s
action_rate_l2:                    -0.055 -> -0.08
feet_air_time:                     +1.25 -> +2.25
foot_parallel_l2:                  -1.0 -> -1.5
foot_sagittal_separation_l1:       -5.0 -> -4.0
forward_velocity_below_l2:         -12.0 -> -16.0
stance_foot_flat_l2:               added at -1.5
swing_foot_overtake_l1:            added at -14.0
world_heading_l2:                  -16.0 -> -20.0
leg_frontal_plane_l1:              added at -5.0
```

Observed scalar summaries:

```text
model_6048 range:
  xy velocity error: 0.17006 m/s
  yaw velocity error: 0.13418 rad/s

model_6599 range:
  xy velocity error: 0.17094 m/s
  yaw velocity error: 0.11420 rad/s
```

Interpretation:

The policy was pushed toward faster walking. Step structure and stance-foot flatness were added. Speed tracking became harder, but yaw improved. This phase produced moving policies, but likely encouraged compensatory lean and hip offsets because forward progress was being pushed before gait symmetry was solved.

#### Strong Frontal-Plane And Hip/Torso Bias Penalties: `8400 -> 10300`

Representative runs:

```text
2026-04-26_19-42-47_X21.1.(1.1.2)2.(1)9, model_8400.pt -> model_8895.pt
2026-04-26_20-29-08_z, model_8950.pt -> model_9449.pt
2026-04-26_21-22-06_z.2, model_9250.pt -> model_9649.pt
2026-04-27_01-50-43_z.2.1, model_9500.pt -> model_10099.pt
2026-04-27_01-57-15_z.2.1.1, model_10050.pt -> model_10349.pt
```

Task/reward changes:

```text
lin_vel_x:                         0.80-1.10 -> 0.75-0.95 m/s
track_lin_vel_xy_exp:              +2.5 -> +3.0
track_ang_vel_z_exp:               +3.0 -> +3.5
base_height_l2:                    -15.0 -> -20.0
forward_velocity_below_l2:         -16.0 -> -20.0
world_heading_l2:                  -20.0 -> -24.0
foot_signed_lateral_clearance_l1:  -20.0 -> -24.0
root_lateral_tilt_l2:              added at -24.0
hip_roll_yaw_position_l2:          added at -1.5
leg_frontal_plane_l1:              -5.0 -> -14.0
left_leg_frontal_plane_l1:         added at -4.0
right_leg_frontal_plane_l1:        added at -4.0
max_leg_frontal_plane_l1:          added at -16.0
foot_lateral_lane_l1:              added at -10.0
foot_lateral_lane_max_l1:          added at -8.0
foot_toe_in_l2:                    added at -8.0
foot_world_parallel_max_l2:        added at -3.0
```

Observed scalar summary at the reference baseline:

```text
2026-04-27_01-57-15_z.2.1.1/model_10300.pt:
  speed mean:            0.77082 m/s
  command mean:          0.81824 m/s
  yaw-rate mean:        -0.01659 rad/s
  torso avg mean:        0.03291
  torso RMS mean:        0.04046
  hip roll/yaw RMS mean: 0.10818 rad
```

Interpretation:

The reward function at `model_10300.pt` had become a dense gait-shaping reward with strong speed, heading, foot lane, frontal-plane, and anti-bias terms. It produced a usable forward-moving policy, but the quality metrics showed a persistent lateral torso bias and persistent hip roll/yaw offsets. This motivated the post-10300 branch plan: reduce overconstraint, slow down, directly target persistent bias, and separate bias from oscillation in the HUD.

### Baseline

Reference checkpoint:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_01-57-15_z.2.1.1/model_10300.pt
```

Baseline metrics:

```text
torso RMS mean:        0.04046
torso mean-bias mean:  0.03291
hip roll/yaw RMS mean: 0.10818
```

The robot could move, but it leaned and used persistent hip roll/yaw offsets. The gait solved balance through a biased posture instead of symmetric walking. The original quality target was later clarified: not a small improvement, but roughly 80-90% lower torso/hip bias metrics.

### Branch A: Deconstrain Foot Yaw / Frontal Plane

Run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_03-46-31_z.2.(1)3/model_10449.pt
```

Why:

It was suspected that the policy was boxed in by too many simultaneous foot yaw, heading, knee, and frontal-plane constraints.

Changes:

```text
foot_world_parallel_l2:      -6.0 -> 0.0
foot_world_parallel_max_l2:  -3.0 -> 0.0
knee_extension_l1:           -80.0 -> -30.0
leg_frontal_plane_l1:        -14.0 -> -7.0
left/right leg plane:        -4.0 -> -2.0
max_leg_frontal_plane_l1:    -16.0 -> -8.0
```

Result:

```text
torso RMS mean:        0.04026
torso mean-bias mean:  0.03212
hip roll/yaw RMS mean: 0.11007
```

Meaning:

The overconstraint hypothesis was only partly useful. Deconstraining did not fix the core persistent lean. It reduced some reward conflict, but did not give the optimizer a direct reason to remove multi-step bias.

### Branch B: Slower Clean Gait

Run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-06-27_z.2.1.1.2/model_10599.pt
```

Why:

The fast command range appeared to make the policy prioritize moving over walking cleanly. The task was slowed so symmetry could emerge first.

Changes:

```text
lin_vel_x:                            0.75-0.95 -> 0.35-0.55
forward_velocity_below_l2 minimum:    0.68 -> 0.30
foot_sagittal_separation target:      0.32 -> 0.20
swing_foot_overtake target:           0.24 -> 0.16
```

Result:

```text
torso RMS mean:        0.03736
torso mean-bias mean:  0.02913
hip roll/yaw RMS mean: 0.10641
```

Meaning:

Slower commands helped slightly, but the robot still leaned. This confirmed that speed pressure was part of the issue, not the whole issue.

### Branch C: Stronger Instantaneous Torso/Hip Penalties

Run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-11-47_z.2.1.1.2.1/model_10798.pt
```

Why:

The task needed a stronger direct penalty on the primary quality quantities.

Changes:

```text
root_lateral_tilt_l2:      -24.0 -> -60.0
hip_roll_yaw_position_l2:  -1.5 -> -6.0
```

Result:

```text
torso RMS mean:        0.02886
torso mean-bias mean:  0.01924
hip roll/yaw RMS mean: 0.10263
```

Meaning:

This was the first clear improvement. Directly penalizing lateral tilt and hip roll/yaw was necessary.

### Branch C Continuation

Run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-15-45_z.(2.1.1)2/model_10997.pt
```

Result:

```text
torso RMS mean:        0.02677
torso mean-bias mean:  0.01218
hip roll/yaw RMS mean: 0.09561
```

Meaning:

Continuing the same branch kept improving. It crossed the first practical hip milestone below `0.10 rad`, but was still far from the original 80-90% reduction target.

### Branch D/E: Phase Scaffold And Warm Start

Key checkpoint:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_11-12-07_z.(2.1.1)3/model_11294.pt
```

Why:

The gait needed a consistent stepping rhythm. `sin/cos` gait phase observations and a light alternating contact reward were added.

Result:

```text
speed mean:            0.41700
command mean:          0.41824
yaw-rate mean:        -0.00681
torso RMS mean:        0.02054
hip roll/yaw RMS mean: 0.09683
```

Meaning:

The phase scaffold helped stabilize the stepping pattern and reduced torso RMS substantially, but hip roll/yaw RMS was still high. The contact schedule should remain light. It is a rhythm hint, not a pose script.

### Branch F: EMA Persistent-Bias Rewards

Why:

Instantaneous penalties improved the policy but plateaued. The real failure was persistent bias over several steps. RMS alone could mix oscillation and lean; mean/EMA terms targeted the constant offset.

Changes:

```text
root_lateral_tilt_ema_l2:  added, tau_s=1.5, weight -300.0
joint_position_ema_l2:     added for hip roll/yaw, tau_s=1.5, weight -24.0
```

Meaning:

This was conceptually important: persistent lateral lean and persistent hip roll/yaw offsets need their own reward signal. The final HUD later separated `torso rms` from `torso avg` for the same reason.

### Branch G: Straight-Posture Ramp 1

Runs:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_17-42-57_z.(2.1.1)3.1/model_11393.pt
logs/rsl_rl/kbot_forward_flat/2026-04-27_17-46-04_z.(2.1.1)3.1.1/model_11492.pt
```

Why:

Gradually tighten straightness and posture without restarting or applying one large shock.

Changes:

```text
lateral_velocity_l2:            -5.0 -> -7.0
yaw_rate_l2:                    -5.0 -> -7.0
root_lateral_tilt_l2:           -60.0 -> -90.0
root_lateral_tilt_ema_l2:       -300.0 -> -450.0
world_heading_l2:               -24.0 -> -32.0
hip_roll_yaw_position_l2:       -9.0 -> -12.0
hip_roll_yaw_position_ema_l2:   -24.0 -> -36.0
foot/leg lane tolerances:       tightened
```

Metrics:

```text
model_11294:
  torso RMS mean 0.02054
  hip roll/yaw RMS mean 0.09683

model_11393:
  torso RMS mean 0.01959
  hip roll/yaw RMS mean 0.09193

model_11492:
  torso RMS mean 0.02087
  hip roll/yaw RMS mean 0.08850
```

Meaning:

This moved hip roll/yaw in the right direction without falls. Torso improved at `11393`, then slightly regressed at `11492`. `11492` was visually close and became the warm start for contact cleanup.

### Branch H: Sole Contact Cleanup

Run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05_z.(2.1.1)3.(1)3/model_11791.pt
```

Why:

The robot still appeared to tiptoe or use edge contact. There was no true sole-area reward, so the foot-flatness proxy was strengthened.

Changes:

```text
foot_flat_l2 and stance_foot_flat_l2 formula:
  old: square(1 - abs(up_z))
  new: 1 - up_z^2
```

This made moderate foot pitch/roll visible to the optimizer.

HUD/video changes:

```text
added torso avg
kept torso rms
kept hip ry rms
added side-by-side trailing + side video
fixed playback reset at 8 seconds
restored L/R rolling-average joint columns
```

Metrics:

```text
speed mean:            0.39986
command mean:          0.41960
yaw-rate mean:         0.00568
torso avg mean:       -0.00087
torso RMS mean:        0.01888
hip roll/yaw RMS mean: 0.08566
```

Meaning:

This was a strong checkpoint. Torso signed bias was essentially gone. Remaining torso motion looked more oscillatory than biased. Hip roll/yaw improved again. Side-view inspection became the deciding factor.

#### How to Run `model_11791.pt`

Use this as the default V1 reproduction checkpoint:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05_z.(2.1.1)3.(1)3/model_11791.pt
```

The trained USD is:

```text
assets/robot/usd/kbot_box_top3.usd
```

Do not use the later pads asset or the root pose-tuning USDs for reproducing V1. The saved run snapshot hard-codes `assets/robot/usd/kbot_box_top3.usd`.

The V1 physics setup depends on these articulation root properties in `source/kbot_loco/kbot_loco/tasks/locomotion/assets.py`:

```text
enabled_self_collisions=True
solver_position_iteration_count=8
solver_velocity_iteration_count=2
```

These settings were present in the saved V1 run and must remain set for `model_11791.pt` to reproduce correctly. If they are missing, the same checkpoint can fail even with the same USD.

Reproduce the 30 second diagnostic metrics:

```bash
cd /media/rnyx/Tapioka/TPs/kbot-rl-loco3

.venv/bin/python scripts/diagnostics/evaluate_checkpoint.py \
  --task Isaac-KBot-Forward-Flat-v0 \
  --headless \
  --checkpoint logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05_z.(2.1.1)3.(1)3/model_11791.pt \
  --output_dir logs/rsl_rl/kbot_forward_flat/2026-05-19_v1_repro/diagnostics/model_11791_restored_articulation_30s \
  --video_length 1500
```

Expected diagnostic result after restoring the V1 articulation settings:

```text
decision: REVIEW_VIDEO
distance_m: 12.7106
speed_mean_mps: 0.4174
command_speed_mean_mps: 0.4362
speed_tracking_ratio: 0.9569
root_height_p05_m: 0.7226
step_count: 272
step_root_advance_mean_m: 0.04665
cycle_root_advance_mean_m: 0.09351
cycle_cadence_hz: 4.536
yaw_drift_rad_per_m: 0.00634
lateral_drift_m_per_m: 0.0671
```

Make a fresh video with the current HUD overlay:

```bash
cd /media/rnyx/Tapioka/TPs/kbot-rl-loco3

.venv/bin/python scripts/rsl_rl/play_trailing.py \
  --task Isaac-KBot-Forward-Flat-Play-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05_z.(2.1.1)3.(1)3/model_11791.pt \
  --video_length 1500 \
  --headless
```

Current restored-articulation video:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05_z.(2.1.1)3.(1)3/videos/play/trailing-hud-model_11791-restored-articulation-current-overlay.mp4
```

Note: `play_trailing.py` currently requires the full checkpoint path for this workflow; `--load_run 2026-04-29_06-29-05_z.(2.1.1)3.(1)3 --checkpoint model_11791.pt` did not resolve the file in this reproduction pass.

### Branch I: Final Sole-Contact Push

Run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-29_07-58-47_z.(2.1.1)3.(1)4/model_11990.pt
```

Why:

One last short continuation before restarting, aimed only at sole contact. Torso/hip rewards were not tightened again because those terms were already doing their job and further tightening risked stiffness or crouch.

Changes:

```text
feet_air_time.weight:          2.25 -> 1.75
foot_flat_l2.weight:          -0.2 -> -0.35
stance_foot_flat_l2.weight:   -1.5 -> -2.5
```

Metrics on 30 s video:

```text
speed mean:            0.42284
command mean:          0.43620
yaw-rate mean:        -0.00835
torso avg mean:       -0.00122
torso RMS mean:        0.02125
hip roll/yaw RMS mean: 0.08552
```

Meaning:

Training stayed stable for 200 iterations: full 400-step episodes, timeout-only, no termination penalty. Hip roll/yaw was effectively unchanged, torso RMS was slightly worse on this 30 s sample, speed tracking was a little better. Choose between `11791` and `11990` visually based on sole contact. If sole contact is not clearly better, `11791` is the safer final checkpoint.

## 3. Recommended Remake

### High-Level Recommendation

Restart from a cleaner reward design rather than continuing to stack patches. The useful lessons are clear:

1. Train slow, clean walking before speed.
2. Penalize persistent bias separately from oscillation.
3. Keep gait phase/contact schedule light.
4. Avoid duplicate terms that all constrain the same thing.
5. Add better foot contact observability before pushing flat-foot rewards too hard.
6. Select checkpoints with video and rolling metrics, not scalar reward alone.

### Initial Task For The New Policy

Start with:

```text
lin_vel_x: 0.30 to 0.50 m/s
lin_vel_y: 0.0
ang_vel_z: 0.0
heading: 0.0
episode length: 8 s
num_envs: 2048 if stable, 1024 if iteration speed or memory is better
```

Keep moderate domain randomization, but do not add pushes, rough terrain, or vision until the gait is good.

### Reward Design To Start With

Use fewer terms at first.

Core task:

```text
track_lin_vel_xy_exp
track_ang_vel_z_exp
alive
base_height_l2
lin_vel_z_l2
ang_vel_xy_l2
action_rate_l2
dof_torques_l2
dof_acc_l2
dof_pos_limits
undesired_contacts
termination_penalty
```

Straight walking:

```text
lateral_velocity_l2
yaw_rate_l2
world_heading_l2
backward_velocity_l2
forward_velocity_below_l2
```

Persistent-bias terms from the beginning:

```text
root_lateral_tilt_l2
root_lateral_tilt_ema_l2
hip_roll_yaw_position_l2
hip_roll_yaw_position_ema_l2
```

Foot placement:

```text
foot_signed_lateral_clearance_l1
foot_lateral_lane_l1
foot_lateral_spacing_l1
foot_sagittal_separation_l1
swing_foot_overtake_l1
foot_toe_in_l2
foot_flat_l2
stance_foot_flat_l2
```

Use only one or two leg frontal-plane terms at first. Do not start with all of:

```text
leg_frontal_plane_l1
left_leg_frontal_plane_l1
right_leg_frontal_plane_l1
max_leg_frontal_plane_l1
foot_lateral_lane_l1
foot_lateral_lane_max_l1
```

That cluster was useful for diagnostics but too redundant for a clean starting design.

### Reward Weights To Try First

Suggested initial restart weights:

```text
track_lin_vel_xy_exp                 +3.0
track_ang_vel_z_exp                  +3.0
alive                                +2.0
feet_air_time                        +1.5
alternating_foot_phase               +0.25

base_height_l2                       -20.0
flat_orientation_l2                  -15.0
lateral_velocity_l2                  -5.0
yaw_rate_l2                          -5.0
world_heading_l2                     -20.0
root_lateral_tilt_l2                 -60.0
root_lateral_tilt_ema_l2             -300.0
hip_roll_yaw_position_l2             -8.0
hip_roll_yaw_position_ema_l2         -24.0

foot_signed_lateral_clearance_l1     -20.0
foot_lateral_spacing_l1              -5.0
foot_lateral_lane_l1                 -5.0
foot_sagittal_separation_l1          -3.0
swing_foot_overtake_l1               -10.0
foot_toe_in_l2                       -6.0
foot_flat_l2                         -0.25
stance_foot_flat_l2                  -1.5

knee_extension_l1                    -25.0
low_body_l2                          -30.0
termination_penalty                  -500.0
```

Then ramp:

```text
if gait is stable and not crossing:
  increase command speed gradually

if torso avg is biased:
  increase root_lateral_tilt_ema_l2

if torso rms is high but avg is near zero:
  do not over-tighten bias; inspect oscillation source

if hip ry rms is high:
  increase hip_roll_yaw_position_ema_l2 before instantaneous hip penalty

if tiptoe remains:
  increase stance_foot_flat_l2 gradually and reduce feet_air_time slightly
```

### Metrics To Log From The Start

Do not wait until late training to add measurement.

Log these every evaluation:

```text
speed mean / p95 / final
command speed mean
yaw-rate mean / p95 / max
torso_tilt_window_mean
torso_tilt_window_rms
hip_roll_yaw_window_mean_abs
hip_roll_yaw_window_rms
foot contact duty factor left/right
double support fraction
airborne fraction
stance foot flatness
step length / sagittal separation
root height
knee angle min/max
```

Always render side-by-side videos:

```text
30 s for quick selection
60 s for final candidates
```

Use `torso avg` and `torso rms` together:

- `torso avg` tells whether it is leaning.
- `torso rms` tells whether it is moving/oscillating.
- A high RMS with near-zero avg is a different problem than a biased mean.

### Do Not Forget

- `hip ry rms` is a joint-position metric for hip roll/yaw joints. It is not the orientation of the whole hip/root/box-top link.
- The whole upper body/box-top orientation should be measured with root orientation or projected gravity, not hip joint names.
- The final side camera should be horizontal around knee/mid-body height and centered on the root/hip link laterally.
- The playback task must extend episode length or long videos reset every 8 seconds.
- A true sole-contact-area reward does not exist in the current code. `stance_foot_flat_l2` is only an orientation proxy.
- The final `1 - up_z^2` foot flatness formula was much more meaningful than `square(1 - abs(up_z))`.
- Do not let `feet_air_time` become too dominant; it can encourage light/tiptoe contact.
- Do not stack many duplicate lateral lane/frontal-plane terms until the failure mode proves they are needed.
- Do not use scalar reward alone to select policies. Some visually worse gaits can score better.
- Keep videos and metrics named by checkpoint and duration.
- The best late checkpoints were close. `model_11791.pt` may be visually safer than `model_11990.pt` unless `11990` clearly improves sole contact.

### Path Toward Future Skills

For obstacle avoidance, vision, path planning, and fall recovery, keep the walking controller reusable:

1. First build a strong flat-ground velocity-following base policy.
2. Then add command variations: turning, lateral motion, speed ramps.
3. Then add terrain and obstacle proprioceptive features.
4. Then add exteroception/vision as a higher-level conditioning signal.
5. Keep fall recovery separate at first, or train it as a reset/recovery skill with its own success metrics.
6. For path planning, avoid mixing global planning into the low-level gait reward. Feed the gait policy local velocity/heading commands.
7. For vision obstacle avoidance, train a perception-conditioned command or local planner that drives this walking policy, not a monolithic policy that must rediscover locomotion.

The abstraction should be:

```text
low-level locomotion policy:
  inputs: proprioception + local command
  output: joint targets

mid-level navigation / recovery / obstacle policy:
  inputs: task state, terrain/vision, robot state
  output: local velocity/heading command or recovery mode
```

The walking policy should be boring and reliable before adding anything clever.
