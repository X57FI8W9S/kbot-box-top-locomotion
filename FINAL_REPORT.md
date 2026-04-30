# KBot Box-Top Locomotion Final Report

This report is a handoff for restarting the walking policy from a cleaner design. It records what this repository does, how the reward function evolved, what each training branch taught us, and what I would change when rebuilding the task.

## 1. What This Code Does

### Goal

The project trains a simulated biped robot with a large box-like upper body to walk forward on flat ground in Isaac Lab using PPO from RSL-RL.

The task is registered as:

```text
Isaac-KBot-Forward-Flat-v0
Isaac-KBot-Forward-Flat-Play-v0
```

The robot is intentionally awkward: a high, wide, box-top body on two legs. That makes lateral balance, hip compensation, and foot contact quality the important problems, not just forward velocity.

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

Actuator groups:

```text
hip pitch + knee: effort 120, stiffness 45, damping 4
hip roll:         effort 60,  stiffness 35, damping 3
hip yaw:          effort 60,  stiffness 25, damping 2
ankle:            effort 17,  stiffness 12, damping 1
```

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
lateral velocity y: 0.0
yaw velocity z: 0.0
heading: 0.0
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
  xyz small
  roll/pitch/yaw: -0.05 to 0.05 rad/s
joint reset multiplier: 0.95 to 1.05
```

Pushes and external force randomization are disabled.

### Final Reward Function

The final reward is a weighted sum of Isaac Lab base terms and custom KBot terms. Positive terms reward tracking, aliveness, foot air time, and phase-consistent contacts. Negative terms penalize falling behaviors, unstable posture, poor foot placement, permanent hip offsets, and tiptoe contact.

Final reward terms:

```text
track_lin_vel_xy_exp                 +3.0
track_ang_vel_z_exp                  +3.5
alive                                +2.0
feet_air_time                        +1.75
alternating_foot_phase               +0.35

lin_vel_z_l2                         -2.0
ang_vel_xy_l2                        -0.25
dof_torques_l2                       -5e-5
dof_acc_l2                           -1e-7
action_rate_l2                       -0.08
undesired_contacts                   -2.0
flat_orientation_l2                  -20.0
dof_pos_limits                       -2.0
base_height_l2                       -20.0
lateral_velocity_l2                  -7.0
yaw_rate_l2                          -7.0
root_lateral_tilt_l2                 -90.0
root_lateral_tilt_ema_l2             -450.0
world_heading_l2                     -32.0
backward_velocity_l2                 -2.0
forward_velocity_below_l2            -20.0
foot_lateral_spacing_l1              -6.0
foot_signed_lateral_clearance_l1     -20.0
foot_lateral_lane_l1                 -7.0
foot_lateral_lane_max_l1             -5.0
leg_frontal_plane_l1                 -7.0
left_leg_frontal_plane_l1            -2.0
right_leg_frontal_plane_l1           -2.0
max_leg_frontal_plane_l1             -8.0
foot_sagittal_separation_l1          -4.0
swing_foot_overtake_l1               -14.0
foot_parallel_l2                     -1.5
foot_world_parallel_l2                0.0
foot_world_parallel_max_l2            0.0
foot_toe_in_l2                       -8.0
foot_flat_l2                         -0.35
stance_foot_flat_l2                  -2.5
wobble_joint_vel_l2                  -0.04
hip_roll_yaw_position_l2             -12.0
hip_roll_yaw_position_ema_l2         -36.0
low_body_l2                          -30.0
knee_extension_l1                    -30.0
termination_penalty                  -500.0
```

Important details:

- `root_lateral_tilt_l2` uses `projected_gravity_b[:, 1]`, which is approximately torso/root roll near upright.
- `root_lateral_tilt_ema_l2` penalizes persistent lateral lean through a 1.5 s exponential moving average.
- `hip_roll_yaw_position_l2` and `hip_roll_yaw_position_ema_l2` use hip roll/yaw joint positions relative to default. They are not the orientation of the whole hip/root/box-top link.
- `foot_flat_l2` and `stance_foot_flat_l2` use `1 - up_z^2`, where `up_z` is the world z component of each foot link local up vector. This is a foot pitch/roll flatness proxy, not a true sole contact-area measurement.
- `stance_foot_flat_l2` only applies the foot-flatness penalty to feet in contact.
- There is no true reward for total sole area on the ground because the current contact sensor data does not directly expose contact patch area.

### Terminations

The final task intentionally disables several hard terminations:

```text
base_contact = None
bad_orientation = None
low_body = None
locked_knees = None
```

This allowed the optimizer to see gradients/costs for bad behavior rather than instantly ending episodes. Stability was monitored with the `termination_penalty` and episode length. The final successful branches reached timeout-only episodes.

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

### Baseline

Reference checkpoint:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-27_01-57-15/model_10300.pt
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
logs/rsl_rl/kbot_forward_flat/2026-04-27_03-46-31/model_10449.pt
```

Why:

We suspected the policy was boxed in by too many simultaneous foot yaw, heading, knee, and frontal-plane constraints.

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
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-06-27/model_10599.pt
```

Why:

The fast command range made the policy prioritize moving over walking cleanly. We slowed the task so symmetry could emerge first.

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
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-11-47/model_10798.pt
```

Why:

The task needed a stronger direct penalty on the quantities we cared about.

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
logs/rsl_rl/kbot_forward_flat/2026-04-27_04-15-45/model_10997.pt
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
logs/rsl_rl/kbot_forward_flat/2026-04-27_11-12-07/model_11294.pt
```

Why:

The gait needed a consistent stepping rhythm. We added `sin/cos` gait phase observations and a light alternating contact reward.

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
logs/rsl_rl/kbot_forward_flat/2026-04-27_17-42-57/model_11393.pt
logs/rsl_rl/kbot_forward_flat/2026-04-27_17-46-04/model_11492.pt
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
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/model_11791.pt
```

Why:

The robot still appeared to tiptoe or use edge contact. There was no true sole-area reward, so we strengthened the foot-flatness proxy.

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

### Branch I: Final Sole-Contact Push

Run:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-29_07-58-47/model_11990.pt
```

Why:

One last short continuation before restarting, aimed only at sole contact. We did not tighten torso/hip again because those terms were already doing their job and further tightening risked stiffness or crouch.

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

## 3. How I Would Remake This

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

### Reward Design I Would Start With

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

### Reward Weights I Would Try First

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

### What I Do Not Want To Forget

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

