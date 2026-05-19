# V1 Hybrid Width/Lane Report

## Purpose

This branch exists to test whether the all-time-best V1 gait can be improved without forcing it into the V2.5 posed-start world.

The task keeps the V1 walking environment as the behavioral base and layers in only the newer lateral foot geometry pressure:

- keep V1 reset pose, command range, action scale, episode length, and gait rewards
- keep V1 command speed range, currently `lin_vel_x = 0.35-0.55 m/s`
- keep V1 root-height target and low-body behavior
- replace foot separation and lane targets with the newer V2.5 width targets
- add the newer sole-center lane max term
- use the existing evaluator and trailing HUD scripts for diagnostics and videos

This should not replace V2.5. It is a separate compatibility branch for V1 checkpoints.

## Task IDs

- Training/evaluation: `Isaac-KBot-Forward-Flat-V1-Hybrid-WidthLane-v0`
- Play variant: `Isaac-KBot-Forward-Flat-V1-Hybrid-WidthLane-Play-v0`

## Config Classes

- `KBotForwardFlatV1HybridWidthLaneEnvCfg`
- `KBotForwardFlatV1HybridWidthLaneEnvCfg_PLAY`

## Geometry Terms

The hybrid task overrides these V1 reward terms:

- `foot_lateral_spacing_l1`
  - weight: `-9.0`
  - target width: `0.3164 m`
- `foot_signed_lateral_clearance_l1`
  - weight: `-12.0`
  - minimum width: `0.28 m`
- `foot_lateral_lane_l1`
  - weight: `-4.0`
  - targets: left `0.1582 m`, right `-0.1582 m`
  - tolerance: `0.08 m`
- `foot_lateral_lane_max_l1`
  - weight: `-2.0`
  - targets: left `0.1582 m`, right `-0.1582 m`
  - tolerance: `0.06 m`
- `foot_sole_lateral_lane_max_l1`
  - weight: `-44.0`
  - targets: left `0.15835 m`, right `-0.15805 m`
  - tolerance: `0.008 m`

## Do Not Confuse With V2.5

V2.5 uses the authored posed start and low-speed gait-quality curriculum. This hybrid does not.

If V1 hybrid improves width/lane behavior but still preserves V1's crouched/fast gait, that is expected. The question for this branch is whether the old gait can be made less narrow/cross-prone without destroying its forward motion.

## 2026-05-19 Inference And Fine Tune

Parent checkpoint:

```text
logs/rsl_rl/kbot_forward_flat/2026-04-29_06-29-05/model_11791.pt
```

Hybrid inference on the parent reproduced the restored V1 gait exactly. The hybrid reward changes do not affect fixed-policy inference, as expected. The parent still walks well but is narrow:

```text
decision = REVIEW_VIDEO
distance_m = 12.7106
speed_tracking_ratio = 0.9569
step_root_advance_mean_m = 0.0467
cycle_cadence_hz = 4.5363
lateral_drift_m_per_m = 0.0671
fsep_mean_m = 0.1913
fsep_p05_m = 0.1752
ksep_mean_m = 0.2818
approved_step_fraction = 0.9926
```

Fine-tune run:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-19_14-21-50_v1_hybrid_width_lane_from_11791
```

Training was stable for the 200-iteration pass: full 400-step episodes and timeout-only terminations. The width and lane reward terms improved during training, but the later checkpoints traded that improvement for lateral drift. The current keeper is `model_11850.pt`, not the final checkpoint.

| checkpoint | decision | fsep mean | fsep p05 | step adv | cadence Hz | lateral drift | speed ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| `11791` | REVIEW_VIDEO | 0.1913 | 0.1752 | 0.0467 | 4.5363 | 0.0671 | 0.9569 |
| `11800` | REVIEW_VIDEO | 0.1963 | 0.1777 | 0.0484 | 4.4251 | 0.0574 | 0.9690 |
| `11850` | REVIEW_VIDEO | 0.2146 | 0.1971 | 0.0488 | 4.4251 | -0.0510 | 0.9730 |
| `11900` | REJECT | 0.2341 | 0.2171 | 0.0453 | 4.3354 | 0.2634 | 0.9119 |
| `11950` | REJECT | 0.2493 | 0.2319 | 0.0438 | 4.2962 | 0.2539 | 0.8722 |
| `11990` | REJECT | 0.2630 | 0.2468 | 0.0415 | 4.4236 | 0.3392 | 0.8793 |

Interpretation: the hybrid fine tune can widen V1, but continuing too far lets lateral drift become the mechanism for satisfying width/lane pressure. `model_11850.pt` is the best compromise so far: it improves `fsep_mean_m` by about `+0.023 m` over `11791`, preserves step advance and cadence, and remains in `REVIEW_VIDEO` instead of `REJECT`.

Keeper video:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-19_14-21-50_v1_hybrid_width_lane_from_11791/videos/play/trailing-hud-model_11850-v1-hybrid-width-lane.mp4
```

Do not promote `11900+` from this branch unless a later correction keeps the added width while bringing lateral drift back down. A next attempt should either stop near `11850` and reduce width pressure, or add a stronger straightness-preserving term before pushing `fsep` closer to `0.3164 m`.

## 2026-05-19 Continuation From `11850`

Run:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-19_14-51-24_v1_hybrid_width_lane_continue_from_11850
```

This continuation did not add new gait shaping. It kept the same hybrid width/lane rewards and let the policy try to recover `y=0` behavior using the existing V1 straightness pressure:

```text
lateral_velocity_l2 = -7.0
yaw_rate_l2 = -7.0
world_heading_l2 = -32.0
root_lateral_position_l2 = not enabled in this hybrid task
```

The direct centerline reward already exists in code as `mdp.root_lateral_position_l2`, but it is currently only enabled in later V2 cleanup configs. The V1 hybrid branch is therefore penalizing sideways velocity and yaw/heading error, not accumulated root `y` displacement.

Evaluation results:

| checkpoint | decision | fsep mean | fsep p05 | step adv | cycle adv | cadence Hz | lateral drift | speed ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| previous `11850` | REVIEW_VIDEO | 0.2146 | 0.1971 | 0.0488 | 0.0978 | 4.4251 | -0.0510 | 0.9730 |
| new `11900` | REJECT | 0.2199 | 0.1996 | -0.0034 | -0.0067 | 4.2546 | invalid | 1.1929 |
| new `11999` | REVIEW_VIDEO | 0.2507 | 0.2322 | 0.0485 | 0.0971 | 4.2114 | 0.0684 | 0.9089 |

Interpretation: `model_11900.pt` was a transient failure, but the final checkpoint recovered useful forward walking while keeping the wider feet. `model_11999.pt` is now the best V1-hybrid width checkpoint by metrics: it closes most of the gap toward the `0.3164 m` target without requiring a direct `y=0` displacement reward yet.

Video:

```text
logs/rsl_rl/kbot_forward_flat/2026-05-19_14-51-24_v1_hybrid_width_lane_continue_from_11850/videos/play/trailing-hud-model_11999-v1-hybrid-width-lane.mp4
```

Next decision: review the `11999` video. If the path drift is visually acceptable, continue from `11999` with the same reward first. If it starts walking sideways or arcing, add `root_lateral_position_l2` to the hybrid config with a conservative weight before increasing width pressure further.
