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
