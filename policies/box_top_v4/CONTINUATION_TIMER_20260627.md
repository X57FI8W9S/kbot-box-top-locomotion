# Continuation Reminder - 2026-06-27

Purpose: resume the staged V4 training campaign after any quota/context reset.

Active user request:

```text
Train last model_300 to 800, then graph/video; then 1200, graph/video;
then 2500, graph/video; then 5000, graph/video; then 10000, graph/video.
Afterward do a careful sim-step math/physics analysis, propose weights, and
train a candidate branch from the best checkpoint.
```

Important constraint:

```text
Do not blindly continue from a collapsed checkpoint. If a stage collapses,
render/check earlier saved checkpoints from that stage and continue from the
best viable checkpoint instead.
```

Current active run when note was written:

```text
logs/rsl_rl/kbot_forward_flat/2026-06-27_01-22-49_K1.L11.1
```

Source checkpoint:

```text
logs/rsl_rl/kbot_forward_flat/2026-06-27_01-04-35_K1.L11/model_300.pt
```

Active command:

```bash
.venv/bin/python scripts/rsl_rl/train.py \
  --task Isaac-KBot-Forward-Flat-V4-Top4Starter-v0 \
  --headless \
  --num_envs 4096 \
  --resume \
  --load_run 2026-06-27_01-04-35_K1.L11 \
  --checkpoint model_300.pt \
  --max_iterations 501 \
  --save_interval 25 \
  --run_name v4_top4_decim2_phase_latched_precycle0_from_m300_to_800_save25_20260627
```

Observed before any reset:

```text
The 300-to-800 stage began normally but collapsed by the mid-330s.
Training scalar stream around iterations 339-375 showed:
  Episode_Termination/bad_orientation ~= 1.0
  Episode_Termination/time_out ~= 0.0
  gait_cycle_support near 0 because episodes terminate before real phase work
  track_lin_vel_xy_exp and world_forward_velocity_clip still positive

This means later checkpoints from this branch may optimize a fallen/tilted
short-episode policy. Before continuing to 1200, render/evaluate saved
checkpoints around 300, 325, 350, 375, and the final 800 if it completes.
```

Resume steps:

```text
1. Check whether train.py is still running:
   pgrep -af 'train.py|play_trailing.py'

2. If the run completed, generate reward graphs:
   .venv/bin/python scripts/diagnostics/plot_reward_components.py --run-dir <run>

3. Render at least model_800 and likely model_325/model_350 if collapse is visible.

4. Use metrics to pick the best continuation checkpoint, not necessarily the
   latest checkpoint.

5. If all post-300 checkpoints collapse, do the sim-step math analysis from
   model_300 behavior and propose safer weights before continuing.
```

