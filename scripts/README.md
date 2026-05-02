# Shared Scripts

These scripts are shared tooling, not policy-specific experiment notes.

```text
rsl_rl/train.py          Launch RSL-RL training.
rsl_rl/play.py           Basic playback.
rsl_rl/play_trailing.py  Side-by-side trailing/side HUD video and metrics.
probe_kbot_stability.py  Static/stability probe helper.
```

Keep this directory runnable. Several scripts compute `REPO_ROOT` from their current path, so moving them requires code updates.

