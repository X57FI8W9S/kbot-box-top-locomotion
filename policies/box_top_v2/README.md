# Box-Top Policy V2

This is the restart area for the new policy.

The main change for v2 is process discipline:

1. Keep diagnostics separate from rewards.
2. Add an evaluator/scorecard before tuning another large reward stack.
3. Use hard gates for safety, yaw/lateral drift, contact quality, roll bias, L/R symmetry, crouch, and gait event regularity.
4. Compare checkpoints against the current best baseline using fixed rollouts, metrics, plots, and video.

Prompt source:

```text
prompts/new_policy_prompt.pdf
prompts/new_policy_prompt.txt
```

Design notes:

```text
design/diagnostics_plan.md
FINAL_REPORT_DRAFT.md
```

Implementation note:

Do not silently overwrite the v1 task config in `source/kbot_loco/kbot_loco/tasks/locomotion`. When v2 reward/task code starts, create a clearly named v2 config/task or document exactly which shared files are being intentionally upgraded.
