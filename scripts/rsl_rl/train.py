#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


POLICY_ONLY_RESUME = "--policy_only_resume" in sys.argv
if POLICY_ONLY_RESUME:
    sys.argv.remove("--policy_only_resume")

REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = REPO_ROOT / "isaac_lab" / "IsaacLab"
ISAAC_RSL_RL_DIR = REPO_ROOT / "isaac_lab" / "IsaacLab" / "scripts" / "reinforcement_learning" / "rsl_rl"

for path in (
    REPO_ROOT / "source" / "kbot_loco",
    ISAACLAB_ROOT / "source" / "isaaclab",
    ISAACLAB_ROOT / "source" / "isaaclab_assets",
    ISAACLAB_ROOT / "source" / "isaaclab_rl",
    ISAACLAB_ROOT / "source" / "isaaclab_tasks",
    ISAAC_RSL_RL_DIR,
):
    sys.path.insert(0, str(path))

import isaacsim  # noqa: F401,E402
import kbot_loco  # noqa: F401,E402
import cli_args  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl.runners import DistillationRunner  # noqa: E402

from rsl_rl_compat import rsl_rl_train_cfg  # noqa: E402


_update_rsl_rl_cfg = cli_args.update_rsl_rl_cfg


def _update_rsl_rl_cfg_with_device(agent_cfg, args_cli):
    agent_cfg = _update_rsl_rl_cfg(agent_cfg, args_cli)
    if getattr(args_cli, "device", None) is not None:
        agent_cfg.device = args_cli.device
    return agent_cfg


cli_args.update_rsl_rl_cfg = _update_rsl_rl_cfg_with_device

_on_policy_runner_init = OnPolicyRunner.__init__
_distillation_runner_init = DistillationRunner.__init__
_on_policy_runner_load = OnPolicyRunner.load


def _on_policy_runner_init_compat(self, env, train_cfg, log_dir=None, device="cpu"):
    return _on_policy_runner_init(self, env, rsl_rl_train_cfg(train_cfg), log_dir=log_dir, device=device)


def _distillation_runner_init_compat(self, env, train_cfg, log_dir=None, device="cpu"):
    return _distillation_runner_init(self, env, rsl_rl_train_cfg(train_cfg), log_dir=log_dir, device=device)


def _on_policy_runner_load_compat(self, path, load_cfg=None, strict=True, map_location=None):
    if POLICY_ONLY_RESUME and load_cfg is None:
        load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": False}
    return _on_policy_runner_load(self, path, load_cfg=load_cfg, strict=strict, map_location=map_location)


OnPolicyRunner.__init__ = _on_policy_runner_init_compat
DistillationRunner.__init__ = _distillation_runner_init_compat
OnPolicyRunner.load = _on_policy_runner_load_compat

runpy.run_path(str(ISAAC_RSL_RL_DIR / "train.py"), run_name="__main__")
