#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as metadata
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = REPO_ROOT / "isaac_lab" / "IsaacLab"
ISAAC_RSL_RL_DIR = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"

for path in (
    REPO_ROOT / "source" / "kbot_loco",
    ISAACLAB_ROOT / "source" / "isaaclab",
    ISAACLAB_ROOT / "source" / "isaaclab_assets",
    ISAACLAB_ROOT / "source" / "isaaclab_rl",
    ISAACLAB_ROOT / "source" / "isaaclab_tasks",
    ISAAC_RSL_RL_DIR,
    REPO_ROOT / "scripts" / "rsl_rl",
):
    sys.path.insert(0, str(path))

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402


parser = argparse.ArgumentParser(description="Print early inference actions and states from an RSL-RL checkpoint.")
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402,F401
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import kbot_loco  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from rsl_rl_compat import rsl_rl_train_cfg  # noqa: E402

installed_version = version.parse(metadata.version("rsl-rl-lib"))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required.")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    resume_path = retrieve_file_path(args_cli.checkpoint)
    base_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, rsl_rl_train_cfg(agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, rsl_rl_train_cfg(agent_cfg.to_dict()), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    print("ACTION_INSPECT checkpoint=", resume_path, flush=True)
    print("ACTION_INSPECT joint_names=", list(robot.data.joint_names), flush=True)
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
        action_list = [round(float(v), 4) for v in actions[0].detach().cpu().tolist()]
        root = [round(float(v), 4) for v in robot.data.root_pos_w[0].detach().cpu().tolist()]
        joints = [round(float(v), 4) for v in robot.data.joint_pos[0].detach().cpu().tolist()]
        print(f"ACTION_INSPECT step={step} root={root} action={action_list} joints={joints}", flush=True)
        obs, _, _, _ = env.step(actions)

    env.close()


try:
    main()
finally:
    simulation_app.close()
