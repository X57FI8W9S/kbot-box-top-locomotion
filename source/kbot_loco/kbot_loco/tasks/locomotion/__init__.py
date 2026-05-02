"""KBot forward locomotion environments."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-KBot-Forward-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:KBotForwardFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KBotForwardFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-KBot-Forward-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:KBotForwardFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KBotForwardFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-KBot-Forward-Flat-V2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:KBotForwardFlatV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KBotForwardFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-KBot-Forward-Flat-V2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:KBotForwardFlatV2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KBotForwardFlatPPORunnerCfg",
    },
)
