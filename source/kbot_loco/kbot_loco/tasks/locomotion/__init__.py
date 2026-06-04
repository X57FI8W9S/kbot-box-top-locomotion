"""KBot forward locomotion environments."""

import gymnasium as gym

from . import agents


def _register(task_id: str, env_cfg: str, runner_cfg: str = "KBotForwardFlatPPORunnerCfg") -> None:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg}",
        },
    )


_register("Isaac-KBot-Forward-Flat-v0", "KBotForwardFlatEnvCfg")
_register("Isaac-KBot-Forward-Flat-Play-v0", "KBotForwardFlatEnvCfg_PLAY")
_register("Isaac-KBot-Forward-Flat-V2-v0", "KBotForwardFlatV2EnvCfg")
_register("Isaac-KBot-Forward-Flat-V2-Play-v0", "KBotForwardFlatV2EnvCfg_PLAY")
_register("Isaac-KBot-Forward-Flat-V2-Scratch-PoseBootstrap-v0", "KBotForwardFlatV2ScratchPoseBootstrapEnvCfg")
_register(
    "Isaac-KBot-Forward-Flat-V2_5-Scratch-PoseWidthBootstrap-v0",
    "KBotForwardFlatV25ScratchPoseWidthBootstrapEnvCfg",
    "KBotForwardFlatConservativePPORunnerCfg",
)
_register(
    "Isaac-KBot-Forward-Flat-V2_5-PoseGaitQuality648Compat-v0",
    "KBotForwardFlatV25PoseGaitQuality648CompatEnvCfg",
    "KBotForwardFlatFineTunePPORunnerCfg",
)
_register(
    "Isaac-KBot-Forward-Flat-V3-648HandTuned-v0",
    "KBotForwardFlatV3HandTuned648EnvCfg",
    "KBotForwardFlatConservativePPORunnerCfg",
)
