from __future__ import annotations

from copy import deepcopy


def rsl_rl_train_cfg(cfg: dict) -> dict:
    """Translate Isaac Lab's older combined policy config to newer RSL-RL actor/critic config."""
    cfg = deepcopy(cfg)
    if "actor" in cfg and "critic" in cfg:
        return cfg

    policy = cfg.pop("policy", None)
    if policy is None:
        return cfg

    actor_hidden_dims = policy.get("actor_hidden_dims", [256, 256, 256])
    critic_hidden_dims = policy.get("critic_hidden_dims", actor_hidden_dims)
    activation = policy.get("activation", "elu")
    actor_obs_normalization = policy.get("actor_obs_normalization", False)
    critic_obs_normalization = policy.get("critic_obs_normalization", False)
    init_noise_std = policy.get("init_noise_std", 1.0)
    noise_std_type = policy.get("noise_std_type", "scalar")
    state_dependent_std = policy.get("state_dependent_std", False)
    distribution_class = "HeteroscedasticGaussianDistribution" if state_dependent_std else "GaussianDistribution"

    cfg["actor"] = {
        "class_name": "MLPModel",
        "hidden_dims": actor_hidden_dims,
        "activation": activation,
        "obs_normalization": actor_obs_normalization,
        "distribution_cfg": {
            "class_name": distribution_class,
            "init_std": init_noise_std,
            "std_type": noise_std_type,
        },
    }
    cfg["critic"] = {
        "class_name": "MLPModel",
        "hidden_dims": critic_hidden_dims,
        "activation": activation,
        "obs_normalization": critic_obs_normalization,
    }
    cfg["obs_groups"] = cfg.get("obs_groups") or {"actor": ["policy"], "critic": ["policy"]}
    return cfg

