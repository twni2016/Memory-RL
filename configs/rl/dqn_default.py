from configs.rl.name_fns import name_fn
from ml_collections import ConfigDict
from typing import Tuple


def dqn_name_fn(
    config: ConfigDict, max_episode_steps: int, max_training_steps: int
) -> Tuple[ConfigDict, str]:
    config, name = name_fn(config)
    # set eps = 1/T, so that the asymptotic prob to
    # sample fully exploited trajectory during exploration is
    # (1-1/T)^T = 1/e
    config.init_eps = 1.0
    config.end_eps = 1.0 / max_episode_steps
    config.schedule_steps = config.schedule_end * max_training_steps

    return config, name


def get_config():
    config = ConfigDict()
    config.name_fn = dqn_name_fn

    config.algo = "dqn"

    config.critic_lr = 3e-4

    config.config_critic = ConfigDict()
    config.config_critic.hidden_dims = (256, 256)

    config.discount = 0.99
    config.tau = 0.005
    config.schedule_end = 0.1  # at least good for TMaze-like envs

    config.replay_buffer_size = 1e6
    config.replay_buffer_num_episodes = 1e3

    return config
