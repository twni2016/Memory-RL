from ml_collections import ConfigDict
from typing import Tuple
from gym.envs.registration import register
from configs.envs.terminal_fns import finite_horizon_terminal

env_name_fn = lambda l: f"T-{l}-v0"


def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    length = config.env_name
    env_name = env_name_fn(length)
    register(
        env_name,
        entry_point="envs.tmaze:TMazeClassicActive",
        kwargs=dict(
            corridor_length=length,
            penalty=-1.0 / length,  # NOTE: \sum_{t=1}^T -1/T = -1
            distract_reward=config.distract_reward,
        ),
        max_episode_steps=length + 2 * 1 + 1,  # NOTE: has to define it here
    )

    del config.create_fn
    return config, env_name


def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "tmaze_active"
    config.terminal_fn = finite_horizon_terminal

    config.eval_interval = 10
    config.save_interval = 10
    config.eval_episodes = 10

    # [1, 2, 5, 10, 30, 50, 100, 300, 500, 1000]
    config.env_name = 10
    config.distract_reward = 0.0

    return config
