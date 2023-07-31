from configs.rl.name_fns import name_fn
from ml_collections import ConfigDict
from typing import Tuple


def sacd_name_fn(config: ConfigDict, *args) -> Tuple[ConfigDict, str]:
    config, name = name_fn(config)
    return config, name + f"alpha-{config.init_temperature}/"


def get_config():
    config = ConfigDict()
    config.name_fn = sacd_name_fn

    config.algo = "sacd"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.config_actor = ConfigDict()
    config.config_actor.hidden_dims = (256, 256)

    config.config_critic = ConfigDict()
    config.config_critic.hidden_dims = (256, 256)

    config.discount = 0.99
    config.tau = 0.005

    config.init_temperature = 0.1
    config.update_temperature = False
    config.target_entropy = None

    config.replay_buffer_size = 1e6
    config.replay_buffer_num_episodes = 1e3

    return config
