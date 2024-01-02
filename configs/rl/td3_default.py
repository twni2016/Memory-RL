from ml_collections import ConfigDict
from configs.rl.name_fns import name_fn


def get_config():
    config = ConfigDict()
    config.name_fn = name_fn

    config.algo = "td3"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.config_actor = ConfigDict()
    config.config_actor.hidden_dims = (256, 256)
    config.config_actor.add_layernorm = False
    config.config_actor.layernorm_use_bias_scale = True

    config.config_critic = ConfigDict()
    config.config_critic.hidden_dims = (256, 256)
    config.config_critic.add_layernorm = False
    config.config_critic.layernorm_use_bias_scale = True

    # only used in shared encoder setting
    config.config_encoder = ConfigDict()
    config.config_encoder.hidden_dims = (256, 256)

    config.discount = 0.99
    config.tau = 0.005

    config.replay_buffer_size = 1e6
    config.replay_buffer_num_episodes = 1e3

    config.exploration_noise = 0.1
    config.target_noise = 0.2
    config.target_noise_clip = 0.5

    return config
