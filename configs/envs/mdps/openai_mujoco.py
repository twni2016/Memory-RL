import ml_collections
from configs.envs.terminal_fns import infinite_horizon_terminal


ENVS = dict(
    ant="Ant-v2",
    cheetah="HalfCheetah-v2",
    hopper="Hopper-v2",
    humanoid="Humanoid-v2",
    swimmer="Swimmer-v2",
    walker="Walker2d-v2",
)


def create_fn(config):
    del config.create_fn
    return config, ENVS[config.env_name]


def get_config():
    config = ml_collections.ConfigDict()
    config.create_fn = create_fn

    config.env_type = "mdp/mujoco"
    config.terminal_fn = infinite_horizon_terminal

    config.eval_interval = 5
    config.save_interval = 10
    config.eval_episodes = 10

    config.env_name = "ant"

    return config
