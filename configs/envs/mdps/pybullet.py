import ml_collections
import pybullet_envs
from configs.envs.terminal_fns import infinite_horizon_terminal

ENVS = dict(
    ant="AntBulletEnv-v0",
    cheetah="HalfCheetahBulletEnv-v0",
    hopper="HopperBulletEnv-v0",
    humanoid="HumanoidBulletEnv-v0",
    walker="Walker2DBulletEnv-v0",
)


def create_fn(config):
    del config.create_fn
    return config, ENVS[config.env_name]


def get_config():
    config = ml_collections.ConfigDict()
    config.create_fn = create_fn

    config.env_type = "mdp/bullet"
    config.terminal_fn = infinite_horizon_terminal

    config.eval_interval = 5
    config.save_interval = 10
    config.eval_episodes = 10

    config.env_name = "ant"

    return config
