import ml_collections
import envs.pomdp
from configs.envs.terminal_fns import infinite_horizon_terminal

ENVS = dict(
    ant="AntBLT-P-v0",
    cheetah="HalfCheetahBLT-P-v0",
    hopper="HopperBLT-P-v0",
    walker="WalkerBLT-P-v0",
)


def create_fn(config):
    del config.create_fn
    return config, ENVS[config.env_name]


def get_config():
    config = ml_collections.ConfigDict()
    config.create_fn = create_fn

    config.env_type = "pomdp/bullet_p"
    config.terminal_fn = infinite_horizon_terminal

    config.eval_interval = 5
    config.save_interval = 10
    config.eval_episodes = 10

    config.env_name = "ant"

    return config
