from typing import Optional

import numpy as np
import gym
from gym.wrappers import RescaleAction

from jaxrl import wrappers


def make_env(
    env_name: str,
    seed: int,
    save_folder: Optional[str] = None,
    store_full_metrics: bool = False,
    flatten: bool = True,
) -> gym.Env:

    env = gym.make(env_name)

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if isinstance(env.action_space, gym.spaces.Box):
        env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = gym.wrappers.RecordVideo(env, save_folder)

    if env.observation_space.dtype in [np.int64, np.float64]:
        env = wrappers.SinglePrecision(env)

    env = wrappers.EpisodeMonitor(env, store_full_metrics)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    print("obs space", env.observation_space)
    print("act space", env.action_space)

    return env
