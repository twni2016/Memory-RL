import time

import gym
import numpy as np

from jaxrl.wrappers.common import TimeStep


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env, store_full_metrics: bool = False):
        super().__init__(env)
        # make the private variable public,
        # as we might use it for finite-horizon problems
        # https://github.com/openai/gym/blob/c33cfd8b2cc8cac6c346bc2182cd568ef33b8821/gym/envs/registration.py#L76
        self.max_episode_steps = getattr(
            self, "max_episode_steps", self.env.spec.max_episode_steps
        )
        # we will use it during training
        self.total_timesteps = 0

        self.store_full_metrics = store_full_metrics

    def _reset_stats(self, observation: np.ndarray):
        self.episode_length = 0
        self.rewards = []
        # self.start_time = time.time()

        if self.store_full_metrics:
            self.observs = [observation]
            self.actions = []

    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, done, info = self.env.step(action)

        self.episode_length += 1
        self.rewards.append(reward)
        self.total_timesteps += 1
        # info["total"] = {"timesteps": self.total_timesteps}

        if self.store_full_metrics:
            self.observs.append(observation)
            self.actions.append(action)

        if done:
            info["episode"] = {}
            info["episode"]["rewards"] = np.stack(self.rewards)
            info["episode"]["return"] = info["episode"]["rewards"].sum()
            info["episode"]["length"] = self.episode_length
            # info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )
            if self.store_full_metrics:
                info["episode"]["observs"] = np.stack(self.observs)
                info["episode"]["actions"] = np.stack(self.actions)

        return observation, reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        observation = self.env.reset()
        self._reset_stats(observation)
        return observation
