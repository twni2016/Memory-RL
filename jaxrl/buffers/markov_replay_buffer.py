from typing import Union

import gym
import numpy as np

from jaxrl.buffers.dataset import Batch


class MarkovReplayBuffer:
    buffer_type = "markov"

    def __init__(
        self,
        max_replay_buffer_size: int,
        observation_space: gym.spaces.Box,
        action_space: Union[gym.spaces.Discrete, gym.spaces.Box],
        **kwargs
    ):
        self._max_replay_buffer_size = max_replay_buffer_size

        self._observations = np.zeros(
            (max_replay_buffer_size, *observation_space.shape),
            dtype=observation_space.dtype,
        )
        self._next_observations = np.zeros(
            (max_replay_buffer_size, *observation_space.shape),
            dtype=observation_space.dtype,
        )

        # for discrete action space, this is (size,) of data type int64
        self._actions = np.zeros(
            (max_replay_buffer_size, *action_space.shape), dtype=action_space.dtype
        )
        self._rewards = np.zeros((max_replay_buffer_size,), dtype=np.float32)

        # terminals are "done" signals, useful for policy training
        #   for each trajectory, it has single 1 like 0000001000 for reaching goal or early stopping
        # 	or simply 0s for timing out.
        self._terminals = np.zeros((max_replay_buffer_size,), dtype=np.float32)

        self.clear()

    def add_sample(
        self,
        observation,
        action,
        reward,
        terminal,
        next_observation,
    ):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_observations[self._top] = next_observation

        self._top = (self._top + 1) % self._max_replay_buffer_size
        self._size = min(self._size + 1, self._max_replay_buffer_size)

    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0

    def random_batch(self, batch_size):
        """batch of unordered transitions"""
        # assert self.can_sample_batch(batch_size)
        indices = np.random.randint(0, self._size, batch_size)
        return Batch(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_observations[indices],
            masks=None,
        )

    def can_sample_batch(self, batch_size):
        return self._size >= batch_size
