from typing import Union

import gym
import numpy as np

from jaxrl.buffers.dataset import Batch


class RAMEfficient_SeqReplayBuffer:
    buffer_type = "seq_efficient"

    def __init__(
        self,
        max_replay_buffer_size: int,
        observation_space: gym.spaces.Box,
        action_space: Union[gym.spaces.Discrete, gym.spaces.Box],
        sampled_seq_len: int,
    ):
        """
        this buffer is used for sequence/trajectory/episode:
                it stored the whole sequence
                into the buffer (not transition), and can sample (sub)sequences
                that has 3D shape (batch_size, sampled_seq_len, *)
                based on some rules below.
        it still uses 2D size as normal (max_replay_buffer_size, dim)
                but tracks the sequences

        NOTE: it only save observations once, no next_observation to reduce RAM by ~2,
            especially useful for image observations

        """
        self._max_replay_buffer_size = max_replay_buffer_size

        # for pixel observation space, this is (H, W, C) of data type int
        # for state observation space, this is (O,) of data type float
        self.observ_dims = observation_space.shape
        self._observations = np.zeros(
            (max_replay_buffer_size, *self.observ_dims),
            dtype=observation_space.dtype,
        )

        # for discrete action space, this is () of data type int
        # for continuous action space, this is (A,) of data type float
        self.action_dims = action_space.shape
        self._actions = np.zeros(
            (max_replay_buffer_size, *self.action_dims), dtype=action_space.dtype
        )
        self._rewards = np.zeros((max_replay_buffer_size,), dtype=np.float32)

        # terminals are "done" signals, useful for policy training
        # For each trajectory, it has single 1 like 0000001000 for reaching goal or early stopping
        # 	or simply 0s for timing out.
        # NOTE: so we cannot use terminals to determine the trajectory boundary!
        self._terminals = np.zeros((max_replay_buffer_size), dtype=np.float32)

        # NOTE: We use ends to label the end of each episode to determine the boundary
        self._ends = np.zeros((max_replay_buffer_size), dtype=np.uint8)

        # NOTE: valid_starts are (internal) masks
        # 	which is 1 if we can SAMPLE the (sub)sequence FROM this index else 0.
        # For each trajectory, the first index has valid_start as 1,
        # 	the LAST sampled_seq_len indices are 0s, and the middle ones are 1s
        # 	That is to say, if its length <= sampled_seq_len, then the valid_starts looks like 100000000
        # 	else looks like 11111000000 (have sampled_seq_len - 1 trailing zeros)
        # See _compute_valid_starts function for details
        self._valid_starts = np.zeros((max_replay_buffer_size), dtype=np.float32)

        assert sampled_seq_len >= 2
        self._sampled_seq_len = sampled_seq_len

        self.clear()

        # optional: calculate RAM usage
        RAM = 0.0
        for name, var in vars(self).items():
            if isinstance(var, np.ndarray):
                RAM += var.nbytes
        print(f"buffer RAM usage: {RAM / 1024 ** 3 :.2f} GB")
        print(f"buffer size {self._max_replay_buffer_size}")

    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0

    def add_episode(self, observations, actions, rewards, terminals, next_observations):
        """
        NOTE: must add one whole episode/sequence/trajectory,
                        not some partial transitions
        the length of different episode can vary, but must be greater than 2
                so that the end of valid_starts is 0.

        all the inputs have 1D/2D shape of (L, *)
        """
        assert (
            observations.shape[0]
            == actions.shape[0]
            == rewards.shape[0]
            == terminals.shape[0]
            == next_observations.shape[0]
        )

        seq_len = observations.shape[0]  # L
        indices = list(
            np.arange(self._top, self._top + seq_len) % self._max_replay_buffer_size
        )

        self._observations[indices] = observations
        self._actions[indices] = actions
        self._rewards[indices] = rewards
        self._terminals[indices] = terminals
        self._valid_starts[indices] = self._compute_valid_starts(seq_len)
        self._ends[indices] = 0

        self._top = (self._top + seq_len) % self._max_replay_buffer_size

        # add final transition: obs is useful but the others are just padding
        self._observations[self._top] = next_observations[-1]  # final obs
        self._actions[self._top] = 0.0
        self._rewards[self._top] = 0.0
        self._terminals[self._top] = 1.0  # not used
        self._valid_starts[self._top] = 0.0  # never be sampled as starts
        self._ends[self._top] = 1  # the end of one episode

        self._top = (self._top + 1) % self._max_replay_buffer_size
        self._size = min(self._size + seq_len + 1, self._max_replay_buffer_size)

    def _compute_valid_starts(self, seq_len):
        valid_starts = np.ones((seq_len), dtype=np.float32)

        num_valid_starts = int(max(1, seq_len - self._sampled_seq_len + 1))

        # set the num_valid_starts: indices are zeros
        valid_starts[num_valid_starts:] = 0.0

        return valid_starts

    def random_batch(self, batch_size):
        """
        return each item has 2D/3D shape (batch_size, sampled_seq_len, *)
        """
        sampled_episode_starts = self._sample_indices(batch_size)  # (B,)

        # get sequential indices
        indices = []
        next_indices = []  # for next obs
        for start in sampled_episode_starts:  # small loop
            end = start + self._sampled_seq_len  # continuous + T
            indices += list(np.arange(start, end) % self._max_replay_buffer_size)
            next_indices += list(
                np.arange(start + 1, end + 1) % self._max_replay_buffer_size
            )

        return Batch(
            observations=self._observations[indices].reshape(
                (batch_size, self._sampled_seq_len) + self.observ_dims
            ),
            actions=self._actions[indices].reshape(
                (batch_size, self._sampled_seq_len) + self.action_dims
            ),
            rewards=self._rewards[indices].reshape(
                batch_size, self._sampled_seq_len, 1
            ),
            terminals=self._terminals[indices].reshape(
                batch_size, self._sampled_seq_len
            ),
            next_observations=self._observations[next_indices].reshape(
                (batch_size, self._sampled_seq_len) + self.observ_dims
            ),
            masks=self._generate_masks(indices, batch_size),  # (B, T)
        )

    def _sample_indices(self, batch_size):
        # self._top points at the start of a new sequence
        # self._top - 1 is the end of the recently stored sequence
        valid_starts_indices = np.where(self._valid_starts > 0.0)[0]

        return np.random.choice(valid_starts_indices, size=batch_size)

    def _generate_masks(self, indices, batch_size):
        """
        input: sampled_indices list of len B*T
        output: masks (B, T)
        """

        # get ends of sampled sequences (B, T)
        # each row starts with 0, like 0000000 or 0000010001
        sampled_seq_ends = (
            np.copy(self._ends[indices])
            .reshape(batch_size, self._sampled_seq_len)
            .astype(np.float32)
        )

        # build masks
        masks = np.ones_like(sampled_seq_ends)  # (B, T), default is 1

        # we want to find the boundary (ending) of sampled sequences
        # 	i.e. **the FIRST 1 after 0** (if exists)
        # 	this is important for varying length episodes
        # the boundary (ending) appears at the FIRST -1 in diff
        diff = sampled_seq_ends[:, :-1] - sampled_seq_ends[:, 1:]  # (B, T-1)
        # add 0s into the first column
        diff = np.concatenate([np.zeros((batch_size, 1)), diff], axis=1)  # (B, T)

        # now the start of next episode appears at the FIRST -1 in diff
        invalid_starts_b, invalid_starts_t = np.where(
            diff == -1.0
        )  # (1D array in batch dim, 1D array in seq dim)
        invalid_indices_b = []
        invalid_indices_t = []
        last_batch_index = -1

        for batch_index, start_index in zip(invalid_starts_b, invalid_starts_t):
            if batch_index == last_batch_index:
                # for same batch_idx, we only care the first appearance of -1
                continue
            last_batch_index = batch_index

            invalid_indices = list(
                np.arange(start_index, self._sampled_seq_len)
            )  # to the end
            # extend to the list
            invalid_indices_b += [batch_index] * len(invalid_indices)
            invalid_indices_t += invalid_indices

        # set invalids in the masks
        masks[invalid_indices_b, invalid_indices_t] = 0.0

        return masks


if __name__ == "__main__":
    buffer_size = 100
    sampled_seq_len = 7
    buffer = RAMEfficient_SeqReplayBuffer(
        buffer_size,
        np.zeros((1,)),
        np.zeros((1,)),
        sampled_seq_len,
    )
    for l in range(sampled_seq_len - 1, sampled_seq_len + 5):
        print(l)
        assert buffer._compute_valid_starts(l)[0] > 0.0
        print(buffer._compute_valid_starts(l))
    for _ in range(200):
        e = np.random.randint(3, 10)
        buffer.add_episode(
            np.arange(e).reshape(e, 1),
            np.zeros((e, 1)),
            np.zeros((e,)),
            np.zeros((e,)),
            np.arange(1, e + 1).reshape(e, 1),
        )
    print(buffer._size, buffer._top)
    print(
        np.concatenate(
            [buffer._observations, buffer._valid_starts[:, np.newaxis]], axis=-1
        )
    )

    for _ in range(10):
        batch = buffer.random_episodes(1)  # (B, T, *)
        print(batch.observations[0, :, 0])
        print(batch.next_observations[0, :, 0])
        print(batch.masks[0, :])
        print("\n")
