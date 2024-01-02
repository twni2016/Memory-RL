import collections
import numpy as np

Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "terminals", "next_observations", "masks"],
)

StepInfo = collections.namedtuple(
    "StepInfo",
    ["hidden_state", "observation", "prev_action", "prev_reward"],
)


class EpisodeInfo:
    def __init__(self) -> None:
        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminals = []
        self.next_observations = []

    def append(self, observation, action, reward, terminal, next_observation):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        self.next_observations.append(next_observation)

    def wrap(self):
        self.observations = np.stack(self.observations)
        self.actions = np.stack(self.actions)
        self.rewards = np.stack(self.rewards)
        self.terminals = np.stack(self.terminals)
        self.next_observations = np.stack(self.next_observations)
