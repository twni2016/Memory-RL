import functools
from typing import Any, Callable, Optional, Sequence, Tuple, Union
from flax.core.frozen_dict import FrozenDict
from jaxrl.networks.types import PRNGKey

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

from distrax import EpsilonGreedy

from jaxrl.networks.memory_net import MemoryEncoder
from jaxrl.networks import markov_critic
from jaxrl.networks.markov_net import MLP

from jaxrl.networks.types import Params, PRNGKey
from jaxrl.buffers import StepInfo


class MemoryDoubleCritic(MemoryEncoder):
    """
    Used in separate actor-critic for continuous control
    """

    config_critic: FrozenDict

    def setup(self):
        super().setup()

        self.critic = markov_critic.MarkovDoubleCritic(self.config_critic)

    def __call__(
        self,
        observations: jnp.ndarray,  # (B, T, O)
        prev_actions: jnp.ndarray,  # (B, T, A)
        prev_rewards: jnp.ndarray,  # (B, T, 1)
        curr_actions: jnp.ndarray,  # (B, T, A)
        rng: PRNGKey,
    ) -> jnp.ndarray:
        outputs = self.train(observations, prev_actions, prev_rewards, rng)

        qs = self.critic(outputs, curr_actions)

        return qs  # (num_qs, B, T)
