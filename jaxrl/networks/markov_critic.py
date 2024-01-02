import functools
from typing import Any, Callable, Sequence, Tuple
from flax.core.frozen_dict import FrozenDict

import flax.linen as nn
import jax
import jax.numpy as jnp

from distrax import EpsilonGreedy

from jaxrl.networks.markov_net import MLP
from jaxrl.networks.types import Params, PRNGKey


class MarkovCritic(nn.Module):
    """
    Q(s, a)
    """

    config_critic: FrozenDict

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        config_critic, hidden_dims = self.config_critic.pop("hidden_dims")
        critic = MLP(hidden_dims=(*hidden_dims, 1), **config_critic)(inputs)
        return jnp.squeeze(critic, -1)


class MarkovDoubleCritic(nn.Module):
    """
    Q_{1,2}(s,a)
    """

    config_critic: FrozenDict
    num_qs: int = 2

    @nn.compact
    def __call__(self, observations, actions):

        VmapCritic = nn.vmap(
            MarkovCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        qs = VmapCritic(self.config_critic)(observations, actions)
        return qs  # (num_qs, B)
