import functools
from typing import Any, Callable, Optional, Sequence, Tuple
from flax.core.frozen_dict import FrozenDict

import flax.linen as nn
import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp
from .distributions.tanh_transform import TanhTransformedDistribution

tfd = tfp.distributions

from jaxrl.networks.markov_net import MLP
from jaxrl.networks.initializer import default_kernel_init, default_bias_init
from jaxrl.networks.types import Params, PRNGKey

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class MarkovDeterministicActor(nn.Module):
    config_actor: FrozenDict
    action_dim: int

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
    ) -> jnp.ndarray:
        outputs = MLP(**self.config_actor, activate_final=True)(observations)

        actions = nn.Dense(
            self.action_dim,
            name="action_last_layer",
            kernel_init=default_kernel_init,
            bias_init=default_bias_init,
        )(outputs)
        return nn.tanh(actions)


class MarkovNormalTanhActor(nn.Module):
    config_actor: FrozenDict
    action_dim: int
    state_dependent_std: bool = True

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
    ) -> tfd.Distribution:
        outputs = MLP(**self.config_actor, activate_final=True)(observations)

        means = nn.Dense(
            self.action_dim,
            name="action_mean_layer",
            kernel_init=default_kernel_init,
            bias_init=default_bias_init,
        )(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim,
                name="action_logstd_layer",
                kernel_init=default_kernel_init,
                bias_init=default_bias_init,
            )(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))

        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        base_dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return TanhTransformedDistribution(base_dist)


@functools.partial(jax.jit, static_argnames=("actor_apply_fn", "mode", "distribution"))
def sample_actions(
    rng: PRNGKey,
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    observations: jnp.ndarray,
    mode: bool,
    distribution: bool = True,
    exploration_noise: float = 0.1,
) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_apply_fn({"params": actor_params}, observations)

    if mode:
        if distribution:
            actions = dist.mode()
        else:
            actions = dist
        return rng, actions

    rng, key = jax.random.split(rng)
    if distribution:
        actions = dist.sample(seed=key)
    else:
        actions = dist
        actions = (
            actions + jax.random.normal(key, shape=actions.shape) * exploration_noise
        )
    return rng, actions
