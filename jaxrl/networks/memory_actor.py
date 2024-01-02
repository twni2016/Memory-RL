import functools
from typing import Any, Callable, Optional, Sequence, Tuple, Union
from flax.core.frozen_dict import FrozenDict
from jaxrl.networks.types import PRNGKey

import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

from jaxrl.networks.memory_net import MemoryEncoder
from jaxrl.networks import markov_actor

from jaxrl.networks.types import Params, PRNGKey
from jaxrl.buffers import StepInfo
from jaxrl.buffers.transform import preprocess_discrete_actions

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class MemoryActorEncoder(MemoryEncoder):
    action_dim: int

    def train(
        self,
        observations: jnp.ndarray,  # (B, T, O) or (B, T, H, W, C)
        prev_actions: jnp.ndarray,  # (B, T, A)
        prev_rewards: jnp.ndarray,  # (B, T, 1)
        rng: PRNGKey,
    ) -> jnp.ndarray:
        """
        During training
        """
        outputs = super().train(observations, prev_actions, prev_rewards, rng)
        return self.forward_actor(outputs)

    def get_initial_info(
        self,
        observation: jnp.ndarray,  # (B=*, O) or (B=*, H, W, C)
    ):
        """
        During evaluation
        """
        if len(observation.shape) >= 3:
            batch_dims = observation.shape[:-3]
        else:
            batch_dims = observation.shape[:-1]
        initial_hidden_state = self.seq_model.initialize_carry(batch_dims)

        return StepInfo(
            hidden_state=initial_hidden_state,
            observation=observation,
            prev_action=jnp.zeros(batch_dims + (self.action_dim,)),
            prev_reward=jnp.zeros(batch_dims + (1,)),
        )

    def __call__(
        self,
        initial_hidden_states: Union[jnp.ndarray, Tuple[jnp.ndarray]],  # (B=*, H)
        observations: jnp.ndarray,  # (B=*, T=*, O)
        prev_actions: jnp.ndarray,  # (B=*, T=*, A)
        prev_rewards: jnp.ndarray,  # (B=*, T=*, 1)
        rng: PRNGKey,
    ) -> jnp.ndarray:
        """
        During evaluation, per step
        """

        (hidden_states, outputs), attn_weights = self.forward_encoder(
            is_training=False,
            initial_hidden_states=initial_hidden_states,
            observations=observations,
            prev_actions=prev_actions,
            prev_rewards=prev_rewards,
            rng=rng,
        )

        return self.forward_actor(outputs), hidden_states, attn_weights

    def forward_actor(
        self,
        outputs: jnp.ndarray,
    ):
        # return actions (B=*, T=*, A)
        raise NotImplementedError


class MemoryDeterministicActor(MemoryActorEncoder):
    config_actor: FrozenDict

    def setup(self):
        super().setup()

        self.actor = markov_actor.MarkovDeterministicActor(
            self.config_actor, self.action_dim
        )

    def forward_actor(
        self,
        outputs: jnp.ndarray,
    ):
        return self.actor(outputs)  # (B=*, T=*, A)


class MemoryNormalTanhActor(MemoryActorEncoder):
    config_actor: FrozenDict
    state_dependent_std: bool = True

    def setup(self):
        super().setup()

        self.actor = markov_actor.MarkovNormalTanhActor(
            self.config_actor, self.action_dim, self.state_dependent_std
        )

    def forward_actor(
        self,
        outputs: jnp.ndarray,
    ):
        return self.actor(outputs)  # dist (B=*, T=*, -)


@functools.partial(jax.jit, static_argnames=("actor_apply_fn", "method"))
def get_initial_info(
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    method: Callable[..., Any],
    observation: jnp.ndarray,  # (T=1, O)
) -> Tuple[PRNGKey, jnp.ndarray]:
    return actor_apply_fn(
        {"params": actor_params},
        observation,
        method=method,
    )


@functools.partial(jax.jit, static_argnames=("actor_apply_fn", "mode", "distribution"))
def sample_actions(
    rng: PRNGKey,
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    hidden_states: Union[jnp.ndarray, Tuple[jnp.ndarray]],  # (*, O)
    observations: jnp.ndarray,  # (*, O)
    prev_actions: jnp.ndarray,  # (*, A)
    prev_rewards: jnp.ndarray,  # (*, 1)
    mode: bool,
    distribution: bool = True,
    exploration_noise: float = 0.1,
) -> Tuple[PRNGKey, jnp.ndarray]:
    if mode:
        dropout_rng = None
    else:
        rng, dropout_rng, action_key = jax.random.split(rng, 3)
    dist, hidden_states, attn_weights = actor_apply_fn(
        {"params": actor_params},
        initial_hidden_states=hidden_states,
        observations=observations,
        prev_actions=prev_actions,
        prev_rewards=prev_rewards,
        rng=dropout_rng,
    )

    if mode:
        if distribution:
            actions = dist.mode()
        else:
            actions = dist
        return rng, (actions, hidden_states, attn_weights)

    if distribution:
        actions = dist.sample(seed=action_key)
    else:
        actions = dist
        actions = (
            actions
            + jax.random.normal(action_key, shape=actions.shape) * exploration_noise
        )
    return rng, (actions, hidden_states, attn_weights)
