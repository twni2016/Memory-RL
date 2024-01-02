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

from jaxrl.networks.memory_net import MemoryEncoder
from jaxrl.networks import markov_actor, markov_critic

from jaxrl.buffers import StepInfo

class MemoryActorCriticEncoder(MemoryEncoder):
    observ_dim: int
    action_dim: int
    config_actor: FrozenDict
    config_critic: FrozenDict

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

    def init_params(
        self,
        observations: jnp.ndarray,  # (B, T, O) or (B, T, H, W, C)
        prev_actions: jnp.ndarray,  # (B, T, A)
        prev_rewards: jnp.ndarray,  # (B, T, 1)
        curr_actions: Optional[jnp.ndarray] = None,  # (B, T, A)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Only used during parameter initialization, NOT for training
        """
        outputs = self.train(observations, prev_actions, prev_rewards, rng=None)
        actor_head = self.forward_actor(outputs)
        critic_heads = self.forward_critic(outputs, curr_actions)
        return actor_head, critic_heads

    def forward_actor(
        self,
        outputs: jnp.ndarray,
    ):
        # return actions (B=*, T=*, A)
        return self.actor(outputs)

    def forward_critic(
        self,
        outputs: jnp.ndarray,
        curr_actions: Optional[jnp.ndarray] = None,
    ):
        # return q-values (num_qs, B=*, T=*) or (num_qs, B=*, T=*, A)
        return self.critic(outputs, curr_actions)


class MemoryDeterministicActorDoubleCritic(MemoryActorCriticEncoder):
    def setup(self):
        super().setup()

        self.actor = markov_actor.MarkovDeterministicActor(
            self.config_actor, self.action_dim
        )
        self.critic = markov_critic.MarkovDoubleCritic(self.config_critic)


class MemoryNormalTanhActorDoubleCritic(MemoryActorCriticEncoder):
    state_dependent_std: bool = True

    def setup(self):
        super().setup()

        self.actor = markov_actor.MarkovNormalTanhActor(
            self.config_actor, self.action_dim, self.state_dependent_std
        )
        self.critic = markov_critic.MarkovDoubleCritic(self.config_critic)
