"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple
from flax.core.frozen_dict import FrozenDict

import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax import struct
from flax.training.train_state import TrainState
import jaxrl.networks.functionals as F

from jaxrl.buffers import Batch
from jaxrl.networks import markov_critic, markov_actor
from jaxrl.networks.types import InfoDict, PRNGKey


class MarkovTD3Learner(struct.PyTreeNode):
    rng: PRNGKey
    actor: TrainState
    target_actor: TrainState
    critic: TrainState
    target_critic: TrainState

    tau: float
    discount: float
    exploration_noise: float
    target_noise: float
    target_noise_clip: float

    @classmethod
    def create(
        cls,
        seed: int,
        observations: jnp.ndarray,  # (O) -> (B, O)
        actions: jnp.ndarray,  # (A) -> (B, A)
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        config_actor: Optional[dict] = None,
        config_critic: Optional[dict] = None,
        discount: float = 0.99,
        tau: float = 0.005,
        exploration_noise: float = 0.1,
        target_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        **kwargs
    ):
        """
        An implementation of [TD3](https://arxiv.org/abs/1802.09477).
        """
        observations = observations[None, ...]
        actions = actions[None, ...]
        action_dim = actions.shape[-1]

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        actor_def = markov_actor.MarkovDeterministicActor(
            config_actor=FrozenDict(config_actor),
            action_dim=action_dim,
        )
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )
        F.print_model(actor_def, actor_params)

        target_actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,  # use same params
            tx=F.optim_null,
        )

        critic_def = markov_critic.MarkovDoubleCritic(
            config_critic=FrozenDict(config_critic),
        )
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        F.print_model(critic_def, critic_params)

        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,  # use same params
            tx=F.optim_null,
        )

        return cls(
            rng=rng,
            actor=actor,
            target_actor=target_actor,
            critic=critic,
            target_critic=target_critic,
            tau=tau,
            discount=discount,
            exploration_noise=exploration_noise,
            target_noise=target_noise,
            target_noise_clip=target_noise_clip,
        )

    def sample_actions(
        self,
        observations: jnp.ndarray,
        mode: bool = False,
    ) -> np.ndarray:
        rng, actions = markov_actor.sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            observations,
            mode,
            distribution=False,
            exploration_noise=self.exploration_noise,
        )

        actions = jax.device_get(actions)
        actions = np.clip(actions, -1, 1)

        return self.replace(rng=rng), actions

    @jax.jit
    def update(self, batch: Batch) -> InfoDict:
        new_agent = self

        new_agent, critic_info = self.update_critic(new_agent, batch)

        new_agent, actor_info = self.update_actor(new_agent, batch)

        return new_agent, {**actor_info, **critic_info}

    @staticmethod
    def update_actor(agent, batch: Batch):
        def actor_loss_fn(actor_params):
            actions = agent.actor.apply_fn({"params": actor_params}, batch.observations)

            q1, q2 = agent.critic.apply_fn(
                {"params": agent.critic.params}, batch.observations, actions
            )
            q = jnp.minimum(q1, q2)
            actor_loss = -q.mean()

            return actor_loss, {"actor_loss": actor_loss}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(agent.actor.params)
        actor = agent.actor.apply_gradients(grads=grads)

        target_actor = F.target_update(actor, agent.target_actor, agent.tau)

        new_agent = agent.replace(actor=actor, target_actor=target_actor)

        return new_agent, actor_info

    @staticmethod
    def update_critic(agent, batch: Batch):
        rng, key = jax.random.split(agent.rng)

        next_actions = agent.target_actor.apply_fn(
            {"params": agent.target_actor.params}, batch.next_observations
        )
        action_noise = jnp.clip(
            jax.random.normal(key, shape=next_actions.shape) * agent.target_noise,
            -agent.target_noise_clip,
            agent.target_noise_clip,
        )
        next_actions = jnp.clip(next_actions + action_noise, -1, 1)

        next_q1, next_q2 = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch.next_observations,
            next_actions,
        )
        next_q = jnp.minimum(next_q1, next_q2)

        target_q = batch.rewards + agent.discount * (1.0 - batch.terminals) * next_q
        # target_q = jax.lax.stop_gradient(target_q) # no need (no speed up / memory reducation)

        def critic_loss_fn(critic_params):
            q1, q2 = agent.critic.apply_fn(
                {"params": critic_params}, batch.observations, batch.actions
            )
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
            }

        grads, critic_info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)

        target_critic = F.target_update(critic, agent.target_critic, agent.tau)

        new_agent = agent.replace(critic=critic, target_critic=target_critic, rng=rng)

        return new_agent, critic_info
