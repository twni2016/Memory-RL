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
from jaxrl.agents.sac.temperature import Temperature
from jaxrl.networks import markov_critic, markov_actor
from jaxrl.networks.types import InfoDict, PRNGKey


class MarkovSACLearner(struct.PyTreeNode):
    rng: PRNGKey
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    temp: TrainState

    tau: float
    discount: float
    target_entropy: float
    backup_entropy: bool = struct.field(pytree_node=False)
    update_temperature: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observations: jnp.ndarray,  # (O) -> (B, O)
        actions: jnp.ndarray,  # (A) -> (B, A)
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        config_actor: Optional[dict] = None,
        config_critic: Optional[dict] = None,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        init_temperature: float = 1.0,
        update_temperature: bool = True,
        **kwargs
    ):
        """
        An implementation of [Soft-Actor-Critic](https://arxiv.org/abs/1812.05905)
        """
        observations = observations[None, ...]
        actions = actions[None, ...]
        action_dim = actions.shape[-1]

        if target_entropy is None:
            target_entropy = -action_dim

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_def = markov_actor.MarkovNormalTanhActor(
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

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr)
            if update_temperature
            else F.optim_null,
        )
        F.print_model(temp_def, temp_params)

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            tau=tau,
            discount=discount,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
            update_temperature=update_temperature,
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
            distribution=True,
        )

        actions = jax.device_get(actions)
        actions = np.clip(actions, -1, 1)

        return self.replace(rng=rng), actions

    @jax.jit
    def update(self, batch: Batch) -> InfoDict:
        new_agent = self

        new_agent, critic_info = self.update_critic(new_agent, batch)

        new_agent, actor_info = self.update_actor(new_agent, batch)

        new_agent, temp_info = self.update_temp(new_agent, actor_info["entropy"])

        return new_agent, {**actor_info, **critic_info, **temp_info}

    @staticmethod
    def update_actor(agent, batch: Batch):
        rng, key = jax.random.split(agent.rng)

        def actor_loss_fn(actor_params):
            dist = agent.actor.apply_fn({"params": actor_params}, batch.observations)
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)  # (B)

            q1, q2 = agent.critic.apply_fn(
                {"params": agent.critic.params}, batch.observations, actions
            )
            q = jnp.minimum(q1, q2)

            actor_loss = (
                log_probs * agent.temp.apply_fn({"params": agent.temp.params}) - q
            ).mean()

            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(agent.actor.params)
        actor = agent.actor.apply_gradients(grads=grads)

        new_agent = agent.replace(actor=actor, rng=rng)

        return new_agent, actor_info

    @staticmethod
    def update_critic(agent, batch: Batch):
        rng, key = jax.random.split(agent.rng)
        dist = agent.actor.apply_fn(
            {"params": agent.actor.params}, batch.next_observations
        )
        next_actions = dist.sample(seed=key)

        next_q1, next_q2 = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch.next_observations,
            next_actions,
        )
        next_q = jnp.minimum(next_q1, next_q2)

        target_q = batch.rewards + agent.discount * (1.0 - batch.terminals) * next_q

        if agent.backup_entropy:  # True: SAC; False: SAC-Lite
            next_log_probs = dist.log_prob(next_actions)
            target_q += (
                agent.discount
                * (1.0 - batch.terminals)
                * agent.temp.apply_fn({"params": agent.temp.params})
                * (-next_log_probs)
            )

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

    @staticmethod
    def update_temp(agent, entropy: jnp.ndarray):
        if not agent.update_temperature:
            return agent, {
                "temperature": agent.temp.apply_fn({"params": agent.temp.params})
            }

        def temperature_loss_fn(temp_params):
            temperature = agent.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - agent.target_entropy).mean()
            return temp_loss, {"temperature": temperature, "temp_loss": temp_loss}

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(
            agent.temp.params
        )

        temp = agent.temp.apply_gradients(grads=grads)

        new_agent = agent.replace(temp=temp)

        return new_agent, temp_info
