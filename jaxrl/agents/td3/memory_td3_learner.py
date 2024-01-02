"""Implementations of algorithms for continuous control."""

import functools
from typing import Callable, Sequence, Tuple, Union, Optional
from ml_collections import ConfigDict
from flax.core.frozen_dict import FrozenDict

import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax import struct
from flax.training.train_state import TrainState
import jaxrl.networks.functionals as F

from jaxrl.buffers import Batch
from jaxrl.networks import memory_actor, memory_critic
from jaxrl.networks.types import InfoDict, PRNGKey
from jaxrl.buffers.transform import zero_prepend

from .markov_td3_learner import MarkovTD3Learner


class MemoryTD3Learner(MarkovTD3Learner):
    # some functions
    behavior_actor_get_initial_info: Callable = struct.field(pytree_node=False)
    behavior_actor_apply: Callable = struct.field(pytree_node=False)
    actor_train: Callable = struct.field(pytree_node=False)

    use_dropout: bool = struct.field(pytree_node=False)
    clip: bool = struct.field(pytree_node=False)
    max_norm: float = struct.field(pytree_node=False)
    use_l2_norm: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observations: jnp.ndarray,  # (O) -> (B, T, O)
        actions: jnp.ndarray,  # (A) -> (B, T, A)
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        config_actor: Optional[dict] = None,
        config_critic: Optional[dict] = None,
        discount: float = 0.99,
        tau: float = 0.005,
        exploration_noise: float = 0.1,
        target_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        config_seq: Optional[ConfigDict] = None,
        **kwargs
    ):
        observations = observations[None, None, ...]
        actions = actions[None, None, ...]
        action_dim = actions.shape[-1]
        rewards = jnp.zeros((1, 1, 1))  # (B=1, T=1, 1)

        clip = config_seq.clip
        max_norm = config_seq.max_norm
        use_l2_norm = config_seq.use_l2_norm

        seq_model_dict = FrozenDict(config_seq.model.to_dict())

        rng = jax.random.PRNGKey(seed)
        use_dropout = config_seq.use_dropout
        if use_dropout:
            (
                rng,
                actor_key,
                critic_key,
                actordropout_key,
                criticdropout_key,
            ) = jax.random.split(rng, 5)
            init_actor_rngs = {"params": actor_key, "dropout": actordropout_key}
            init_critic_rngs = {"params": critic_key, "dropout": criticdropout_key}
        else:
            rng, actor_key, critic_key = jax.random.split(rng, 3)
            init_actor_rngs = actor_key
            init_critic_rngs = critic_key

        actor_def = memory_actor.MemoryDeterministicActor(
            config_actor=FrozenDict(config_actor),
            action_dim=action_dim,
            **seq_model_dict,
        )
        actor_params = actor_def.init(
            init_actor_rngs,
            observations,
            actions,
            rewards,
            rng=None,
            method=actor_def.train,
        )["params"]
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

        critic_def = memory_critic.MemoryDoubleCritic(
            config_critic=FrozenDict(config_critic),
            **seq_model_dict,
        )
        critic_params = critic_def.init(
            init_critic_rngs, observations, actions, rewards, actions, rng=None
        )["params"]
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
            # td3 related
            exploration_noise=exploration_noise,
            target_noise=target_noise,
            target_noise_clip=target_noise_clip,
            # memory related
            behavior_actor_get_initial_info=actor_def.get_initial_info,
            behavior_actor_apply=actor_def.apply,
            actor_train=actor_def.train,
            # bools
            use_dropout=use_dropout,
            clip=clip,
            max_norm=max_norm,
            use_l2_norm=use_l2_norm,
        )

    def get_initial_info(self, observation: jnp.array):
        return memory_actor.get_initial_info(
            self.behavior_actor_apply,
            self.actor.params,
            self.behavior_actor_get_initial_info,
            observation,  # (B=*, O)
        )

    def sample_actions(
        self,
        hidden_states: Union[jnp.ndarray, Tuple[jnp.ndarray]],  # (*, H)
        observations: np.ndarray,  # (*, O)
        prev_actions: np.ndarray,  # (*, A)
        prev_rewards: np.ndarray,  # (*, 1)
        mode: bool = False,
    ) -> jnp.ndarray:
        rng, (actions, new_hidden_states, attn_weights,) = memory_actor.sample_actions(
            self.rng,
            self.behavior_actor_apply,
            self.actor.params,
            hidden_states,
            observations,
            prev_actions,
            prev_rewards,
            mode,
            distribution=False,
            exploration_noise=self.exploration_noise,
        )

        actions = jax.device_get(actions)
        actions = np.clip(actions, -1, 1)

        return self.replace(rng=rng), actions, (new_hidden_states, attn_weights)

    @jax.jit
    def update(self, batch: Batch) -> InfoDict:
        batch = zero_prepend(batch)  # preprocess to (B, T+1, dim)

        new_agent = self

        new_agent, critic_info = new_agent.update_critic(batch)

        new_agent, actor_info = new_agent.update_actor(batch)

        return new_agent, {**actor_info, **critic_info}

    def update_actor(self, batch: Batch):
        agent = self
        num_valid = jnp.clip(batch.masks.sum(), a_min=1.0)

        if agent.use_dropout:
            rng, dropout_key1, dropout_key2 = jax.random.split(agent.rng, 3)
        else:
            rng = agent.rng

        def actor_loss_fn(actor_params):
            actions = agent.actor.apply_fn(
                {"params": actor_params},
                observations=batch.observations,
                prev_actions=batch.actions,
                prev_rewards=batch.rewards,
                rng=dropout_key1 if agent.use_dropout else None,
                method=agent.actor_train,
            )  # (B, T+1, A)

            q1, q2 = agent.critic.apply_fn(
                {"params": agent.critic.params},
                observations=batch.observations,
                prev_actions=batch.actions,
                prev_rewards=batch.rewards,
                curr_actions=actions,
                rng=dropout_key2 if agent.use_dropout else None,
            )  # (B, T+1)
            q = jnp.minimum(q1, q2)[:, :-1]  # (B, T)
            actor_loss = (-q * batch.masks).sum() / num_valid

            return actor_loss, {"actor_loss": actor_loss}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(agent.actor.params)

        clipped_grads, norm = F.clip_grad_norm(
            grads, max_norm=agent.max_norm, use_l2_norm=agent.use_l2_norm
        )
        actor_info.update({"actor_grad_norm": norm})

        grads = clipped_grads if agent.clip else grads
        actor = agent.actor.apply_gradients(grads=grads)
        target_actor = F.target_update(actor, agent.target_actor, agent.tau)
        agent = agent.replace(actor=actor, target_actor=target_actor, rng=rng)

        return agent, actor_info

    def update_critic(self, batch: Batch):
        agent = self
        num_valid = jnp.clip(batch.masks.sum(), a_min=1.0)

        if agent.use_dropout:
            (
                rng,
                action_key,
                dropout_key1,
                dropout_key2,
                dropout_key3,
            ) = jax.random.split(agent.rng, 5)
        else:
            rng, action_key = jax.random.split(agent.rng)

        next_actions = agent.target_actor.apply_fn(
            {"params": agent.target_actor.params},
            observations=batch.observations,
            prev_actions=batch.actions,
            prev_rewards=batch.rewards,
            rng=dropout_key1 if agent.use_dropout else None,
            method=agent.actor_train,
        )  # (B, T+1, A)
        action_noise = jnp.clip(
            jax.random.normal(action_key, shape=next_actions.shape)
            * agent.target_noise,
            -agent.target_noise_clip,
            agent.target_noise_clip,
        )
        next_actions = jnp.clip(next_actions + action_noise, -1, 1)

        next_q1, next_q2 = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            observations=batch.observations,
            prev_actions=batch.actions,
            prev_rewards=batch.rewards,
            curr_actions=next_actions,
            rng=dropout_key2 if agent.use_dropout else None,
        )  # (B, T+1)
        next_q = jnp.minimum(next_q1, next_q2)

        target_q = (
            batch.rewards.squeeze(-1)
            + agent.discount * (1.0 - batch.terminals) * next_q
        )
        target_q = target_q[:, 1:] * batch.masks  # (B, T)

        # target_q = jax.lax.stop_gradient(target_q) # no need (no speed up / memory reducation)

        def critic_loss_fn(critic_params):
            q1, q2 = agent.critic.apply_fn(
                {"params": critic_params},
                observations=batch.observations[:, :-1],
                prev_actions=batch.actions[:, :-1],
                prev_rewards=batch.rewards[:, :-1],
                curr_actions=batch.actions[:, 1:],
                rng=dropout_key3 if agent.use_dropout else None,
            )  # (B, T)
            q1, q2 = (q1 * batch.masks, q2 * batch.masks)

            critic_loss = (
                (q1 - target_q) ** 2 + (q2 - target_q) ** 2
            ).sum() / num_valid

            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": q1.sum() / num_valid,
                "q2": q2.sum() / num_valid,
            }

        grads, critic_info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)

        clipped_grads, norm = F.clip_grad_norm(
            grads, max_norm=agent.max_norm, use_l2_norm=agent.use_l2_norm
        )
        critic_info.update({"critic_grad_norm": norm})

        grads = clipped_grads if agent.clip else grads
        critic = agent.critic.apply_gradients(grads=grads)
        target_critic = F.target_update(critic, agent.target_critic, agent.tau)

        agent = agent.replace(critic=critic, target_critic=target_critic, rng=rng)

        return agent, critic_info
