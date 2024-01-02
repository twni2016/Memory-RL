"""Implementations of algorithms for continuous control."""

import functools
from typing import Callable, Sequence, Tuple, Union, Optional
from ml_collections import ConfigDict
from flax.core.frozen_dict import FrozenDict, freeze
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.traverse_util import flatten_dict, unflatten_dict
from flax import struct
from flax.training.train_state import TrainState
import jaxrl.networks.functionals as F

from jaxrl.buffers import Batch
from jaxrl.agents.sac.temperature import Temperature
from jaxrl.networks import memory_actor
from jaxrl.networks.memory_actor_critic import (
    MemoryNormalTanhActorDoubleCritic as MemoryActorCritic,
)
from jaxrl.networks.types import InfoDict
from jaxrl.buffers.transform import zero_prepend

from .markov_sac_learner import MarkovSACLearner


class SharedMemorySACLearner(MarkovSACLearner):
    actorcritic: TrainState
    target_actorcritic: TrainState
    freeze_critic: bool = struct.field(pytree_node=False)
    freeze_all: bool = struct.field(pytree_node=False)

    # some functions
    # during training
    forward_encoder: Callable = struct.field(pytree_node=False)
    forward_actor: Callable = struct.field(pytree_node=False)
    forward_critic: Callable = struct.field(pytree_node=False)

    # during evaluation
    behavior_actor_get_initial_info: Callable = struct.field(pytree_node=False)
    behavior_actor_apply: Callable = struct.field(pytree_node=False)

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
        temp_lr: float = 3e-4,
        config_actor: Optional[dict] = None,
        config_critic: Optional[dict] = None,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        init_temperature: float = 1.0,
        update_temperature: bool = True,
        config_seq: Optional[ConfigDict] = None,
        freeze_critic: bool = False,
        freeze_all: bool = False,
        **kwargs
    ):
        observations = observations[None, None, ...]
        prev_actions = actions[None, None, ...]
        curr_actions = actions[None, None, ...]
        action_dim = actions.shape[-1]
        rewards = jnp.zeros((1, 1, 1))  # (B=1, T=1, 1)

        if target_entropy is None:
            target_entropy = -action_dim

        clip = config_seq.clip
        max_norm = config_seq.max_norm
        use_l2_norm = config_seq.use_l2_norm

        seq_model_dict = FrozenDict(config_seq.model.to_dict())

        rng = jax.random.PRNGKey(seed)
        use_dropout = config_seq.use_dropout
        if use_dropout:
            rng, actorcritic_key, dropout_key, temp_key, coef_key = jax.random.split(
                rng, 5
            )
            init_rngs = {"params": actorcritic_key, "dropout": dropout_key}
        else:
            rng, actorcritic_key, temp_key, coef_key = jax.random.split(rng, 4)
            init_rngs = actorcritic_key

        actorcritic_def = MemoryActorCritic(
            config_actor=FrozenDict(config_actor),
            config_critic=FrozenDict(config_critic),
            action_dim=action_dim,
            observ_dim=observations.shape[-1],
            **seq_model_dict,
        )
        actorcritic_params = actorcritic_def.init(
            init_rngs,
            observations,
            prev_actions,
            rewards,
            curr_actions,
            method=actorcritic_def.init_params,
        )["params"]

        # assign a mask of learning rates
        # https://github.com/nyx-ai/stylegan2-flax-tpu/blob/main/optimizers.py
        params_mask = flatten_dict(actorcritic_params)
        for key in params_mask.keys():
            if "actor" in key:
                params_mask[key] = "actor"  # actor_lr
            elif "critic" in key:
                params_mask[key] = "critic"  # critic_lr
            else:
                params_mask[key] = "encoder"  # config_seq.lr
        params_mask = freeze(unflatten_dict(params_mask))

        actorcritic = TrainState.create(
            apply_fn=actorcritic_def.apply,
            params=actorcritic_params,
            tx=optax.multi_transform(
                {
                    "actor": optax.adam(actor_lr),
                    "critic": optax.adam(critic_lr),
                    "encoder": optax.adam(config_seq.lr),
                },
                params_mask,
            ),
        )
        F.print_model(actorcritic_def, actorcritic_params)

        target_actorcritic = TrainState.create(
            apply_fn=actorcritic_def.apply,
            params=actorcritic_params,  # use same params
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
            actor=None,
            critic=None,
            target_critic=None,
            actorcritic=actorcritic,
            target_actorcritic=target_actorcritic,
            temp=temp,
            tau=tau,
            discount=discount,
            # sac related
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
            update_temperature=update_temperature,
            # memory related
            forward_encoder=actorcritic_def.train,
            forward_actor=actorcritic_def.forward_actor,
            forward_critic=actorcritic_def.forward_critic,
            behavior_actor_get_initial_info=actorcritic_def.get_initial_info,
            behavior_actor_apply=actorcritic_def.apply,
            # bools
            use_dropout=use_dropout,
            clip=clip,
            max_norm=max_norm,
            use_l2_norm=use_l2_norm,
            freeze_critic=True if freeze_all else freeze_critic,
            freeze_all=freeze_all,
        )

    def get_initial_info(self, observation: jnp.array):
        return memory_actor.get_initial_info(
            self.behavior_actor_apply,
            self.actorcritic.params,
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
            self.actorcritic.params,
            hidden_states,
            observations,
            prev_actions,
            prev_rewards,
            mode,
            distribution=True,
        )

        actions = jax.device_get(actions)
        actions = np.clip(actions, -1, 1)

        return self.replace(rng=rng), actions, (new_hidden_states, attn_weights)

    @jax.jit
    def update(self, batch: Batch) -> InfoDict:
        batch = zero_prepend(batch)  # preprocess to (B, T+1, dim)

        new_agent = self

        new_agent, actorcritic_info = new_agent.update_actorcritic(batch)

        new_agent, temp_info = new_agent.update_temp(
            new_agent, actorcritic_info["entropy"]
        )

        return new_agent, {**actorcritic_info, **temp_info}

    def update_actorcritic(self, batch: Batch):

        agent = self
        num_valid = jnp.clip(batch.masks.sum(), a_min=1.0)

        if agent.use_dropout:
            (
                rng,
                dropout_key1,
                dropout_key2,
                dropout_key3,
                action_key1,
                action_key2,
            ) = jax.random.split(agent.rng, 6)
        else:
            rng, action_key1, action_key2 = jax.random.split(agent.rng, 3)

        actor_encoding = agent.actorcritic.apply_fn(
            {"params": agent.actorcritic.params},
            observations=batch.observations,
            prev_actions=batch.actions,
            prev_rewards=batch.rewards,
            rng=dropout_key1 if agent.use_dropout else None,
            method=agent.forward_encoder,
        )  # (B, T+1, H); will be used by the actor in target
        next_dist = agent.actorcritic.apply_fn(
            {"params": agent.actorcritic.params},
            outputs=actor_encoding,
            method=agent.forward_actor,
        )  # (B, T+1, A)
        next_actions = next_dist.sample(seed=action_key1)  # (B, T+1, A)

        targetcritic_encoding = agent.target_actorcritic.apply_fn(
            {"params": agent.target_actorcritic.params},
            observations=batch.observations,
            prev_actions=batch.actions,
            prev_rewards=batch.rewards,
            rng=dropout_key2 if agent.use_dropout else None,
            method=agent.forward_encoder,
        )  # (B, T+1, H); will be used by target critic
        next_q1, next_q2 = agent.target_actorcritic.apply_fn(
            {"params": agent.target_actorcritic.params},
            outputs=targetcritic_encoding,
            curr_actions=next_actions,
            method=agent.forward_critic,
        )  # (B, T+1)
        next_q = jnp.minimum(next_q1, next_q2)

        target_q = (
            batch.rewards.squeeze(-1)
            + agent.discount * (1.0 - batch.terminals) * next_q
        )
        if agent.backup_entropy:  # True: SAC; False: SAC-Lite
            next_log_probs = next_dist.log_prob(next_actions)
            target_q += (
                agent.discount
                * (1.0 - batch.terminals)
                * agent.temp.apply_fn({"params": agent.temp.params})
                * (-next_log_probs)
            )

        target_q = target_q[:, 1:] * batch.masks  # (B, T)

        target_q = jax.lax.stop_gradient(target_q)  # unsure this is necessary?

        def actorcritic_loss_fn(actorcritic_params):
            all_encoding = agent.actorcritic.apply_fn(
                {"params": actorcritic_params},
                observations=batch.observations,
                prev_actions=batch.actions,
                prev_rewards=batch.rewards,
                rng=dropout_key3 if agent.use_dropout else None,
                method=agent.forward_encoder,
            )  # (B, T, H+1); will be used by both actor and critic
            encoding = all_encoding[:, :-1]  # (B, T, H)

            q1, q2 = agent.actorcritic.apply_fn(
                {"params": actorcritic_params},
                outputs=encoding,
                curr_actions=batch.actions[:, 1:],
                method=agent.forward_critic,
            )  # (B, T)
            q1, q2 = (q1 * batch.masks, q2 * batch.masks)

            critic_loss = (
                (q1 - target_q) ** 2 + (q2 - target_q) ** 2
            ).sum() / num_valid

            dist = agent.actorcritic.apply_fn(
                {"params": actorcritic_params},
                # NOTE: freezing all follows SAC+AE and DrQ
                outputs=jax.lax.stop_gradient(encoding)
                if agent.freeze_all
                else encoding,
                method=agent.forward_actor,
            )  # (B, T, A)
            actions = dist.sample(seed=action_key2)  # (B, T, A)
            log_probs = dist.log_prob(actions)  # (B, T)

            #### Start of freezing critic
            new_q1, new_q2 = agent.actorcritic.apply_fn(
                {  # NOTE: agent.actorcritic.params does not backprop
                    "params": agent.actorcritic.params
                    if agent.freeze_critic
                    else actorcritic_params
                },
                outputs=jax.lax.stop_gradient(encoding)
                if agent.freeze_critic
                else encoding,
                curr_actions=actions,
                method=agent.forward_critic,
            )  # (B, T)
            #### End of freezing critic

            new_q = jnp.minimum(new_q1, new_q2)

            soft_q = new_q - log_probs * agent.temp.apply_fn(
                {"params": agent.temp.params}
            )

            actor_loss = (-soft_q * batch.masks).sum() / num_valid

            losses = actor_loss + critic_loss

            def check_collapse(data, mask):
                # thanks to chatgpt
                mask = jnp.repeat(jnp.expand_dims(mask, -1), data.shape[-1], axis=-1)
                masked_data = jnp.where(mask, data, jnp.nan)
                return jnp.nanstd(masked_data, axis=(0, 1)).mean()

            info = {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "q1": q1.sum() / num_valid,
                "q2": q2.sum() / num_valid,
                "entropy": -(log_probs * batch.masks).sum() / num_valid,
                "z_std": check_collapse(encoding, batch.masks),
            }

            return losses, info

        grads, actorcritic_info = jax.grad(actorcritic_loss_fn, has_aux=True)(
            agent.actorcritic.params
        )

        clipped_grads, norm = F.clip_grad_norm(
            grads, max_norm=agent.max_norm, use_l2_norm=agent.use_l2_norm
        )
        actorcritic_info.update({"actorcritic_grad_norm": norm})

        grads = clipped_grads if agent.clip else grads
        actorcritic = agent.actorcritic.apply_gradients(grads=grads)
        target_actorcritic = F.target_update(
            actorcritic, agent.target_actorcritic, agent.tau
        )
        agent = agent.replace(
            actorcritic=actorcritic, target_actorcritic=target_actorcritic, rng=rng
        )

        return agent, actorcritic_info
