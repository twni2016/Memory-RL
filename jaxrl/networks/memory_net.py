from typing import Any, Callable, Optional, Sequence, Tuple, Union
from flax.core.frozen_dict import FrozenDict
from jaxrl.networks.types import PRNGKey

import flax.linen as nn
import jax.numpy as jnp

from jaxrl.networks.markov_net import MLP
from jaxrl.networks.seq_models import get_seq_model
from jaxrl.networks.vis_models.convolution import CNN


def select_embedder(name):
    if name == "mlp":
        return MLP
    elif name == "cnn":
        return CNN
    else:
        raise ValueError


class MemoryEncoder(nn.Module):
    observ_embedder: FrozenDict
    action_embedder: FrozenDict
    reward_embedder: FrozenDict
    seq_model_config: FrozenDict

    def setup(self):
        if self.observ_embedder is not None:
            observ_embedder, name = self.observ_embedder.pop("name")
            self.obs_embedding_layer = select_embedder(name)(**observ_embedder)

        if self.action_embedder is not None:
            action_embedder, name = self.action_embedder.pop("name")
            self.act_embedding_layer = select_embedder(name)(**action_embedder)

        if self.reward_embedder is not None:
            reward_embedder, name = self.reward_embedder.pop("name")
            self.rew_embedding_layer = select_embedder(name)(**reward_embedder)

        seq_model_config, seq_model_name = self.seq_model_config.pop("name")
        self.seq_model = get_seq_model(seq_model_name)(**seq_model_config)

    def forward_encoder(
        self,
        is_training: bool,
        initial_hidden_states: Union[jnp.ndarray, Tuple[jnp.ndarray]],  # (B=*, H)
        observations: jnp.ndarray,  # (B=*, T=*, O)
        prev_actions: jnp.ndarray,  # (B=*, T=*, None or A)
        prev_rewards: jnp.ndarray,  # (B=*, T=*, 1)
        rng: PRNGKey,
    ) -> jnp.ndarray:
        """
        During training: 3D array.   B = batch_size, T = sampled_seq_len
        During evaluation: 1D array. B = (), T = ()
        """
        rest_dims = prev_rewards.shape[:-1]  # () or (B, T)

        if self.observ_embedder is not None:
            obs_embed = self.obs_embedding_layer(observations)
        else:
            obs_embed = jnp.zeros(rest_dims + (0,))  # a placeholder

        if self.action_embedder is not None:
            prev_actions = self.preprocess_actions(prev_actions)  # for discrete action
            act_embed = self.act_embedding_layer(prev_actions)
        else:
            act_embed = jnp.zeros(rest_dims + (0,))  # a placeholder

        if self.reward_embedder is not None:
            rew_embed = self.rew_embedding_layer(prev_rewards)
        else:
            rew_embed = jnp.zeros(rest_dims + (0,))  # a placeholder

        input_embeds = jnp.concatenate(
            [obs_embed, act_embed, rew_embed], axis=-1
        )  # (B=*, T=*, D)

        if is_training:
            seq_forward_fn = self.seq_model.forward
        else:
            seq_forward_fn = self.seq_model.forward_per_step

        (hidden_states, outputs), attn_weights = seq_forward_fn(
            input_embeds, initial_hidden_states, rng=rng
        )  # (B=*, T=*, H)

        return (hidden_states, outputs), attn_weights

    def train(
        self,
        observations: jnp.ndarray,  # (B, T, O) or (B, T, H, W, C)
        prev_actions: jnp.ndarray,  # (B, T, none or A)
        prev_rewards: jnp.ndarray,  # (B, T, 1)
        rng: PRNGKey,
    ) -> jnp.ndarray:
        batch_dims = prev_rewards.shape[:-2]
        initial_hidden_states = self.seq_model.initialize_carry(batch_dims)

        (_, outputs), _ = self.forward_encoder(
            is_training=True,
            initial_hidden_states=initial_hidden_states,
            observations=observations,
            prev_actions=prev_actions,
            prev_rewards=prev_rewards,
            rng=rng,
        )

        return outputs

    def preprocess_actions(self, prev_actions: jnp.ndarray):
        return prev_actions
