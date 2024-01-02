import functools
from typing import Any, Callable, Optional, Tuple

from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from .base import SequenceModelBase

from .trajectory_flax_gpt2 import FlaxGPT2Module
from transformers import GPT2Config
from flax.core.frozen_dict import unfreeze


class SinePositionalEncoding(nn.Module):
    d_model: int  # Hidden dimensionality of the input.
    max_len: int = 5000  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [max_len, d] representing the positional encoding for max_len inputs
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)  # (1, max_len, d)

    def __call__(self, x, timestep=None):
        if timestep is None:
            # x: (B, T, d)
            # NOTE: assume sampling the whole sequence
            x = x + self.pe[:, : x.shape[1]]
        else:
            # x: (d), t: ()
            x = x + self.pe[0, timestep]
        return x


class LearnedPositionalEncoding(nn.Module):
    d_model: int  # Hidden dimensionality of the input.
    max_len: int = 5000  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [max_len, d] representing the positional encoding for max_len inputs
        self.pe = nn.Embed(num_embeddings=self.max_len, features=self.d_model)

    def __call__(self, x, timestep=None):
        if timestep is None:
            # x: (B, T, d)
            # NOTE: assume sampling the whole sequence
            x = x + self.pe(jnp.arange(x.shape[1]))[None]
        else:
            # x: (d), t: ()
            x = x + self.pe(timestep)
        return x


class TransformerGPT(SequenceModelBase):
    hidden_size: int
    max_seq_length: int
    n_layer: int = 2
    n_head: int = 4
    pdrop: float = 0.1
    position_encoding: str = "sine"

    def setup(self):
        config = GPT2Config(
            vocab_size=1,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.hidden_size,
            attn_pdrop=self.pdrop,
            resid_pdrop=self.pdrop,
            embd_pdrop=self.pdrop,
            # Maximum length sequence the transformer will see; default 1024 might be not long
            n_positions=self.max_seq_length + 2,
        )
        self.gpt_module = FlaxGPT2Module(config)
        PositionalEncoding = (
            SinePositionalEncoding
            if self.position_encoding == "sine"
            else LearnedPositionalEncoding
        )
        self.positional_encoding = PositionalEncoding(
            self.hidden_size, self.max_seq_length + 2
        )

    def __call__(self, embedded_inputs, rng=None):
        if self.has_variable("params", "gpt_module"):
            gpt_params = {"params": self.variables["params"]["gpt_module"]}

            if rng is None:  # evaluation
                deterministic, dropout_rng = True, None
            else:  # training or exploration
                # https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.InvalidRngError
                # Module.apply() has rng for dropout
                deterministic, dropout_rng = False, {"dropout": rng}

            out = self.gpt_module.apply(
                gpt_params,
                embedded_inputs,
                mutable=False,
                output_attentions=True,
                deterministic=deterministic,
                rngs=dropout_rng,
            )

            return out
        else:
            # init
            return self.gpt_module(
                embedded_inputs, output_attentions=True, deterministic=False
            )

    def forward(self, embedded_inputs, initial_states, rng):
        """
        Inputs:
            embedded_inputs: (B, T, D)
            initial_states: Not used
            rng: for dropout
        Outputs:
            carry on: None, Not used
            outputs: (B, T, D)
            attn_weights: Tuple of jnp.ndarray (one for each layer) of shape
            (batch_size, num_heads, sequence_length, sequence_length).
        """
        embedded_inputs = self.positional_encoding(embedded_inputs)
        out = self.__call__(embedded_inputs, rng=rng)
        return (None, out["last_hidden_state"]), out["attentions"]

    def forward_per_step(self, raw_embedded_inputs, initial_states, rng):
        """
        Inputs:
            raw_embedded_inputs: (D)
            initial_states (carry on): (B, L, D)
            rng: for dropout (if rng is None, then determinisitic)
        Outputs:
            carry on: (B, L, D)
            current states: (D)
            attention: to be processed
        """
        batch_size = (
            1 if len(raw_embedded_inputs.shape) == 1 else raw_embedded_inputs.shape[0]
        )
        embedded_inputs = raw_embedded_inputs.reshape(
            batch_size, raw_embedded_inputs.shape[-1]
        )  # (B, D)

        recent_embedded_inputs, timestep = initial_states  # (B, L, D), int
        recent_embedded_inputs, clipped_timestep = jax.lax.cond(
            timestep > self.max_seq_length - 1,
            lambda x, y: (jnp.roll(x, shift=-1, axis=-2), self.max_seq_length - 1),
            lambda x, y: (x, y),
            recent_embedded_inputs,  # x
            timestep,  # y
        )
        recent_embedded_inputs = recent_embedded_inputs.at[:, clipped_timestep].set(
            embedded_inputs
        )

        current_embedded_inputs = self.positional_encoding(recent_embedded_inputs)

        # NOTE: when t < L-1, the inputs after clipped_timestep are dummy, but they won't change
        # the prediction at current time, as we have causal mask
        # and we will extract the hidden state of current time below
        out = self.__call__(current_embedded_inputs, rng=rng)
        current_outputs = out["last_hidden_state"][:, clipped_timestep].reshape(
            raw_embedded_inputs.shape
        )  # (D)
        attns = out["attentions"]  # TODO remove future attention

        timestep += 1
        return ((recent_embedded_inputs, timestep), current_outputs), attns

    def initialize_carry(self, batch_dims):
        """
        Only effective in evaluation
        Carry recent inputs of max_seq_length
        """
        batch_size = batch_dims[0] if len(batch_dims) == 1 else 1
        recent_embedded_inputs = jnp.zeros(
            (batch_size, self.max_seq_length, self.hidden_size)
        )

        timestep = jnp.array(0, dtype=jnp.int32)
        return recent_embedded_inputs, timestep
