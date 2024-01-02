from typing import Any, Callable, Optional, Union, Sequence, Tuple
from flax import linen as nn

import jax.numpy as jnp

Hidden_state_shape = Union[Sequence[jnp.ndarray], jnp.ndarray]
Output_shape = Union[Sequence[jnp.ndarray], jnp.ndarray]
Attn_shape = Optional[jnp.ndarray]


class SequenceModelBase(nn.Module):
    def forward(
        self,
        embedded_inputs: jnp.ndarray,  # (B, T, X)
        initial_states: Hidden_state_shape,  # (B, D)
        **kwargs,
    ) -> Tuple[Tuple[Hidden_state_shape, Output_shape], Attn_shape]:
        """
        training on a batch of sequences with initial hidden states
        """
        raise NotImplementedError

    def forward_per_step(
        self,
        embedded_inputs: jnp.ndarray,  # (B=*, X)
        initial_states: Hidden_state_shape,  # (B=*, D)
        **kwargs,
    ) -> Tuple[Tuple[Hidden_state_shape, Output_shape], Attn_shape]:
        """
        evaluate on a batch of single-step inputs with initial hidden states
        """
        raise NotImplementedError

    def initialize_carry(self, batch_dims: Tuple[int]) -> Hidden_state_shape:
        """
        return initial hidden state (B=*, D), where batch_dims can be (), (B,)
        """
        raise NotImplementedError
