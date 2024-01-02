import functools
from typing import Any, Callable, Optional

from flax import linen as nn
import jax
from jax import numpy as jnp
from jaxrl.networks.initializer import default_kernel_init, default_bias_init
from .base import SequenceModelBase


class VanillaLSTM(SequenceModelBase):
    """A simple uni-directional LSTM."""

    hidden_size: int

    def setup(self):
        self.cell = nn.OptimizedLSTMCell(
            kernel_init=default_kernel_init, bias_init=default_bias_init
        )
        # recurrent_kernel_init=orthogonal is default.

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    def __call__(self, carry, x):
        return self.cell(carry, x)

    def forward(self, embedded_inputs, initial_states, **kwargs):
        hidden_states, outputs = self.__call__(initial_states, embedded_inputs)

        # ((B, T, D), (B, T, D)), (B, T, D)
        return (hidden_states, outputs), None

    def forward_per_step(self, embedded_inputs, initial_states, **kwargs):
        hidden_states, outputs = self.cell(initial_states, embedded_inputs)

        # ((D), (D)), (D)
        return (hidden_states, outputs), None

    def initialize_carry(self, batch_dims):
        # Use fixed random key since default state init fn is just zeros.
        return self.cell.initialize_carry(
            jax.random.PRNGKey(0), batch_dims, self.hidden_size
        )
