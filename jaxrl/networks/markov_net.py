from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl.networks.initializer import default_kernel_init, default_bias_init


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    add_layernorm: bool = False
    layernorm_epsilon: float = 1e-5  # keep it same as huggingface and pytorch
    layernorm_use_bias_scale: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(
                size, kernel_init=default_kernel_init, bias_init=default_bias_init
            )(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.add_layernorm:
                    x = nn.LayerNorm(
                        epsilon=self.layernorm_epsilon,
                        use_bias=self.layernorm_use_bias_scale,
                        use_scale=self.layernorm_use_bias_scale,
                    )(x)
                x = self.activations(x)
        return x
