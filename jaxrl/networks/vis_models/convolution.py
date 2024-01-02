from typing import Tuple

import jax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl.networks.initializer import default_kernel_init, default_bias_init


class CNN(nn.Module):
    channels: Tuple[int]
    kernel_sizes: Tuple[int]
    strides: Tuple[int]
    embedding_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B=*, T=*, H, W, C)
        batch_dims = x.shape[:-3]

        if x.dtype == jnp.uint8:
            x = x / 255.0  # convert to float32 automatically
        else:
            x = x.astype(jnp.float32)

        for channel, kernel_size, stride in zip(
            self.channels, self.kernel_sizes, self.strides
        ):
            x = nn.Conv(
                features=channel,
                kernel_size=(kernel_size, kernel_size),
                strides=stride,
                padding=0,
            )(
                x
            )  # lecun_normal by default
            x = nn.relu(x)

        x = x.reshape(batch_dims + (-1,))  # (B=*, T=*, D)

        x = nn.Dense(
            features=self.embedding_size,
            kernel_init=default_kernel_init,
            bias_init=default_bias_init,
        )(x)

        return x
