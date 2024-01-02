import flax.linen as nn
import jax.numpy as jnp


# used in Ilya's repo for weight (they use scale=sqrt(2))
orthogonal = nn.initializers.orthogonal
# used in torch for weight and bias
he_uniform = nn.initializers.he_uniform
# used in flax/tensorflow for weight
lecun_normal = nn.initializers.lecun_normal
# used in flax/tensorflow for bias
zeros = nn.initializers.zeros


default_kernel_init = orthogonal()
default_bias_init = zeros
