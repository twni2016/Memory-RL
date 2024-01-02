import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from functools import partial
from utils import logger


@jax.jit
def parameter_size(params):
    return jnp.sum(
        jnp.stack(
            jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(
                    lambda x: jnp.product(jnp.array(x.shape)), params
                )
            )
        )
    )


def print_model(model_def, params):
    logger.log(model_def)
    logger.log(jax.tree_util.tree_map(lambda x: x.shape, params))
    logger.log("parameter size:", parameter_size(params), "\n")


def target_update(
    new_model: TrainState, target_model: TrainState, tau: float
) -> TrainState:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), new_model.params, target_model.params
    )

    return target_model.replace(params=new_target_params)


optim_null = optax.GradientTransformation(lambda _: None, lambda _: None)


@partial(jax.jit, static_argnums=(1, 2))
def clip_grad_norm(grads, max_norm: float = 1.0, use_l2_norm: bool = True):
    """
    optax: https://github.com/deepmind/optax/blob/master/optax/_src/clipping.py#L91#L125
        they did not support inf norm
    torch: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
        follow their code on inf norm
    flax: https://github.com/google/flax/issues/859#issuecomment-1063790456
        also follow their code
    NOTE: if jnp.isnan(norm), you may omit the gradient update,
            please do this in the training script, not here
    """
    if use_l2_norm:  # l2 norm. jnp.linalg.norm will flatten the 2D array to vector
        norm = jnp.linalg.norm(
            jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.linalg.norm, grads))
        )  # it may be inf, as it is proportional to sqrt(param_size)
    else:  # inf norm.
        norm = jnp.max(
            jnp.stack(
                jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(lambda x: jnp.max(jnp.abs(x)), grads)
                )
            )
        )  # it is bounded

    clip = lambda x: jnp.where(norm < max_norm, x, x * max_norm / (norm + 1e-6))

    return jax.tree_util.tree_map(clip, grads), norm
