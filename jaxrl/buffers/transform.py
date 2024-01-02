import jax
import jax.numpy as jnp

from .dataset import Batch

import functools


@functools.partial(jax.jit, static_argnames=("action_n",))
def preprocess_discrete_actions(actions: jnp.ndarray, action_n: int):
    return jax.nn.one_hot(actions, num_classes=action_n)


@jax.jit
def zero_prepend(batch: Batch):
    """
    batch's elements are jax arrays, thus jittable.
    """
    batch_size, action_dim = batch.actions.shape[0], batch.actions.shape[2:]
    if action_dim == ():  # discrete actions
        prepend_actions = -jnp.ones((batch_size, 1), dtype=jnp.int32)
    else:
        prepend_actions = jnp.zeros((batch_size, 1) + action_dim)

    return Batch(
        observations=jnp.concatenate(
            (batch.observations[:, [0]], batch.next_observations), axis=1
        ),  # (B, T+1, O)
        actions=jnp.concatenate(
            (prepend_actions, batch.actions), axis=1
        ),  # (B, T+1, A) or (B, T+1) int32
        rewards=jnp.concatenate(
            (jnp.zeros((batch_size, 1, 1)), batch.rewards), axis=1
        ),  # (B, T+1, 1)
        terminals=jnp.concatenate(
            (jnp.zeros((batch_size, 1)), batch.terminals), axis=1
        ),  # (B, T+1)
        next_observations=None,
        masks=batch.masks,  # (B, T)
    )
