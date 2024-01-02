from typing import Dict

import time
import gym
import numpy as np
import jax
import jax.numpy as jnp

from jaxrl.buffers import StepInfo
from utils import logger
from utils.plot_attn import plot_attnmaps

discounts = [0.99, 0.999]


def evaluate(
    is_markov: bool,
    is_attn: bool,
    agent,
    env: gym.Env,
    num_episodes: int,
) -> Dict[str, float]:
    stats = {"length": [], "return": []}
    for discount in discounts:
        stats[f"return-{discount}"] = []
    successes = None

    if env.store_full_metrics:
        metrics = []

    for _ in range(num_episodes):
        observation, done = env.reset(), False

        if not is_markov:
            stepinfo = agent.get_initial_info(jax.device_put(observation))

        while not done:
            # t0 = time.time()
            if is_markov:
                _, action = agent.sample_actions(
                    jax.device_put(observation),
                    mode=True,  # NOTE: deterministic
                )
            else:
                _, action, (hidden_state, attn_weight) = agent.sample_actions(
                    stepinfo.hidden_state,  # (H)
                    stepinfo.observation,  # (O)
                    stepinfo.prev_action,  # (A) or ()
                    stepinfo.prev_reward,  # (1)
                    mode=True,  # NOTE: deterministic
                )
            # logger.log("sample det", time.time() - t0)

            # t0 = time.time()
            observation, reward, done, info = env.step(action)
            # logger.log("transit", time.time()-t0)

            if not is_markov:
                stepinfo = jax.device_put(
                    StepInfo(
                        hidden_state=hidden_state,  # (H)
                        observation=observation,  # (O)
                        prev_action=action,  # (A) or ()
                        prev_reward=np.array([reward]),  # (1)
                    )
                )

        for k in ["length", "return"]:
            stats[k].append(info["episode"][k])
        for discount in discounts:
            discounted_return = np.sum(
                info["episode"]["rewards"]
                * (discount ** np.arange(len(info["episode"]["rewards"])))
            )
            stats[f"return-{discount}"].append(discounted_return)

        if "is_success" in info:
            if successes is None:
                successes = 0.0
            successes += info["is_success"]

        if env.store_full_metrics:
            metrics.append(info["episode"])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats["success"] = successes / num_episodes

    if env.store_full_metrics:
        return stats, metrics

    return stats
