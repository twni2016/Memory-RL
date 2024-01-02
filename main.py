import os
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
pid = str(os.getpid())
if "SLURM_JOB_ID" in os.environ:
    jobid = str(os.environ["SLURM_JOB_ID"])
else:
    jobid = pid

import math
import numpy as np
import jax
from absl import app, flags
from ml_collections import config_flags
import pickle
from utils import system, logger

from jaxrl.agents import get_learner
from jaxrl.buffers import (
    StepInfo,
    EpisodeInfo,
    MarkovReplayBuffer,
    SeqReplayBuffer,
)
from jaxrl.evaluation import evaluate
from jaxrl.make_env import make_env
import gym

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config_env",
    None,
    "File path to the environment configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_rl",
    None,
    "File path to the RL algorithm configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_seq",
    "configs/seq_models/mlp_default.py",
    "File path to the seq model configuration.",
    lock_config=False,
)

flags.mark_flags_as_required(["config_rl", "config_env"])

# shared encoder settings
flags.DEFINE_boolean("shared_encoder", False, "share encoder in actor-critic or not")
flags.DEFINE_boolean(
    "freeze_critic", False, "in shared encoder, freeze critic params in actor loss"
)
flags.DEFINE_boolean(
    "freeze_all", False, "in shared encoder, freeze 'all' params in actor loss"
)

# training settings
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("train_episodes", 1000, "Number of episodes during training.")
flags.DEFINE_float("updates_per_step", 1.0, "Gradient updates per step.")
flags.DEFINE_integer("start_training", 10, "Number of episodes to start training.")
flags.DEFINE_integer("reset", 0, "Reset the agent parameters every N episodes")

# logging settings
flags.DEFINE_boolean("debug", False, "debug mode")
flags.DEFINE_string("save_dir", "logs", "logging dir.")
flags.DEFINE_string("submit_time", None, "used in sbatch")


def main(argv):
    if FLAGS.seed < 0:
        seed = int(pid)  # to avoid conflict within a job which has same datetime
    else:
        seed = FLAGS.seed

    config_env = FLAGS.config_env
    config_rl = FLAGS.config_rl
    config_seq = FLAGS.config_seq

    config_env, env_name = config_env.create_fn(config_env)
    env = make_env(env_name, seed)
    eval_env = make_env(env_name, seed + 42)
    system.reproduce(seed)

    max_training_steps = int(FLAGS.train_episodes * env.max_episode_steps)
    start_training_steps = int(FLAGS.start_training * env.max_episode_steps)

    ## now only use env and time as directory name
    run_name = f"{config_env.env_type}/{config_env.env_name}/"
    config_seq, name = config_seq.name_fn(config_seq, env.max_episode_steps)
    # run_name += name
    config_rl, name = config_rl.name_fn(
        config_rl, env.max_episode_steps, max_training_steps
    )
    # run_name += name
    run_name += f"{system.now_str()}+{jobid}-{pid}"

    format_strs = ["csv"]  # now disable "tensorboard" as we cannot parse
    if FLAGS.debug:
        FLAGS.save_dir = "debug"
        format_strs.extend(["stdout", "log"])  # logger.log

    log_path = os.path.join(FLAGS.save_dir, run_name)
    logger.configure(dir=log_path, format_strs=format_strs)

    # write flags to a txt
    key_flags = FLAGS.get_key_flags_for_module(argv[0])
    with open(os.path.join(log_path, "flags.txt"), "w") as text_file:
        text_file.write("\n".join(f.serialize() for f in key_flags) + "\n")
    # write flags to pkl
    with open(os.path.join(log_path, "flags.pkl"), "wb") as f:
        pickle.dump(FLAGS.flag_values_dict(), f)

    if config_seq.is_markov:
        replay_buffer_def = MarkovReplayBuffer
    else:
        replay_buffer_def = SeqReplayBuffer
    replay_buffer = replay_buffer_def(
        max(
            int(config_rl.replay_buffer_size),
            int(config_rl.replay_buffer_num_episodes * env.max_episode_steps),
        ),
        env.observation_space,
        env.action_space,
        sampled_seq_len=config_seq.sampled_seq_len,
    )

    def create_agent(new_seed, old_agent=None):
        return get_learner(
            config_seq.is_markov,
            config_rl.algo,
            FLAGS.shared_encoder,
        ).create(
            new_seed,
            jax.device_put(env.observation_space.sample()),
            jax.device_put(env.action_space.sample())
            if isinstance(env.action_space, gym.spaces.Box)
            else env.action_space.n,
            **config_rl.to_dict(),
            count=getattr(old_agent, "count", 0),  # keep the flow
            config_seq=config_seq,
            freeze_critic=FLAGS.freeze_critic,
            freeze_all=FLAGS.freeze_all,
        )

    agent = create_agent(seed)

    random_sampling = True
    last_eval_time = 0
    last_eval_step = 0
    total_grad_steps = 0
    last_reset_time = 0
    t0 = time.time()

    def log_training(update_info):
        update_info = jax.device_get(update_info)
        # record in logger
        logger.record_step("env_steps", env.total_timesteps)
        for k, v in update_info.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()

    def log_eval(eval_stats):
        eval_stats = jax.device_get(eval_stats)
        # record in logger
        logger.record_step("env_steps", env.total_timesteps)
        for k, v in eval_stats.items():
            logger.record_tabular(k, v)
        system_fps = (env.total_timesteps - last_eval_step) / (time.time() - t0)
        logger.record_tabular("FPS", system_fps)
        logger.dump_tabular()

    while env.total_timesteps < max_training_steps:

        if random_sampling and env.total_timesteps >= start_training_steps:
            random_sampling = False

        # collect one episode
        observation, done = env.reset(), False
        if not config_seq.is_markov:
            episodeinfo = EpisodeInfo()
            if not random_sampling:
                stepinfo = agent.get_initial_info(jax.device_put(observation))

        while not done:
            if random_sampling:
                action = env.action_space.sample()
            else:
                # t0 = time.time()
                if config_seq.is_markov:
                    agent, action = agent.sample_actions(jax.device_put(observation))
                else:
                    (agent, action, (hidden_state, _),) = agent.sample_actions(
                        stepinfo.hidden_state,  # (H)
                        stepinfo.observation,  # (O)
                        stepinfo.prev_action,  # (A) or ()
                        stepinfo.prev_reward,  # (1)
                    )
                # logger.log("sample sto", time.time() - t0)

            next_observation, reward, done, info = env.step(action)
            terminal = config_env.terminal_fn(env, done, info)  # NOTE: designed by env

            if config_seq.is_markov:
                replay_buffer.add_sample(
                    observation, action, reward, terminal, next_observation
                )
            else:
                episodeinfo.append(
                    observation, action, reward, terminal, next_observation
                )

            observation = next_observation

            if not config_seq.is_markov and not random_sampling:
                stepinfo = jax.device_put(
                    StepInfo(
                        hidden_state=hidden_state,  # (H)
                        observation=observation,  # (O)
                        prev_action=action,  # (A) or ()
                        prev_reward=np.array([reward]),  # (1)
                    )
                )
        # print(env.episode_length, done, info, terminal, observation)

        if not config_seq.is_markov:
            episodeinfo.wrap()
            replay_buffer.add_episode(
                episodeinfo.observations,
                episodeinfo.actions,
                episodeinfo.rewards,
                episodeinfo.terminals,
                episodeinfo.next_observations,
            )

        if FLAGS.reset > 0:
            current_reset_time = total_grad_steps // (
                env.max_episode_steps * FLAGS.reset
            )
            if current_reset_time != last_reset_time:
                # NOTE: (Nikishin et al., 2022) RL with reset
                # here we reset every N grad steps, not env steps for scaling
                agent = create_agent(seed + current_reset_time, agent)
                last_reset_time = current_reset_time

        # (optional) training
        if env.total_timesteps >= start_training_steps:
            grad_updates = math.ceil(FLAGS.updates_per_step * info["episode"]["length"])
            total_grad_steps += grad_updates
            for _ in range(grad_updates):
                batch = jax.device_put(replay_buffer.random_batch(FLAGS.batch_size))

                agent, update_info = agent.update(batch)

            log_training(update_info)

        # (optional) evaluation
        current_eval_time = env.total_timesteps // (
            env.max_episode_steps * config_env.eval_interval
        )
        if current_eval_time != last_eval_time:
            eval_stats = evaluate(
                config_seq.is_markov,
                config_seq.is_attn,
                agent,
                eval_env,
                config_env.eval_episodes,
            )
            log_eval(eval_stats)

            t0 = time.time()
            last_eval_time = current_eval_time
            last_eval_step = env.total_timesteps


if __name__ == "__main__":
    app.run(main)
