import os
import time

import math
import numpy as np
import torch
from torch.nn import functional as F

from .models import AGENT_CLASSES, AGENT_ARCHS
from torchkit.networks import ImageEncoder

# Markov policy
from buffers.simple_replay_buffer import SimpleReplayBuffer

# RNN policy on vector-based task
from buffers.seq_replay_buffer_vanilla import SeqReplayBuffer

# RNN policy on image/vector-based task
from buffers.seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from utils import logger


class Learner:
    def __init__(self, env, eval_env, FLAGS, config_rl, config_seq, config_env):
        self.train_env = env
        self.eval_env = eval_env
        self.FLAGS = FLAGS
        self.config_rl = config_rl
        self.config_seq = config_seq
        self.config_env = config_env

        self.init_env()

        self.init_agent()

        self.init_train()

    def init_env(
        self,
    ):
        # get action / observation dimensions
        assert len(self.train_env.observation_space.shape) == 1  # flatten
        if self.train_env.action_space.__class__.__name__ == "Box":
            # continuous action space
            self.act_dim = self.train_env.action_space.shape[0]
            self.act_continuous = True
        else:
            assert self.train_env.action_space.__class__.__name__ == "Discrete"
            self.act_dim = self.train_env.action_space.n
            self.act_continuous = False
        self.obs_dim = self.train_env.observation_space.shape[0]
        logger.log("obs_dim", self.obs_dim, "act_dim", self.act_dim)

    def init_agent(
        self,
    ):
        # initialize agent
        if self.config_seq.is_markov:
            agent_class = AGENT_CLASSES["Policy_MLP"]
        else:
            if self.config_rl.algo == "dqn":
                agent_class = AGENT_CLASSES["Policy_DQN_RNN"]
            elif self.FLAGS.shared_encoder:
                agent_class = AGENT_CLASSES["Policy_Shared_RNN"]
            else:
                agent_class = AGENT_CLASSES["Policy_Separate_RNN"]

        self.agent_arch = agent_class.ARCH
        logger.log(agent_class, self.agent_arch)

        if self.config_seq.model.observ_embedder.name == "cnn":
            image_encoder_fn = lambda: ImageEncoder(
                image_shape=self.train_env.image_space.shape,
                normalize_pixel=(self.train_env.observation_space.dtype == np.uint8),
                **self.config_seq.model.observ_embedder.to_dict(),
            )
        else:
            image_encoder_fn = lambda: None

        self.agent = agent_class(
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            config_seq=self.config_seq,
            config_rl=self.config_rl,
            image_encoder_fn=image_encoder_fn,
            freeze_critic=self.FLAGS.freeze_critic,
        ).to(ptu.device)
        logger.log(self.agent)

    def init_train(
        self,
    ):

        if self.agent_arch == AGENT_ARCHS.Markov:
            self.policy_storage = SimpleReplayBuffer(
                max_replay_buffer_size=int(self.config_rl.replay_buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                max_trajectory_len=self.train_env.max_episode_steps,
                add_timeout=False,  # no timeout storage
            )

        else:  # memory, memory-markov
            if self.config_seq.is_markov:
                buffer_class = SeqReplayBuffer
            else:
                buffer_class = RAMEfficient_SeqReplayBuffer
            logger.log(buffer_class)

            self.policy_storage = buffer_class(
                max_replay_buffer_size=max(
                    int(self.config_rl.replay_buffer_size),
                    int(
                        self.config_rl.replay_buffer_num_episodes
                        * self.train_env.max_episode_steps
                    ),
                ),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                sampled_seq_len=self.config_seq.sampled_seq_len,
                observation_type=self.train_env.observation_space.dtype,
            )

        total_rollouts = self.FLAGS.start_training + self.FLAGS.train_episodes
        self.n_env_steps_total = self.train_env.max_episode_steps * total_rollouts
        logger.log(
            "*** total rollouts",
            total_rollouts,
            "total env steps",
            self.n_env_steps_total,
        )

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_env_steps_total_last = 0
        self._n_rl_update_steps_total = 0
        self._n_rollouts_total = 0
        self._successes_in_buffer = 0

        self._start_time = time.time()
        self._start_time_last = time.time()

    def train(self):
        """
        training loop
        """

        self._start_training()

        if self.FLAGS.start_training > 0:
            logger.log("Collecting initial pool of data..")
            while (
                self._n_env_steps_total
                < self.FLAGS.start_training * self.train_env.max_episode_steps
            ):
                self.collect_rollouts(
                    num_rollouts=1,
                    random_actions=True,
                )
            logger.log(
                "Done! env steps",
                self._n_env_steps_total,
                "rollouts",
                self._n_rollouts_total,
            )

            train_stats = self.update(
                int(self._n_env_steps_total * self.FLAGS.updates_per_step)
            )
            self.log_train_stats(train_stats)

        last_eval_num_iters = 0
        while self._n_env_steps_total < self.n_env_steps_total:
            env_steps = self.collect_rollouts(num_rollouts=1)
            logger.log("env steps", self._n_env_steps_total)

            train_stats = self.update(
                int(math.ceil(self.FLAGS.updates_per_step * env_steps))
            )  # NOTE: ceil to make sure at least 1 step
            self.log_train_stats(train_stats)

            # evaluate and log
            current_num_iters = self._n_env_steps_total // (
                self.train_env.max_episode_steps
            )
            if (
                current_num_iters != last_eval_num_iters
                and current_num_iters % self.config_env.eval_interval == 0
            ):
                last_eval_num_iters = current_num_iters
                self.log()

    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """

        before_env_steps = self._n_env_steps_total
        for idx in range(num_rollouts):
            steps = 0

            obs = ptu.from_numpy(self.train_env.reset())  # reset
            obs = obs.reshape(1, obs.shape[-1])
            done_rollout = False

            if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                # temporary storage
                obs_list, act_list, rew_list, next_obs_list, term_list = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )

            if self.agent_arch == AGENT_ARCHS.Memory and not random_actions:
                # get hidden state at timestep=0, None for markov
                # NOTE: assume initial reward = 0.0 (no need to clip)
                action, reward, internal_state = self.agent.get_initial_info(
                    self.config_seq.sampled_seq_len
                )

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor(
                        [self.train_env.action_space.sample()]
                    )  # (1, A) for continuous action, (1) for discrete action
                    if not self.act_continuous:
                        action = F.one_hot(
                            action.long(), num_classes=self.act_dim
                        ).float()  # (1, A)
                else:
                    # policy takes hidden state as input for memory-based actor,
                    # while takes obs for markov actor
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        action, internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward,
                            obs=obs,
                            deterministic=False,
                        )
                    else:
                        action = self.agent.act(obs, deterministic=False)

                # observe reward and next obs (B=1, dim)
                next_obs, reward, done, info = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )

                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                # update statistics
                steps += 1

                # NOTE: designed by env
                term = self.config_env.terminal_fn(self.train_env, done_rollout, info)

                # add data to policy buffer
                if self.agent_arch == AGENT_ARCHS.Markov:
                    self.policy_storage.add_sample(
                        observation=ptu.get_numpy(obs.squeeze(dim=0)),
                        action=ptu.get_numpy(
                            action.squeeze(dim=0)
                            if self.act_continuous
                            else torch.argmax(
                                action.squeeze(dim=0), dim=-1, keepdims=True
                            )  # (1,)
                        ),
                        reward=ptu.get_numpy(reward.squeeze(dim=0)),
                        terminal=np.array([term], dtype=float),
                        next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                    )
                else:  # append tensors to temporary storage
                    obs_list.append(obs)  # (1, dim)
                    act_list.append(action)  # (1, dim)
                    rew_list.append(reward)  # (1, dim)
                    term_list.append(term)  # bool
                    next_obs_list.append(next_obs)  # (1, dim)

                # set: obs <- next_obs
                obs = next_obs.clone()

            if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                # add collected sequence to buffer
                act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
                if not self.act_continuous:
                    act_buffer = torch.argmax(
                        act_buffer, dim=-1, keepdims=True
                    )  # (L, 1)

                self.policy_storage.add_episode(
                    observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                    actions=ptu.get_numpy(act_buffer),  # (L, dim)
                    rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                    terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                    next_observations=ptu.get_numpy(
                        torch.cat(next_obs_list, dim=0)
                    ),  # (L, dim)
                )
                print(
                    f"steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}"
                )
            self._n_env_steps_total += steps
            self._n_rollouts_total += 1
        return self._n_env_steps_total - before_env_steps

    def sample_rl_batch(self, batch_size):
        if self.agent_arch == AGENT_ARCHS.Markov:
            batch = self.policy_storage.random_batch(batch_size)
        else:  # rnn: all items are (sampled_seq_len, B, dim)
            batch = self.policy_storage.random_episodes(batch_size)
        return ptu.np_to_pytorch_batch(batch)

    def update(self, num_updates):
        rl_losses_agg = {}
        for update in range(num_updates):
            # sample random RL batch: in transitions
            batch = self.sample_rl_batch(self.FLAGS.batch_size)

            # RL update
            rl_losses = self.agent.update(batch)

            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
        # statistics
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += num_updates

        return rl_losses_agg

    @torch.no_grad()
    def evaluate(self, deterministic=True):
        self.agent.eval()  # set to eval mode for deterministic dropout

        returns_per_episode = np.zeros(self.config_env.eval_episodes)
        success_rate = np.zeros(self.config_env.eval_episodes)
        total_steps = np.zeros(self.config_env.eval_episodes)

        for task_idx in range(self.config_env.eval_episodes):
            step = 0
            running_reward = 0.0
            done_rollout = False

            obs = ptu.from_numpy(self.eval_env.reset())  # reset
            obs = obs.reshape(1, obs.shape[-1])

            if self.agent_arch == AGENT_ARCHS.Memory:
                # assume initial reward = 0.0
                action, reward, internal_state = self.agent.get_initial_info(
                    self.config_seq.sampled_seq_len
                )

            while not done_rollout:
                if self.agent_arch == AGENT_ARCHS.Memory:
                    action, internal_state = self.agent.act(
                        prev_internal_state=internal_state,
                        prev_action=action,
                        reward=reward,
                        obs=obs,
                        deterministic=deterministic,
                    )
                else:
                    action = self.agent.act(obs, deterministic=deterministic)

                # observe reward and next obs
                next_obs, reward, done, info = utl.env_step(
                    self.eval_env, action.squeeze(dim=0)
                )

                # add raw reward
                running_reward += reward.item()
                step += 1
                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                # set: obs <- next_obs
                obs = next_obs.clone()

            returns_per_episode[task_idx] = running_reward
            total_steps[task_idx] = step
            if "success" in info and info["success"] == True:  # keytodoor
                success_rate[task_idx] = 1.0

        self.agent.train()  # set it back to train
        return returns_per_episode, success_rate, total_steps

    def log_train_stats(self, train_stats):
        logger.record_step("env_steps", self._n_env_steps_total)
        ## log losses
        for k, v in train_stats.items():
            logger.record_tabular(k, v)
        ## gradient norms
        if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
            results = self.agent.report_grad_norm()
            for k, v in results.items():
                logger.record_tabular(k, v)
        logger.dump_tabular()

    def log(self):
        logger.record_step("env_steps", self._n_env_steps_total)
        returns_eval, success_rate_eval, total_steps_eval = self.evaluate()
        logger.record_tabular("return", np.mean(returns_eval))
        logger.record_tabular("success", np.mean(success_rate_eval))
        logger.record_tabular("length", np.mean(total_steps_eval))
        logger.record_tabular(
            "FPS",
            (self._n_env_steps_total - self._n_env_steps_total_last)
            / (time.time() - self._start_time_last),
        )
        self._n_env_steps_total_last = self._n_env_steps_total
        self._start_time_last = time.time()

        logger.dump_tabular()

        return np.mean(returns_eval)

    def save_model(self, total_steps, perf):
        save_path = os.path.join(
            logger.get_dir(),
            "save",
            f"agent_{total_steps:0{self._digit()}d}_perf{perf:.3f}.pt",
        )
        torch.save(self.agent.state_dict(), save_path)

    def load_model(self, ckpt_path):
        self.agent.load_state_dict(torch.load(ckpt_path, map_location=ptu.device))
        print("load successfully from", ckpt_path)

    def _digit(self):
        # zero pad with total env steps
        return int(math.log10(self.n_env_steps_total) + 1)
