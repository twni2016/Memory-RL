import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from utils import helpers as utl
from policies.rl import RL_ALGORITHMS
from policies.seq_models import SEQ_MODELS
import torchkit.pytorch_utils as ptu
from utils import logger


class ModelFreeOffPolicy_Shared_RNN(nn.Module):
    """
    Recurrent Actor and Recurrent Critic with shared RNN
    We find `freeze_critic = True` can prevent degradation shown in https://github.com/twni2016/pomdp-baselines
    """

    ARCH = "memory"

    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
        config_rl,
        freeze_critic: bool,
        # pixel obs
        image_encoder_fn=lambda: None,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = config_rl.discount
        self.tau = config_rl.tau
        self.clip = config_seq.clip
        self.clip_grad_norm = config_seq.max_norm

        self.freeze_critic = freeze_critic

        self.algo = RL_ALGORITHMS[config_rl.algo](
            action_dim=action_dim, **config_rl.to_dict()
        )

        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)
        if image_encoder_fn() is None:
            observ_embedding_size = config_seq.model.observ_embedder.hidden_size
            self.observ_embedder = utl.FeatureExtractor(
                obs_dim, observ_embedding_size, F.relu
            )
        else:  # for pixel observation, use external encoder
            self.observ_embedder = image_encoder_fn()
            observ_embedding_size = self.observ_embedder.embedding_size  # reset it

        self.action_embedder = utl.FeatureExtractor(
            action_dim, config_seq.model.action_embedder.hidden_size, F.relu
        )
        self.reward_embedder = utl.FeatureExtractor(
            1, config_seq.model.reward_embedder.hidden_size, F.relu
        )

        ## 2. build RNN model
        rnn_input_size = (
            observ_embedding_size
            + config_seq.model.action_embedder.hidden_size
            + config_seq.model.reward_embedder.hidden_size
        )
        self.seq_model = SEQ_MODELS[config_seq.model.seq_model_config.name](
            input_size=rnn_input_size, **config_seq.model.seq_model_config.to_dict()
        )

        ## 3. build actor-critic
        # q-value networks
        self.qf1, self.qf2 = self.algo.build_critic(
            input_size=self.seq_model.hidden_size + action_dim
            if self.algo.continuous_action
            else self.seq_model.hidden_size,
            hidden_sizes=config_rl.config_critic.hidden_dims,
            action_dim=action_dim,
        )
        # target networks
        self.qf1_target = deepcopy(self.qf1)
        self.qf2_target = deepcopy(self.qf2)

        # policy network
        self.policy = self.algo.build_actor(
            input_size=self.seq_model.hidden_size,
            action_dim=self.action_dim,
            hidden_sizes=config_rl.config_actor.hidden_dims,
        )
        # target networks
        self.policy_target = deepcopy(self.policy)

        # use joint optimizer
        assert config_rl.critic_lr == config_rl.actor_lr
        self.optimizer = Adam(self._get_parameters(), lr=config_rl.critic_lr)

    def _get_parameters(self):
        # exclude targets
        return [
            *self.observ_embedder.parameters(),
            *self.action_embedder.parameters(),
            *self.reward_embedder.parameters(),
            *self.seq_model.parameters(),
            *self.qf1.parameters(),
            *self.qf2.parameters(),
            *self.policy.parameters(),
        ]

    def get_hidden_states(
        self, prev_actions, rewards, observs, initial_internal_state=None
    ):
        # all the input have the shape of (T+1, B, *)
        # get embedding of initial transition
        input_a = self.action_embedder(prev_actions)
        input_r = self.reward_embedder(rewards)
        input_s = self.observ_embedder(observs)
        inputs = torch.cat((input_a, input_r, input_s), dim=-1)

        # feed into RNN: output (T+1, B, hidden_size)
        if initial_internal_state is None:  # training
            initial_internal_state = self.seq_model.get_zero_internal_state(
                batch_size=inputs.shape[1]
            )  # initial_internal_state is zeros
            output, _ = self.seq_model(inputs, initial_internal_state)
            return output
        else:  # useful for one-step rollout during testing
            output, current_internal_state = self.seq_model(
                inputs, initial_internal_state
            )
            return output, current_internal_state

    def forward(self, actions, rewards, observs, dones, masks):
        """
        For actions a, rewards r, observs o, dones d: (T+1, B, dim)
                where for each t in [0, T], take action a[t], then receive reward r[t], done d[t], and next obs o[t]
                the hidden state h[t](, c[t]) = RNN(h[t-1](, c[t-1]), a[t], r[t], o[t])
                specially, a[0]=r[0]=d[0]=h[0]=c[0]=0.0, o[0] is the initial obs

        The loss is still on the Q value Q(h[t], a[t]) with real actions taken, i.e. t in [1, T]
                based on Masks (T, B, 1)
        """
        assert (
            actions.dim()
            == rewards.dim()
            == dones.dim()
            == observs.dim()
            == masks.dim()
            == 3
        )
        assert (
            actions.shape[0]
            == rewards.shape[0]
            == dones.shape[0]
            == observs.shape[0]
            == masks.shape[0] + 1
        )
        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss

        ### 1. get hidden/belief states of the whole/sub trajectories, aligned with observs
        # return the hidden states (T+1, B, dim)
        # import time; t0 = time.time()
        hidden_states = self.get_hidden_states(
            prev_actions=actions, rewards=rewards, observs=observs
        )
        # print("forward seq model", time.time() - t0)
        # NOTE: cost 30% time of single pass

        ### 2. Critic loss

        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            # first next_actions from target/current policy, (T+1, B, dim) including reaction to last obs
            # new_next_actions: (T+1, B, dim), new_next_log_probs: (T+1, B, 1 or A)
            new_next_actions, new_next_log_probs = self.algo.forward_actor_in_target(
                actor=self.policy,
                actor_target=self.policy_target,
                next_observ=hidden_states,
            )
            if self.algo.continuous_action:
                joint_q_embeds = torch.cat(
                    (hidden_states, new_next_actions), dim=-1
                )  # (T+1, B, dim)
            else:
                joint_q_embeds = hidden_states

            next_q1 = self.qf1_target(joint_q_embeds)  # return (T, B, 1 or A)
            next_q2 = self.qf2_target(joint_q_embeds)
            min_next_q_target = torch.min(next_q1, next_q2)

            # min_next_q_target (T+1, B, 1 or A)
            min_next_q_target += self.algo.entropy_bonus(new_next_log_probs)
            if not self.algo.continuous_action:
                min_next_q_target = (new_next_actions * min_next_q_target).sum(
                    dim=-1, keepdims=True
                )  # (T+1, B, 1)

            q_target = rewards + (1.0 - dones) * self.gamma * min_next_q_target
            q_target = q_target[1:]  # (T, B, 1)

        # Q(h(t), a(t)) (T, B, 1)
        # 3. joint embeds
        if self.algo.continuous_action:
            curr_joint_q_embeds = torch.cat(
                (hidden_states[:-1], actions[1:]), dim=-1
            )  # (T, B, dim)
        else:
            curr_joint_q_embeds = hidden_states[:-1]

        q1_pred = self.qf1(curr_joint_q_embeds)
        q2_pred = self.qf2(curr_joint_q_embeds)
        if not self.algo.continuous_action:
            stored_actions = actions[1:]  # (T, B, A)
            stored_actions = torch.argmax(
                stored_actions, dim=-1, keepdims=True
            )  # (T, B, 1)
            q1_pred = q1_pred.gather(
                dim=-1, index=stored_actions
            )  # (T, B, A) -> (T, B, 1)
            q2_pred = q2_pred.gather(
                dim=-1, index=stored_actions
            )  # (T, B, A) -> (T, B, 1)

        # masked Bellman error: masks (T,B,1) ignore the invalid error
        # this is not equal to masks * q1_pred, cuz the denominator in mean()
        # 	should depend on masks > 0.0, not a constant B*T
        q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
        q_target = q_target * masks

        qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid  # TD error
        qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid  # TD error

        ### 3. Actor loss
        # Q(h(t), pi(h(t))) + H[pi(h(t))]
        # new_actions: (T+1, B, dim)
        new_actions, new_log_probs = self.algo.forward_actor(
            actor=self.policy, observ=hidden_states
        )

        if self.freeze_critic:
            ######## freeze critic parameters
            ######## and detach critic hidden states
            ######## such that the gradient only through new_actions
            if self.algo.continuous_action:
                new_joint_q_embeds = torch.cat(
                    (hidden_states.detach(), new_actions), dim=-1
                )  # (T+1, B, dim)
            else:
                new_joint_q_embeds = hidden_states.detach()

            freezed_qf1 = deepcopy(self.qf1).to(ptu.device)
            freezed_qf2 = deepcopy(self.qf2).to(ptu.device)
            q1 = freezed_qf1(new_joint_q_embeds)
            q2 = freezed_qf2(new_joint_q_embeds)

        else:
            if self.algo.continuous_action:
                new_joint_q_embeds = torch.cat(
                    (hidden_states, new_actions), dim=-1
                )  # (T+1, B, dim)
            else:
                new_joint_q_embeds = hidden_states

            q1 = self.qf1(new_joint_q_embeds)
            q2 = self.qf2(new_joint_q_embeds)

        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1 or A)

        policy_loss = -min_q_new_actions
        policy_loss += -self.algo.entropy_bonus(new_log_probs)

        if not self.algo.continuous_action:
            policy_loss = (new_actions * policy_loss).sum(
                axis=-1, keepdims=True
            )  # (T+1,B,1)
            new_log_probs = (new_actions * new_log_probs).sum(
                axis=-1, keepdims=True
            )  # (T+1,B,1)

        policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs
        policy_loss = (policy_loss * masks).sum() / num_valid

        ### 4. update
        total_loss = 0.5 * (qf1_loss + qf2_loss) + policy_loss

        outputs = {
            "critic_loss": (qf1_loss + qf2_loss).item(),
            "q1": (q1_pred.sum() / num_valid).item(),
            "q2": (q2_pred.sum() / num_valid).item(),
            "actor_loss": policy_loss.item(),
        }

        # import time; t0 = time.time()
        self.optimizer.zero_grad()
        total_loss.backward()
        # print("backward", time.time() - t0)
        # NOTE: cost 2/3 time of single pass

        if self.clip and self.clip_grad_norm > 0.0:
            grad_norm = nn.utils.clip_grad_norm_(
                self._get_parameters(), self.clip_grad_norm
            )
            outputs["raw_grad_norm"] = grad_norm.item()

        self.optimizer.step()

        ### 5. soft update
        self.soft_target_update()

        ### 6. update others like alpha
        if new_log_probs is not None:
            # extract valid log_probs
            with torch.no_grad():
                current_log_probs = (new_log_probs[:-1] * masks).sum() / num_valid
                current_log_probs = current_log_probs.item()

            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.policy, self.policy_target, self.tau)

    def report_grad_norm(self):
        return {
            "seq_grad_norm": utl.get_grad_norm(self.seq_model),
            "critic_grad_norm": utl.get_grad_norm(self.qf1),
            "actor_grad_norm": utl.get_grad_norm(self.policy),
        }

    def update(self, batch):
        # all are 3D tensor (T,B,dim)
        actions, rewards, dones = batch["act"], batch["rew"], batch["term"]
        _, batch_size, _ = actions.shape
        if not self.algo.continuous_action:
            # for discrete action space, convert to one-hot vectors
            actions = F.one_hot(
                actions.squeeze(-1).long(), num_classes=self.action_dim
            ).float()  # (T, B, A)

        masks = batch["mask"]
        obs, next_obs = batch["obs"], batch["obs2"]  # (T, B, dim)

        # extend observs, actions, rewards, dones from len = T to len = T+1
        observs = torch.cat((obs[[0]], next_obs), dim=0)  # (T+1, B, dim)
        actions = torch.cat(
            (ptu.zeros((1, batch_size, self.action_dim)).float(), actions), dim=0
        )  # (T+1, B, dim)
        rewards = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), rewards), dim=0
        )  # (T+1, B, dim)
        dones = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), dones), dim=0
        )  # (T+1, B, dim)

        # import time; t0 = time.time()
        outputs = self.forward(actions, rewards, observs, dones, masks)
        # print("single pass", time.time() - t0)
        return outputs

    @torch.no_grad()
    def get_initial_info(self, max_attn_span: int = -1):
        # here we assume batch_size = 1

        ## here we set the ndim = 2 for action and reward for compatibility
        prev_action = ptu.zeros((1, self.action_dim)).float()
        reward = ptu.zeros((1, 1)).float()
        internal_state = self.seq_model.get_zero_internal_state()
        return prev_action, reward, internal_state

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        reward,
        obs,
        deterministic=False,
    ):
        # for evaluation (not training), so no target actor, and T = 1
        # a function that generates action, works like a pytorch module
        prev_action = prev_action.unsqueeze(0)  # (1, B, dim)
        reward = reward.unsqueeze(0)  # (1, B, 1)
        obs = obs.unsqueeze(0)  # (1, B, 1)

        hidden_state, current_internal_state = self.get_hidden_states(
            prev_actions=prev_action,
            rewards=reward,
            observs=obs,
            initial_internal_state=prev_internal_state,
        )
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(0)  # (B, dim)

        # 4. Actor head, generate action tuple
        current_action = self.algo.select_action(
            actor=self.policy,
            observ=hidden_state,
            deterministic=deterministic,
        )

        return current_action, current_internal_state
