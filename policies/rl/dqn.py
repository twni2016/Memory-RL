import torch
from .base import RLAlgorithmBase
from torchkit.networks import FlattenMlp
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import optax
import torch.nn.functional as F
import torchkit.pytorch_utils as ptu


class DQN(RLAlgorithmBase):
    name = "dqn"
    continuous_action = False

    def __init__(self, init_eps=1.0, end_eps=0.01, schedule_steps=1000, **kwargs):
        self.epsilon_schedule = optax.linear_schedule(
            init_value=init_eps,
            end_value=end_eps,
            transition_steps=schedule_steps,
        )
        self.count = 0

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim
        qf = FlattenMlp(
            input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes
        )
        return qf

    def select_action(self, qf, observ, deterministic: bool):
        action_logits = qf(observ)  # (B=1, A)
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)  # (*)
        else:
            random_action = torch.randint(
                high=action_logits.shape[-1], size=action_logits.shape[:-1]
            ).to(
                ptu.device
            )  # (*)
            optimal_action = torch.argmax(action_logits, dim=-1)  # (*)

            eps = self.epsilon_schedule(self.count).item()
            # mask = 0 means 1-eps exploit; mask = 1 means eps explore
            mask = torch.multinomial(
                input=ptu.FloatTensor([1 - eps, eps]),
                num_samples=action_logits.shape[0],
                replacement=True,
            )  # (*)
            action = mask * random_action + (1 - mask) * optimal_action

            self.count += 1
            # print(eps, self.count, random_action, optimal_action, action)

        # convert to one-hot vectors
        action = F.one_hot(
            action.long(), num_classes=action_logits.shape[-1]
        ).float()  # (*, A)
        return action

    def critic_loss(
        self,
        markov_critic: bool,
        critic,
        critic_target,
        observs,
        actions,
        rewards,
        dones,
        gamma,
        next_observs=None,  # used in markov_critic
    ):
        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            if markov_critic:  # (B, A)
                next_v = critic(next_observs)
                next_target_v = critic_target(next_observs)
            else:
                next_v = critic(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    current_actions=None,
                )  # (T+1, B, A)
                next_target_v = critic_target(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    current_actions=None,
                )  # (T+1, B, A)

            next_actions = torch.argmax(next_v, dim=-1, keepdim=True)  # (*, 1)
            next_target_q = next_target_v.gather(dim=-1, index=next_actions)  # (*, 1)

            # q_target: (T, B, 1)
            q_target = rewards + (1.0 - dones) * gamma * next_target_q  # next q
            if not markov_critic:
                q_target = q_target[1:]  # (T, B, 1)

        if markov_critic:
            v_pred = critic(observs)
            q_pred = v_pred.gather(dim=-1, index=actions.long())

        else:
            # Q(h(t), a(t)) (T, B, 1)
            v_pred = critic(
                prev_actions=actions[:-1],
                rewards=rewards[:-1],
                observs=observs[:-1],
                current_actions=None,
            )  # (T, B, A)

            stored_actions = actions[1:]  # (T, B, A)
            stored_actions = torch.argmax(
                stored_actions, dim=-1, keepdim=True
            )  # (T, B, 1)
            q_pred = v_pred.gather(
                dim=-1, index=stored_actions
            )  # (T, B, A) -> (T, B, 1)

        return q_pred, q_target
