from .sacd import SACD
from .dqn import DQN

RL_ALGORITHMS = {
    SACD.name: SACD,
    DQN.name: DQN,
}
