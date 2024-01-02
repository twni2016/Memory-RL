from .td3.markov_td3_learner import MarkovTD3Learner
from .sac.markov_sac_learner import MarkovSACLearner
from .td3.memory_td3_learner import MemoryTD3Learner
from .sac.memory_sac_learner import MemorySACLearner
from .td3.sharedmemory_td3_learner import SharedMemoryTD3Learner
from .sac.sharedmemory_sac_learner import SharedMemorySACLearner


def get_learner(markov: bool, rl_algo_name: str, shared_encoder: bool):
    if markov:
        # separate encoders
        if rl_algo_name == "td3":
            return MarkovTD3Learner
        elif rl_algo_name == "sac":
            return MarkovSACLearner
        else:
            raise ValueError

    ## memory-based

    # actor-critic methods
    if shared_encoder:
        if rl_algo_name == "td3":
            return SharedMemoryTD3Learner
        elif rl_algo_name == "sac":
            return SharedMemorySACLearner
        else:
            raise ValueError
    else:
        # separate encoders
        if rl_algo_name == "td3":
            return MemoryTD3Learner
        elif rl_algo_name == "sac":
            return MemorySACLearner
        else:
            raise ValueError
