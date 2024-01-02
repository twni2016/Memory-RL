from ml_collections import ConfigDict
from typing import Tuple


def name_fn(config: ConfigDict, max_episode_steps: int) -> Tuple[ConfigDict, str]:
    name = ""
    if not config.is_markov:
        input_str = ""
        input_str += "o" if config.model.observ_embedder is not None else ""
        input_str += "a" if config.model.action_embedder is not None else ""
        input_str += "r" if config.model.reward_embedder is not None else ""
        name += f"{input_str}/"

    if config.sampled_seq_len == -1:
        config.sampled_seq_len = max_episode_steps

    name += f"{config.model.seq_model_config.name}-len-{config.sampled_seq_len}/"

    assert config.clip is False

    del config.name_fn
    return config, name
