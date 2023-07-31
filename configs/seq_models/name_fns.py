from ml_collections import ConfigDict
from typing import Tuple


def name_fn(config: ConfigDict, max_episode_steps: int) -> Tuple[ConfigDict, str]:
    name = ""

    if config.sampled_seq_len == -1:
        config.sampled_seq_len = max_episode_steps

    name += f"{config.model.seq_model_config.name}-len-{config.sampled_seq_len}/"

    assert config.clip is False

    del config.name_fn
    return config, name
