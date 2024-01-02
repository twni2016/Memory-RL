from ml_collections import ConfigDict
from typing import Tuple
from configs.seq_models.name_fns import name_fn


def attn_name_fn(config: ConfigDict, max_episode_steps: int) -> Tuple[ConfigDict, str]:
    config, name = name_fn(config, max_episode_steps)

    config.model.seq_model_config.hidden_size = 0
    if config.model.observ_embedder is not None:
        config.model.seq_model_config.hidden_size += (
            config.model.observ_embedder.hidden_dims[-1]
        )
    if config.model.action_embedder is not None:
        config.model.seq_model_config.hidden_size += (
            config.model.action_embedder.hidden_dims[-1]
        )
    if config.model.reward_embedder is not None:
        config.model.seq_model_config.hidden_size += (
            config.model.reward_embedder.hidden_dims[-1]
        )

    config.model.seq_model_config.max_seq_length = (
        config.sampled_seq_len + 1
    )  # NOTE: zero-prepend

    return config, name


def get_config():
    config = ConfigDict()
    config.name_fn = attn_name_fn

    config.is_markov = False
    config.is_attn = True
    config.use_dropout = True

    config.sampled_seq_len = -1

    config.clip = False
    config.max_norm = 1.0
    config.use_l2_norm = False
    config.lr = 3e-4

    # fed into Module
    config.model = ConfigDict()

    # seq_model_config specific
    config.model.seq_model_config = ConfigDict()
    config.model.seq_model_config.name = "gpt"

    config.model.seq_model_config.hidden_size = (
        128  # NOTE: will be overwritten by name_fn
    )
    config.model.seq_model_config.n_layer = 1
    config.model.seq_model_config.n_head = 1
    config.model.seq_model_config.pdrop = 0.1
    config.model.seq_model_config.position_encoding = "sine"

    # embedders
    config.model.observ_embedder = ConfigDict()
    config.model.observ_embedder.name = "mlp"
    config.model.observ_embedder.hidden_dims = (64,)
    config.model.observ_embedder.activate_final = True
    config.model.observ_embedder.add_layernorm = False
    config.model.observ_embedder.layernorm_use_bias_scale = True

    config.model.action_embedder = ConfigDict()
    config.model.action_embedder.name = "mlp"
    config.model.action_embedder.hidden_dims = (64,)
    config.model.action_embedder.activate_final = True
    config.model.action_embedder.add_layernorm = False
    config.model.action_embedder.layernorm_use_bias_scale = True

    config.model.reward_embedder = None

    return config
