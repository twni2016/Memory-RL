from ml_collections import ConfigDict
from configs.seq_models.name_fns import name_fn


def get_config():
    config = ConfigDict()
    config.name_fn = name_fn

    config.is_markov = False
    config.is_attn = False
    config.use_dropout = False

    config.sampled_seq_len = -1

    config.clip = False
    config.max_norm = 1.0
    config.use_l2_norm = False
    config.lr = 3e-4

    # fed into Module
    config.model = ConfigDict()

    # seq_model specific
    config.model.seq_model_config = ConfigDict()
    config.model.seq_model_config.name = "lstm"
    config.model.seq_model_config.hidden_size = 128

    # embedders
    config.model.observ_embedder = ConfigDict()
    config.model.observ_embedder.name = "mlp"
    config.model.observ_embedder.hidden_dims = (32,)
    config.model.observ_embedder.activate_final = True
    config.model.observ_embedder.add_layernorm = False
    config.model.observ_embedder.layernorm_use_bias_scale = True

    config.model.action_embedder = ConfigDict()
    config.model.action_embedder.name = "mlp"
    config.model.action_embedder.hidden_dims = (16,)
    config.model.action_embedder.activate_final = True
    config.model.action_embedder.add_layernorm = False
    config.model.action_embedder.layernorm_use_bias_scale = True

    config.model.reward_embedder = None

    return config
