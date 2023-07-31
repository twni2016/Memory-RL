import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.channels = (8, 16)
    config.kernel_sizes = (2, 2)
    config.strides = (1, 1)
    config.embedding_size = 100

    return config
