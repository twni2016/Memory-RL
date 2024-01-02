from ml_collections import config_dict
import numpy as np
import jax

# Valid types according to
# https://github.com/tensorflow/tensorboard/blob/1204566da5437af55109f7a4af18f9f8b7c4f864/tensorboard/plugins/hparams/summary_v2.py
valid_types = (
    bool,
    int,
    float,
    str,
    np.bool_,
    np.integer,
    np.floating,
    np.character,
)


def flatten_dict(input_dict: config_dict, parent_key="", sep="."):
    """https://github.com/google/flax/blob/main/flax/metrics/tensorboard.py
    Flattens and simplifies dict such that it can be used by hparams.
    Args:
            input_dict: Input dict, e.g., from ConfigDict.
            parent_key: String used in recursion.
            sep: String used to separate parent and child keys.
    Returns:
    Flattened dict.
    """

    items = []
    for k, v in input_dict.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, config_dict.ConfigDict):
            # Recursively flatten the configdict.
            items.extend(flatten_dict(v, new_key, sep=sep).items())
            continue
        elif not isinstance(v, valid_types):
            # Cast any incompatible values as strings such that they can be handled by hparams
            v = str(v)
        items.append((new_key, v))

    return dict(items)


def strify(input_dict: config_dict):
    return jax.tree_util.tree_map(
        lambda v: v if isinstance(v, valid_types) else str(v), input_dict.to_dict()
    )
