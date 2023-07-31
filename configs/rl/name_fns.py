from ml_collections import ConfigDict
from typing import Tuple


def name_fn(config: ConfigDict, *args) -> Tuple[ConfigDict, str]:
    name = f"{config.algo}/"
    del config.name_fn
    return config, name
