import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]
