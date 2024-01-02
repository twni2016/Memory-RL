import numpy as np
import random
import datetime
import dateutil.tz


def reproduce(seed):
    """
    This can only fix the randomness of numpy and torch
    To fix the environment's, please use
        env.seed(seed)
        env.action_space.np_random.seed(seed)
    We have add these in our training script
    """
    assert seed >= 0
    np.random.seed(seed)
    random.seed(seed)


def now_str():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime(
        "%Y-%m-%d-%H-%M-%S"
    )  # may cause collision, please use PID to prevent
