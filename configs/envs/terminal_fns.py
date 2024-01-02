import gym


def finite_horizon_terminal(env: gym.Env, done: bool, info: dict) -> bool:
    return done


def infinite_horizon_terminal(env: gym.Env, done: bool, info: dict) -> bool:
    if not done or "TimeLimit.truncated" in info:
        terminal = False
    else:
        terminal = True
    return terminal


# meta RL terminals
