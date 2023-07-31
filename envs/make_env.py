import gym
from gym.wrappers import RescaleAction


def make_env(
    env_name: str,
    seed: int,
) -> gym.Env:
    # Check if the env is in gym.
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    if env_name in env_ids:
        env = gym.make(env_name)

    env.max_episode_steps = getattr(
        env, "max_episode_steps", env.spec.max_episode_steps
    )

    if isinstance(env.action_space, gym.spaces.Box):
        env = RescaleAction(env, -1.0, 1.0)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    print("obs space", env.observation_space)
    print("act space", env.action_space)

    return env
