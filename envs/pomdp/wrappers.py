import gym
from gym import spaces
import numpy as np


class OcclusionWrapper(gym.Wrapper):
    def __init__(self, env, partially_obs_dims: list):
        super().__init__(env)
        self.partially_obs_dims = partially_obs_dims
        # can equal to the fully-observed env
        assert 0 < len(self.partially_obs_dims) <= self.observation_space.shape[0]

        self.observation_space = spaces.Box(
            low=self.observation_space.low[self.partially_obs_dims],
            high=self.observation_space.high[self.partially_obs_dims],
            dtype=np.float32,
        )

    def get_unobs(self):
        return self.state[
            ~np.isin(np.arange(len(self.state)), self.partially_obs_dims)
        ].copy()

    def get_obs(self):
        return self.state[self.partially_obs_dims].copy()

    def reset(self):
        self.state = self.env.reset()  # no kwargs
        return self.get_obs()

    def step(self, action):
        self.state, reward, done, info = self.env.step(action)

        return self.get_obs(), reward, done, info


if __name__ == "__main__":
    import envs.pomdp

    env = gym.make("HopperBLT-F-v0")
    obs = env.reset()
    done = False
    step = 0
    while not done:
        next_obs, rew, done, info = env.step(env.action_space.sample())
        step += 1
        print(step, done, info)
