import gymnasium as gym
import highway_env

class HighwayEnv:
    def __init__(self, config):
        self.env = gym.make("highway-v0", render_mode = "rgb_array_list")
        self.env.configure(config)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def __getattr__(self, name):
        return getattr(self.env, name)
