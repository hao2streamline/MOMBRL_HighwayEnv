import gymnasium as gym
import highway_env

env = gym.make("highway-v0")
obs = env.reset()
print(obs)  # 打印观测空间维度
