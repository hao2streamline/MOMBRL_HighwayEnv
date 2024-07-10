import gymnasium as gym
import highway_env
import numpy as np
from deap import base, creator, tools, algorithms
from config import Config
from environments.highway import HighwayEnv
from models.mlp import MLPModel
from policies.evolution import EvolutionaryPolicy
from utils.logger import Logger

def main():
    config = Config()
    env = HighwayEnv(config.env_config)
    model = MLPModel(config.model_config)
    policy = EvolutionaryPolicy(config.policy_config, env, model)
    logger = Logger(config.log_config)

    best_policy = policy.evolve()

    # 评估最优策略
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = policy.predict(obs)
        print("Action:",action)
        obs, reward, done, info2,info = env.step(action)
        total_reward += reward
        logger.log_step(reward, info)
    logger.log_episode()
    logger.save()

    print(f"Total reward with best policy: {total_reward}")

if __name__ == "__main__":
    main()
