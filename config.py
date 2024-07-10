class Config:
    def __init__(self):
        self.env_config = {
            "config_key": "config_value"
        }
        self.model_config = {
            "input_dim": 5,  # 调整为环境的观测空间维度
            "output_dim": 2,  # 你可以根据需求调整输出维度
            "hidden_layers": [6, 6]  # 简化的隐藏层配置
        }
        self.policy_config = {
            "population_size": 10,  # 减少种群大小
            "mutation_rate": 0.1,
            "crossover_rate": 0.5,
            "multi_objective": ["fuel_consumption", "distance"]
        }
        self.log_config = {
            "log_dir": "./logs"
        }
        self.num_episodes = 100  # 减少训练回合数
