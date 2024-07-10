import numpy as np
from deap import base, creator, tools, algorithms
from utils.env import VirtualHighwayEnv
import gymnasium as gym

class EvolutionaryPolicy:
    def __init__(self, config, env, model):
        self.population_size = config["population_size"]
        self.mutation_rate = config["mutation_rate"]
        self.crossover_rate = config["crossover_rate"]
        self.multi_objective = config["multi_objective"]
        self.model = model
        self.env = env

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        if isinstance(env.action_space, gym.spaces.Discrete):
            n_actions = env.action_space.n
            self.toolbox.register("attr_int", np.random.randint, 0, n_actions)
            self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_int, n=1)
        elif isinstance(env.action_space, gym.spaces.Box):
            n_actions = env.action_space.shape[0]
            self.toolbox.register("attr_float", np.random.uniform, low=env.action_space.low, high=env.action_space.high)
            self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=n_actions)
        else:
            raise ValueError("Unsupported action space type")

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.get_evaluate_function(env, model))

        self.population = self.toolbox.population(n=self.population_size)

    def get_evaluate_function(self, env, model):
        def evaluate(individual):
            virtual_env = VirtualHighwayEnv(env, model)
            obs = virtual_env.reset()
            total_reward = 0
            for _ in range(100):
                if isinstance(env.action_space, gym.spaces.Discrete):
                    action = np.clip(individual[0], 0, env.action_space.n - 1)  # 确保动作在有效范围内
                else:
                    action = np.clip(np.array(individual), env.action_space.low, env.action_space.high)  # 确保动作在有效范围内
                obs, reward, done, info = virtual_env.step(action)
                total_reward += reward  # 现在 reward 应该是一个标量
                if done:
                    break
            return total_reward,

        return evaluate

    def evolve(self):
        population = self.toolbox.population(n=self.population_size)
        NGEN = 1  # 迭代次数
        CXPB = 0.5  # 交叉概率
        MUTPB = 0.2  # 变异概率

        for gen in range(NGEN):
            fitnesses = list(map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.rand() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring

            fits = [ind.fitness.values[0] for ind in population]
            length = len(population)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            print(f"Generation {gen}: Max {max(fits)}, Avg {mean}, Std {std}")

        best_ind = tools.selBest(population, 1)[0]
        self.best_individual = best_ind
        print("Best individual is:", best_ind)
        print("With fitness:", best_ind.fitness.values)

        return best_ind


    def predict(self, obs):
        # 使用最优个体的值作为预测的动作
        best_ind = tools.selBest(self.population, 1)[0]
        print("Best individual in predict:", self.best_individual)  # 打印最优个体
        return np.array(best_ind)
