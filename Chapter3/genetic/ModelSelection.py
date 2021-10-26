# -*- coding: utf-8 -*-

import sys
sys.path.append(r"..\problem")
from Baseball import input_data, aic
import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm


class ModelSelection():
    """
    Implement genetic algorithm in model selection problem \n
    
    Args:
        max_iteration: The max number of iterations \n
        population: Specify generation size \n
        file_path: The file path of the dataset \n
        seed: Random number seed. Default: 1234 \n
        strata: Specify the number of partition in tournament selection. Default: 1 \n
        fit_choice: Choose fitness function and strategy. e.g.: "33".
                    1: random selection; 2: selection with probability proportional to ﬁtness
                    3: selection with probability proportional to rank. Default: '11' \n
        z: Constant in the scaled fitness function. Default: 1.5 \n
        mutation_rate: Specify mutation rate. Default: 0.01
    """

    def __init__(self, max_iteration, population, file_path, seed = 1234, strata = 1, 
                 fit_choice = "11", z = 1.5, mutation_rate = 0.01):
        self.max_iteration = max_iteration
        self.population = population
        self.seed = seed
        self.strata = strata
        self.z = z
        self.fit_choice = fit_choice
        self.mutation_rate = mutation_rate
        self.x_full, self.y = input_data(file_path)

    def get_initial_solution(self):
        """
        Return the intial solution
        """
        np.random.seed(self.seed)
        return np.random.binomial(1, 0.5, (self.population, self.x_full.shape[1]))
    
    def aic(self, individual):
        """
        Calculate AIC
        """
        return np.array([aic(individual[i, :], self.x_full, self.y) for i in range(individual.shape[0])])
    
    def selection_randomly(self, parents):
        """
        Selection of one parent at random 
        """
        index = np.random.randint(self.population)
        return parents[index, :]
    
    def selection_proportion(self, parents):
        """
        Selection of one parent with probability proportional to scaled ﬁtness function
        """
        parents_aic = self.aic(parents)
        parents_fitness = -(parents_aic - (np.mean(parents_aic) - self.z * np.std(parents_aic)))
        for i in range(parents_fitness.shape[0]):
            parents_fitness[i] = parents_fitness[i] + np.abs(np.min(parents_fitness))
        parents_fitness_prop = parents_fitness / np.sum(parents_fitness)
        return parents[np.random.choice(np.arange(self.population), size = 1, p = parents_fitness_prop), :]
    
    def selection_rank(self, parents):
        """
        Selection of one parent with probability proportional to rank
        """
        parents_rank = rankdata(-self.aic(parents), method = "min")
        parents_fitness = (2 * parents_rank) / (self.population * (self.population + 1))
        parents_fitness_prop = parents_fitness / np.sum(parents_fitness)
        return parents[np.random.choice(np.arange(self.population), size = 1, p = parents_fitness_prop), :]
    
    def selection_procedure(self, parents):
        """
        Seletion of parents. Return two arrays
        """
        def parent_selection(parents, num):
            # Normal selection
            parent_lst = []
            for i in range(int(self.population / 2)):
                if num == "2":
                    parent_lst.append(self.selection_proportion(parents))
                elif num == "3":
                    parent_lst.append(self.selection_rank(parents))
                else:
                    parent_lst.append(self.selection_randomly(parents))
            return parent_lst
        
        def parent_tournament(parents):
            # Tournament selection
            winners = []
            strata_num = self.population // self.strata
            while len(winners) < int(self.population / 2):
                parents_copy = parents.copy()
                np.random.shuffle(parents_copy)
                for i in range(self.strata):
                    parents_strata = parents_copy[[j for j in range(i * strata_num, (i + 1) * strata_num)], :]
                    parents_strata_aic = self.aic(parents_strata)
                    index = np.argmin(parents_strata_aic)
                    winners.append(parents_copy[index, :])
            return winners[:int(self.population / 2)]
        
        if self.strata == 1:
            selected_fathers = parent_selection(parents, self.fit_choice[0])
            selected_mothers = parent_selection(parents, self.fit_choice[1])
        else:
            selected_fathers = parent_tournament(parents)
            selected_mothers = parent_tournament(parents)
            
        selected_fathers = np.array(selected_fathers).reshape((int(self.population / 2), self.x_full.shape[1]))
        selected_mothers = np.array(selected_mothers).reshape((int(self.population / 2), self.x_full.shape[1]))
        return selected_fathers, selected_mothers
    
    def crossover(self, selected_fathers, selected_mothers):
        """
        Implement crossover procedure
        """
        position = np.random.randint(1, self.x_full.shape[1] - 1)
        fathers_before = selected_fathers[:, [i for i in range(position)]]
        fathers_after = selected_fathers[:, [i for i in range(position, self.x_full.shape[1])]]
        mothers_before = selected_mothers[:, [i for i in range(position)]]
        mothers_after = selected_mothers[:, [i for i in range(position, self.x_full.shape[1])]]
        generations1 = np.concatenate((fathers_before, mothers_after), axis = 1)
        generations2 = np.concatenate((mothers_before, fathers_after), axis = 1)
        return np.concatenate((generations1, generations2), axis = 0)
    
    def mutation(self, generations):
        """
        Implement mutation procedure
        """
        mutation_matrix = np.random.binomial(1, self.mutation_rate, generations.shape)
        return np.array(np.abs(generations - mutation_matrix), dtype = np.int32)
        
    def genetic_algorithm(self):
        """
        Implement genetic algorithm \n
        
        Returns:
            history_generations: list \n
            history_aic: list
        """
        # Initiate
        current_generations = self.get_initial_solution()
        current_aic = self.aic(current_generations)
        history_generations = []
        history_generations.append(current_generations)
        history_aic = []
        history_aic.append(current_aic)
        
        # Main logic
        for iteration in tqdm(range(self.max_iteration)):
            selected_fathers, selected_mothers = self.selection_procedure(current_generations)
            generations = self.crossover(selected_fathers, selected_mothers)
            current_generations = self.mutation(generations)
            current_aic = self.aic(current_generations)
            history_generations.append(current_generations)
            history_aic.append(current_aic)
            
        return history_generations, history_aic
