# -*- coding: utf-8 -*-

import sys
sys.path.append(r"..\problem")
from Baseball import input_data, get_initial_solution, get_neighborhood, aic
from random import sample
from tqdm import tqdm
import numpy as np


class ModelSelection():
    """
    Implement simulated annealing algorithm in model selection problem \n
    
    Args:
        k: k-neighborhoods \n
        tao: Initial temperature \n
        file_path: The file path of the dataset \n
        seed: Random number seed. Default: 1234        
    """
    
    def __init__(self, k, tau, file_path, seed = 1234):
        self.k = k
        self.tau = tau
        self.seed = seed
        self.x_full, self.y = input_data(file_path)
        self.ALPHA = 0.9
        self.M = (60, 60, 60, 60, 60, 120, 120, 120, 120, 120,
                  220, 220, 220, 220, 220)
        
    def annealing(self):
        """
        Implementation simulated annealing algorithm \n
        
        Returns: \n
            history_solutions: list \n
            history_solutions_aic: list
        """
        # Initiate
        current_solution = get_initial_solution(self.seed, self.x_full.shape[1])
        current_aic = aic(current_solution, self.x_full, self.y)
        current_tau = self.tau
        history_solutions = []
        history_solutions.append(current_solution)
        history_aic = []
        history_aic.append(current_aic)
        
        # Main logic        
        for stage in tqdm(range(len(self.M))):
            for one_time in range(self.M[stage]):
                # Discrete uniform
                current_neighborhood = sample([list(i) for i in get_neighborhood(current_solution, self.k)], 1)[0]
                for i in range(self.k):
                    candidate_solution = current_solution.copy()
                    candidate_solution[current_neighborhood[i][0]] = current_neighborhood[i][1]
                candidate_aic = aic(candidate_solution, self.x_full, self.y)
                # Randomly decision
                try:
                    boltzmann = np.min(np.array([1, np.exp((current_aic - candidate_aic) / current_tau)]))
                except OverflowError:
                    boltzmann = 1
                decision = np.random.choice(np.array([0, 1]), size = 1, replace = False,
                                            p = np.array([1 - boltzmann, boltzmann]))[0]
                if candidate_aic < current_aic or decision == 1:
                    current_solution = candidate_solution
                    current_aic = candidate_aic
                history_solutions.append(current_solution)
                history_aic.append(current_aic)
            # Update the temperature
            current_tau = self.ALPHA * current_tau
                
        return history_solutions, history_aic
