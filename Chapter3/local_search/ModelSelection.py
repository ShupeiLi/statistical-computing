# -*- coding: utf-8 -*-

import sys
sys.path.append(r"..\problem")
from Baseball import input_data, get_initial_solution, get_neighborhood, aic
import pandas as pd
from tqdm import tqdm


class ModelSelection():
    """
    Implement local search algorithm in model selection problem \n
    
    Args:
        max_iteration: The max number of iterations \n
        k: k-neighborhoods \n
        file_path: The file path of the dataset \n
        seed: Random number seed. Default: 1234        
    """
    
    def __init__(self, max_iteration, k, file_path, seed = 1234):
        self.max_iteration = max_iteration
        self.k = k
        self.seed = seed
        self.x_full, self.y = input_data(file_path)
    
    def local_search(self):
        """
        Implementation local search algorithm \n
        
        Returns: \n
            history_solutions: list \n
            history_solutions_aic: list
        """
        # Initiate
        current_solution = get_initial_solution(self.seed, self.x_full.shape[1])
        current_aic = aic(current_solution, self.x_full, self.y)
        history_solutions = []
        history_solutions.append(current_solution)
        history_aic = []
        history_aic.append(current_aic)
        
        # Main logic
        for iteration in tqdm(range(self.max_iteration)):
            current_neighborhood = get_neighborhood(current_solution, self.k)
            current_neighborhood_lst = []
            
            # Steepest ascent
            for one_neighborhood in current_neighborhood:
                # Generate candidate solution
                for i in range(self.k):
                    candidate_solution = current_solution.copy()
                    candidate_solution[one_neighborhood[i][0]] = one_neighborhood[i][1]
                current_neighborhood_lst.append([one_neighborhood, candidate_solution, aic(candidate_solution, self.x_full, self.y)])
                
            current_neighborhood_df = pd.DataFrame(current_neighborhood_lst, columns = ["Neighborhood", "Candidate", "AIC"])
            current_neighborhood_df = current_neighborhood_df.sort_values(by = ["AIC"])
            
            current_solution = current_neighborhood_df.iloc[0, 1]
            current_aic = current_neighborhood_df.iloc[0, 2]
            history_solutions.append(current_solution)
            history_aic.append(current_aic)
                
        return history_solutions, history_aic