# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from itertools import combinations
from random import sample
from tqdm import tqdm


class GeneticMapping():
    """
    Implement local search algorithm in genetic mapping problem \n
    
    Args:
        max_iteration: The max number of iterations \n
        file_path: The file path of the dataset \n
        seed: Random number seed. Default: 1234 \n
        steepest: Use steepest descent. Default: False \n
        size: The size of neighborhood. Default: 20 \n
        optim: Adopt optimized search strategy. Default: False
    """
    
    def __init__(self, max_iteration, file_path, seed = 1234, steepest = False, size = 20, optim = False):
        self.max_iteration = max_iteration
        self.seed = seed
        self.steepest = steepest
        self.size = size
        self.data = pd.read_table(file_path, sep = " ").to_numpy(dtype = np.float64)
        self.optim = optim
        
    def get_initial_solution(self):
        """
        Return the intial solution
        """        
        initial_solution = np.arange(1, self.data.shape[1] + 1)
        np.random.seed(self.seed)
        np.random.shuffle(initial_solution)
        return initial_solution
    
    def get_neighborhood(self, current_solution):
        """
        Return the neighborhood of current solution
        """
        index_lst = [i for i in range(self.data.shape[1])]
        swaps = combinations(index_lst, 2)
        neighbors = []
        size = self.size
        
        for swap in swaps:
            candidate = current_solution.copy()
            candidate[swap[0]], candidate[swap[1]] = candidate[swap[1]], candidate[swap[0]]
            neighbors.append(candidate)
        
        if self.optim:
            neighbors_lst = [list(item) for item in neighbors]
            for neighbor in neighbors_lst:
                if [element for element in reversed(neighbor)] in neighbors_lst:
                    neighbors_lst.remove(neighbor)
            neighbors = [np.array(neighbor) for neighbor in neighbors_lst]
            if len(neighbors) < size:
                size = len(neighbors)
            
        if self.steepest:
            return neighbors
        else:
            return sample(neighbors, size)
            
    def distances_fun(self, solution):
        """
        Calculate estimated distances
        """
        permutation = solution - 1
        data_reorder_j = self.data[:, permutation]
        data_reorder_j_plus_1 = self.data[:, np.append(permutation[1:], permutation[0])]
        return np.mean(np.abs(data_reorder_j_plus_1 - data_reorder_j), axis = 0)[:-1]
    
    def likelihood_fun(self, distances):
        """
        Calculate profile log-likehood
        """
        likelihood = self.data.shape[0] * (distances * np.log(distances) + (1 - distances) * np.log(1 - distances))
        for i in range(distances.shape[0]):
            if distances[i] == 0 or distances[i] == 1:
                likelihood[i] = 0
        return np.sum(likelihood)

    def local_search(self):
        """
        Implementation local search algorithm \n
        
        Returns: \n
            history_solutions: list \n
            history_distances: list \n
            history_likelihood : list
        """
        # Initiate
        current_solution = self.get_initial_solution()
        current_distances = self.distances_fun(current_solution)
        current_likelihood = self.likelihood_fun(current_distances)
        history_solutions = []
        history_solutions.append(current_solution)
        history_distances = []
        history_distances.append(current_distances)
        history_likelihood = []
        history_likelihood.append(current_likelihood)

        # Main logic
        for iteration in tqdm(range(self.max_iteration)):
            candidate_solutions = self.get_neighborhood(current_solution)
            candidate_solutions_lst = []
            
            for candidate_solution in candidate_solutions:
                candidate_distances = self.distances_fun(candidate_solution)
                candidate_solutions_lst.append([candidate_solution, candidate_distances, self.likelihood_fun(candidate_distances)])
            candidate_solutions_df = pd.DataFrame(candidate_solutions_lst, columns = ["Candidate", "Distances", "Likelihood"])
            candidate_solutions_df = candidate_solutions_df.sort_values(by = ["Likelihood"], ascending = False)
            
            current_solution = candidate_solutions_df.iloc[0, 0]
            current_distances = candidate_solutions_df.iloc[0, 1]
            current_likelihood = candidate_solutions_df.iloc[0, 2]
            history_solutions.append(current_solution)
            history_distances.append(current_distances)
            history_likelihood.append(current_likelihood)
                
        return history_solutions, history_distances, history_likelihood 
  