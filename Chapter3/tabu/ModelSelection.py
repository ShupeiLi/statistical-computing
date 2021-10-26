# -*- coding: utf-8 -*-

from Template import Template
from abc import abstractmethod
import numpy as np
import sys
sys.path.append(r"..\problem")
from Baseball import input_data, get_initial_solution, get_neighborhood, aic


class ModelSelection(Template):
    """
    A template for model selection problem with tabu algorithm \n
    
    Args: \n
        tenure: The number of iterations over which an attribute is tabu \n
        max_iteration: The max number of iterations \n
        file_path: The file path of the dataset \n
        seed: Random number seed. Default: 1234
    """
    
    @abstractmethod
    def __init__(self, tenure, max_iteration, file_path, seed = 1234):
        self.tenure = tenure
        self.max_iteration = max_iteration
        self.file_path = file_path
        self.seed = seed
        self.x_full, self.y = self.input_data()
        
    def input_data(self):
        return input_data(self.file_path)
    
    def get_initial_solution(self):
        return get_initial_solution(self.seed, self.x_full.shape[1])

    def get_neighborhood(self, current_solution):
        return get_neighborhood(current_solution, 1)

    def get_tabu_structure(self):
        """
        Return a dict of tabu attributes as keys and tabu_times as values
        """
        one_list = list(enumerate(np.ones(self.x_full.shape[1], dtype = np.int32).tolist()))
        zero_list = list(enumerate(np.zeros(self.x_full.shape[1], dtype = np.int32).tolist()))
        tabu_list = [*one_list, *zero_list]
        tabu_dict = {}
        for item in tabu_list:
            tabu_dict[item] = 0
        return tabu_dict

    def obj_fun(self, solution):
        return aic(solution, self.x_full, self.y)