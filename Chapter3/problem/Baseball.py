# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations


"""
Problems based on the baseball data set \n

Args:
    max_iteration: The max number of iterations \n
    k: k-neighborhoods \n
    file_path: The file path of the dataset \n
    seed: Random number seed. Default: 1234        
"""

def input_data(file_path):
    """
    Input dataset
    """
    data = pd.read_table(file_path, sep = " ")
    x = data.iloc[:, 1:].to_numpy()
    y = np.log(data.iloc[:, 0].to_numpy())
    return x, y

def get_initial_solution(seed, x_full_shape):
    """
    Return the intial solution
    """
    np.random.seed(seed)
    return np.random.binomial(1, 0.5, x_full_shape)

def get_neighborhood(current_solution, k):
    """
    Return the neighborhood of current solution, [(predictor, move)]
    
    Args:
        k: Specify k-neighborhood
    """
    neighborhood_set = list(enumerate(np.abs(current_solution - 1).tolist()))  
    return combinations(neighborhood_set, k)
    
def aic(solution, x_full, y):
    """
    Calculate AIC
    """
    x_index = [item[0] for item in filter(lambda x: x[1] == 1, list(enumerate(solution)))]
    x_selected = sm.add_constant(x_full[:, x_index])
    model = sm.OLS(y, x_selected).fit()
    return model.aic