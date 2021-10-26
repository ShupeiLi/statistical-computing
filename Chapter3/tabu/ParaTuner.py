# -*- coding: utf-8 -*-

import sys
sys.path.append(r".\TabuPy")
from ModelSelectionStandard import ModelSelectionStandard
from ModelSelectionLong import ModelSelectionLong
from ModelSelectionDeltaAIC import ModelSelectionDeltaAIC
from ModelSelectionInfluence import ModelSelectionInfluence
import pandas as pd
from tqdm import tqdm

file_path = r".\baseball.dat"

# Parameters
tenure_list = [i for i in range(1, 21)]
max_iteration = 100
seed = 1234
best_tenure = 5 # According to the results of tenureTrials(tenureList)
c_list = [i * 0.5 for i in range(1, 11)] # Best: c = 5
aic_list = [i * 0.1 for i in range(1, 21)] # Best: thresholdAIC = 1
r2_list = [i * 0.01 for i in range(1, 101)] # Best: thresholdR2 = 0.01


def tenure_trials(tenure_list, max_iteration, file_path, seed):
    """
    Find the best tenure in class ModelSelectionStandard \n
    Args: \n
        tenure_list: List of candidate values
    """
    result = []
    for i in tqdm(range(len(tenure_list))):
        tenure = tenure_list[i]
        model= ModelSelectionStandard(tenure, max_iteration, file_path, seed)
        history_solutions, history_aic = model.tabu_search()
        min_aic = min(history_aic)
        min_index = min(range(len(history_aic)), key = history_aic.__getitem__)
        result.append([tenure, min_aic, min_index])
    return pd.DataFrame(result, columns = ["Tenure", "AIC", "Iteration"])
    
def c_trials(c_list, tenure, max_iteration, file_path, seed):
    """
    Find the best c in class ModelSelectionLong \n
    Args: \n
        c_list: List of candidate values
    """
    result = []
    for i in tqdm(range(len(c_list))):
        c = c_list[i]
        model = ModelSelectionLong(tenure, max_iteration, file_path, c, seed)
        history_solutions, history_aic = model.tabu_search()
        min_aic = min(history_aic)
        min_index = min(range(len(history_aic)), key = history_aic.__getitem__)
        result.append([c, min_aic, min_index])
    return pd.DataFrame(result, columns = ["c", "AIC", "Iteration"])

def threshold_aic_trials(aic_list, tenure, max_iteration, file_path, seed):
    """
    Find the best threshold_aic in class ModelSelectionDeltaAIC \n
    Args: \n
        aic_list: List of candidate values
    """
    result = []
    for i in tqdm(range(len(aic_list))):
        threshold_aic = aic_list[i]
        model = ModelSelectionDeltaAIC(tenure, max_iteration, file_path, threshold_aic, seed)
        history_solutions, history_aic = model.tabu_search()
        min_aic = min(history_aic)
        min_index = min(range(len(history_aic)), key = history_aic.__getitem__)
        result.append([threshold_aic, min_aic, min_index])
    return pd.DataFrame(result, columns = ["threshold_aic", "AIC", "Iteration"])        

def threshold_r2_trials(r2_list, tenure, max_iteration, file_path, seed):
    """
    Find the best threshold_r2 in class ModelSelectionInfluence \n
    Args: \n
        r2_list: List of candidate values
    """
    result = []
    for i in tqdm(range(len(r2_list))):
        threshold_r2 = r2_list[i]
        model = ModelSelectionInfluence(tenure, max_iteration, file_path, threshold_r2, seed)
        history_solutions, history_aic = model.tabu_search()
        min_aic = min(history_aic)
        min_index = min(range(len(history_aic)), key = history_aic.__getitem__)
        result.append([threshold_r2, min_aic, min_index])
    return pd.DataFrame(result, columns = ["threshold_r2", "AIC", "Iteration"])        
        

if __name__ == '__main__':
    tenure_result = tenure_trials(tenure_list, max_iteration, file_path, seed)
    c_result = c_trials(c_list, best_tenure, max_iteration, file_path, seed)
    threshold_aic_result = threshold_aic_trials(aic_list, best_tenure, max_iteration, file_path, seed)
    threshold_r2_result = threshold_r2_trials(r2_list, best_tenure, max_iteration, file_path, seed)