# -*- coding: utf-8 -*-

import sys
sys.path.append(r".\TabuPy")
from ModelSelectionStandard import ModelSelectionStandard
from ModelSelectionLong import ModelSelectionLong
from ModelSelectionDeltaAIC import ModelSelectionDeltaAIC
from ModelSelectionInfluence import ModelSelectionInfluence
from ParaTuner import tenure_trials
from matplotlib import pyplot as plt
import numpy as np

file_path = r".\baseball.dat"
save_path = r".\Results"

# Parameters
max_iteration = 100
seed = 1234
tenure = 5
c = 5
threshold_aic = 1
threshold_r2 = 0.01


def report(history_solutions, history_aic, info):
    """
    Generate the reports about the result of model selection problem \n
    Args: \n
        history_solutions: List of solutions \n
        history_aic: List of solutions' AIC \n
        info: Model name (str)
    """
    print("\n")
    print("-" * 50)
    print("The Report of " + info)
    print("\n")    
    print("The min AIC achieved is: " + str(min(history_aic)))
    min_index = min(range(len(history_aic)), key = history_aic.__getitem__)
    print("The corresponding iteration is: " + str(min_index))
    print("The corresponding solution is: ")
    print(history_solutions[min_index])
    print("Plot AIC at each iteration...")
    plt.plot(np.array([i for i in range(max_iteration + 1)]), history_aic)
    plt.xlabel("Iteration")
    plt.ylabel("AIC")
    plt.title(info)
    plt.show()
    print("Done. This is the end of the report.")
    print("-" * 50)
    print("\n")
    

if __name__ == '__main__':
    # a.
    tenure_list = [i for i in range(1, 21)]
    tenure_result = tenure_trials(tenure_list, max_iteration, file_path, seed)
    tenure_result.to_csv(save_path + r"\tenureResult.csv", index = False)
    
    # Compare short-term memory and long-term memory
    model_short = ModelSelectionStandard(tenure, max_iteration, file_path, seed)
    history_solutions_short, history_aic_short = model_short.tabu_search()
    model_long = ModelSelectionLong(tenure, max_iteration, file_path, c, seed)
    history_solutions_long, history_aic_long = model_long.tabu_search()
    report(history_solutions_short, history_aic_short, "Short-term memory")
    report(history_solutions_long, history_aic_long, "Long-term memory")
    
    # b.
    model_b = ModelSelectionDeltaAIC(tenure, max_iteration, file_path, threshold_aic, seed)
    history_solutions_b, history_aic_b = model_b.tabu_search()
    report(history_solutions_b, history_aic_b, "Delta AIC as attribute")
    
    # c.
    model_c = ModelSelectionInfluence(tenure, max_iteration, file_path, threshold_r2, seed)
    history_solutions_c, history_aic_c = model_c.tabu_search()
    report(history_solutions_c, history_aic_c, "Aspiration by inﬂuence")
