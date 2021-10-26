# -*- coding: utf-8 -*-

from ModelSelection import ModelSelection
import matplotlib.pyplot as plt
import numpy as np

file_path = r".\baseball.dat"

max_iteration = 100 
population = 20 
seed = 1234 
strata = 1
fit_choice = "13"

def report(history_generations, history_aic):
    """
    Generate reports
    """
    history_aic_plt = np.array(history_aic)
    history_best_aic = np.min(history_aic_plt)
    history_best_iteration = sorted(list(set(np.where(history_aic_plt == history_aic_plt.min())[0])))
    print("\n")
    print("-" * 50)
    print("Best AIC: " + str(history_best_aic))
    print("The corresponding iteration is: " + str(history_best_iteration))
    print("-" * 50)
    iterations = np.tile(np.expand_dims(np.arange(history_aic_plt.shape[0]), axis = 1), (1, history_aic_plt.shape[1])).reshape(history_aic_plt.shape[0] * history_aic_plt.shape[1])
    history_aic_plt = history_aic_plt.reshape(history_aic_plt.shape[0] * history_aic_plt.shape[1])
    plt.scatter(iterations, history_aic_plt, s = 1, marker = '.')
    plt.xlabel("Iteration")
    plt.ylabel("AIC")
    plt.show()
    
if __name__ == '__main__':
    # b.
    model = ModelSelection(max_iteration, population, file_path, seed = seed, strata = strata, 
                           fit_choice = fit_choice, z = 1.5, mutation_rate = 0.01)
    # trials:  generation sizes = {10, 20, 50}
    model.population = 10
    history_generations, history_aic = model.genetic_algorithm()
    report(history_generations, history_aic)
    model.population = 50
    history_generations, history_aic = model.genetic_algorithm()
    report(history_generations, history_aic)
    model.population = population
    history_generations, history_aic = model.genetic_algorithm()
    report(history_generations, history_aic)
    
    # c.1
    model.fit_choice = "12"
    history_generations, history_aic = model.genetic_algorithm()
    report(history_generations, history_aic)
    
    # c.2
    model.fit_choice = "22"
    history_generations, history_aic = model.genetic_algorithm()
    report(history_generations, history_aic)
    
    # c.3
    model.fit_choice = fit_choice
    model.strata = 4
    history_generations, history_aic = model.genetic_algorithm()
    report(history_generations, history_aic)
    
    model.strata = 10
    history_generations, history_aic = model.genetic_algorithm()
    report(history_generations, history_aic)
