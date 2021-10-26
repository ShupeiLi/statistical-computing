# -*- coding: utf-8 -*-

from GeneticMapping import GeneticMapping
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from tqdm import tqdm

file_path = r".\geneticmapping.dat"
max_iteration = 200
n = 100

def seed_generator(n):
    """
    Generate random seeds \n
    
    Args:
        n: Sample size
    """
    return [randint(0, 100000) for i in range(n)]
    
def report(seed, history_solutions, history_distances, history_likelihood):
    """
    Generate the report
    """
    print("\n")
    print("-" * 50)
    print("seed: {}".format(seed))
    max_index = max(range(len(history_likelihood)), key = history_likelihood.__getitem__)
    print("Best solution: " + str(history_solutions[max_index]))
    print("Best distances: " + str(history_distances[max_index]))
    print("The corresponding iteration is: " + str(max_index))
    print("Best likelihood value: " + str(max(history_likelihood)))
    print("-" * 50)
    plt.plot(np.array([i for i in range(max_iteration + 1)]), history_likelihood)
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Seed: {}".format(seed))
    plt.show()

if __name__ == '__main__':
    # a.
    seeds = seed_generator(n)
    best_lst = []
    best_likelihood = -10000
    
    for i in tqdm(range(n)):
        seed = seeds[i]
        model = GeneticMapping(max_iteration, file_path, seed)
        history_solutions, history_distances, history_likelihood = model.local_search()
        if max(history_likelihood) > best_likelihood:
            best_likelihood = max(history_likelihood)
            best_lst = [seed, history_solutions, history_distances, history_likelihood]
    report(best_lst[0], best_lst[1], best_lst[2], best_lst[3])
        
    # Improved search strategy
    seed = 25676
    model = GeneticMapping(max_iteration, file_path, seed, optim = True)
    history_solutions, history_distances, history_likelihood = model.local_search()
    report(seed, history_solutions, history_distances, history_likelihood)
    
    # b.
    for i in tqdm(range(n)):
        seed = seeds[i]
        model = GeneticMapping(max_iteration, file_path, seed, steepest = True, optim = True)
        history_solutions, history_distances, history_likelihood = model.local_search()
        if max(history_likelihood) > best_likelihood:
            best_likelihood = max(history_likelihood)
            best_lst = [seed, history_solutions, history_distances, history_likelihood]
    report(best_lst[0], best_lst[1], best_lst[2], best_lst[3])