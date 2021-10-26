# -*- coding: utf-8 -*-

from ModelSelection import ModelSelection
import matplotlib.pyplot as plt
import numpy as np

file_path = r".\baseball.dat"
max_iteration = 14 
ks = [1, 2] 
seeds = [4, 6, 8, 20, 1234]

if __name__ == '__main__':
    for seed in seeds:
        for k in ks:
            model = ModelSelection(max_iteration, k, file_path, seed)
            history_solutions, history_aic = model.local_search()
            print("\n")
            print("-" * 50)
            print("k: {}, seed: {}".format(k, seed))
            min_index = min(range(len(history_aic)), key = history_aic.__getitem__)
            print("Best solution: " + str(history_solutions[min_index]))
            print("The corresponding iteration is: " + str(min_index))
            print("Best AIC: " + str(min(history_aic)))
            print("-" * 50)
            plt.plot(np.array([i for i in range(max_iteration + 1)]), history_aic)
        plt.xlabel("Iteration")
        plt.ylabel("AIC")
        plt.title("Seed: {}".format(seed))
        plt.show()