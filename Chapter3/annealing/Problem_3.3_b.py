# -*- coding: utf-8 -*-

from ModelSelection import ModelSelection
import matplotlib.pyplot as plt
import numpy as np

file_path = r".\baseball.dat"
ks = [2, 3]
seed = 1234
taus = [1, 6]
max_iteration = np.sum(np.array([60, 60, 60, 60, 60, 120, 120, 120, 120, 120, 
                                 220, 220, 220, 220, 220]))

for tau in taus:
    for k in ks:
        model = ModelSelection(k, tau, file_path, seed)
        history_solutions, history_aic = model.annealing()
        print("\n")
        print("-" * 50)
        print("k: {}, tao: {}".format(k, tau))
        min_index = min(range(len(history_aic)), key = history_aic.__getitem__)
        print("Best solution: " + str(history_solutions[min_index]))
        print("The corresponding iteration is: " + str(min_index))
        print("Best AIC: " + str(min(history_aic)))
        print("-" * 50)
        plt.plot(np.array([i for i in range(max_iteration + 1)]), history_aic)
    plt.xlabel("Iteration")
    plt.ylabel("AIC")
    plt.title("Tau_0: {}".format(tau))
    plt.show()