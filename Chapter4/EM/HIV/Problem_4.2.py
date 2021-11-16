# -*- coding: utf-8 -*-

import sys
sys.path.append(r".\Chapter4\EM\HIV")
from Standard import Standard
from Bootstrapping import Bootstrapping
import numpy as np


# params
alpha = 0.3
beta = 0.2
mu = 2
Lambda = 3
initial_params = (alpha, beta, mu, Lambda)
epsilon = 10 ** (-6)
max_iter = 1000
epoch = 100


if __name__ == '__main__':
    # b.
    model = Standard(initial_params, epsilon, max_iter)
    alpha_history, beta_history, mu_history, lambda_history, R_history, iteration = model.em()
    
    # c.
    model = Bootstrapping(initial_params, epsilon, epoch, max_iter)
    theta_history = model.em_bootstrap()
    print("c. Estimated standard errors:\n" + str(np.sqrt(np.diag(np.cov(np.transpose(theta_history))))))
    print("c. Estimated correlation:\n" + str(np.corrcoef(np.transpose(theta_history))))