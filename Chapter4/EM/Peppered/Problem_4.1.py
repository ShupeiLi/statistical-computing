# -*- coding: utf-8 -*-

import sys
sys.path.append(r".\Chapter4\EM\Peppered")
from Standard import Standard
from SEM import SEM
from Bootstrapping import Bootstrapping
from Gradient import Gradient
from Aitken import Aitken
import numpy as np

initial_params = (0.3, 0.3) 
epsilon = 10 ** (-6)
sem_epsilon = 10 ** (-6)
epoch = 100
max_iter = 100


if __name__ == '__main__':
    # b.
    model = Standard(initial_params, epsilon)
    p_C_history, p_I_history, R_history, iteration = model.em()
    
    # c.
    model = SEM(initial_params, epsilon, sem_epsilon)
    r_history = model.sem()
    psi_prime = r_history[6]
    var_hat = model.var_estimation(psi_prime)
    sd_hat = np.sqrt(np.diag(var_hat))
    print("c. Estimated standard errors (p_C, p_I): " + str(sd_hat))
    cor_hat = (1 / np.tile(np.expand_dims(sd_hat, axis = 1), 2)) * var_hat * (1 / np.transpose(np.tile(np.expand_dims(sd_hat, axis = 1), 2)))
    
    # d.
    model = Bootstrapping(initial_params, epsilon, epoch)
    theta_history = model.em_bootstrap()
    print("d. Estimated standard errors of p_C: " + str(np.std(theta_history[:, 0])))
    print("d. Estimated standard errors of p_I: " + str(np.std(theta_history[:, 1])))
    print("d. Estimated correlation matrix:\n" + str(np.corrcoef(theta_history[:, 0], theta_history[:, 1])))
    
    # e.
    model = Gradient(initial_params, epsilon, 0.5, max_iter, True, False)
    p_C_history, p_I_history, R_history, iteration = model.em_gradient()
    
    # f.
    model = Aitken(initial_params, epsilon, 0.5, max_iter)
    p_C_history, p_I_history, R_history, iteration = model.em_aitken()