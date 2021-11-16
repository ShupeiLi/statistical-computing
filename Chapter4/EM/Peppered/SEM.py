# -*- coding: utf-8 -*-

import sys
sys.path.append(r".\Chapter4\EM\Peppered")
from Standard import Standard
import numpy as np
import sympy as sp


class SEM(Standard):
    """
    Implement SEM in peppered moth analysis \n
    Args:
        initial_params: (p_C^0, p_I^0) \n
        epsilon: Threshold of the stopping rule in EM \n
        sem_epsilon: Threshold of the stopping rule in r estimation \n
        max_iter: Max iteration. Default: 100
    """
    
    def __init__(self, initial_params, epsilon, sem_epsilon, max_iter = 100):
        super().__init__(initial_params, epsilon, max_iter)
        self.sem_epsilon = sem_epsilon
        
    def r_estimation(self, current_array, em_array, params_equation):
        """
        Calculate r_ij. (p_C, p_I)
        """
        r = np.zeros((current_array.shape[0], current_array.shape[0]))
        for j in range(current_array.shape[0]):
            theta_j = em_array.copy()
            theta_j[j] = current_array[j]
            for i in range(current_array.shape[0]):
                psi = params_equation[i]
                r[i, j] = (psi(self.n_C, self.n_I, self.n_T, self.n_U, theta_j[0], theta_j[1])
                           - em_array[i]) / (theta_j[j] - em_array[j])
        return r

    def r_stopping_rule(self, r_previous, r_current):
        """
        Relative stopping rule: delta r
        """
        return np.sqrt(np.sum((r_current - r_previous) ** 2)) / np.sqrt(np.sum((r_previous) ** 2))

    def sem(self):
        """
        SEM
        Return:
            r_history
        """
        # Implement standard EM
        p_C_history, p_I_history, _, _ = super().em()
        p_C_em = p_C_history[-1]
        p_I_em = p_I_history[-1]
        p_hat = np.array([p_C_em, p_I_em])
        self.p_hat = p_hat
        
        # Initiate
        p_C = self.p_C
        p_I = self.p_I
        p_C_t_sub_1 = p_C - self.sem_epsilon - 1
        p_I_t_sub_1 = p_I - self.sem_epsilon - 1
        p_C_np, p_I_np = self.update_equation()
        
        r_previous = self.r_estimation(np.array([p_C_t_sub_1, p_I_t_sub_1]), p_hat, [p_C_np, p_I_np])
        r_current = self.r_estimation(np.array([p_C, p_I]), p_hat, [p_C_np, p_I_np])
        r_history = []
        r_history.append(r_current)

        # Main logic
        while self.r_stopping_rule(r_previous, r_current) > self.sem_epsilon:
            p_C_t_sub_1 = p_C
            p_I_t_sub_1 = p_I
            r_previous = r_current
            
            p_C = p_C_np(self.n_C, self.n_I, self.n_T, self.n_U, p_C_t_sub_1, p_I_t_sub_1)
            p_I = p_I_np(self.n_C, self.n_I, self.n_T, self.n_U, p_C_t_sub_1, p_I_t_sub_1)            
            r_current = self.r_estimation(np.array([p_C, p_I]), p_hat, [p_C_np, p_I_np])        
            r_history.append(r_current)        

        return r_history
    
    def var_estimation(self, psi_prime):
        """
        Calculate hat{var}
        """
        # hat{i}_Y
        n_C, n_I, n_T, n_U = sp.symbols("n_C, n_I, n_T, n_U")
        p_C, p_I = sp.symbols("p_C, p_I")
        Q_2prime_eq = super().Q_2prime()
        d_p_C_p_C = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), - Q_2prime_eq[0][0], "numpy")
        d_p_C_p_I = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), - Q_2prime_eq[0][1], "numpy")
        d_p_I_p_I = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), - Q_2prime_eq[1][1], "numpy")
        matrix_i_Y = np.array([[d_p_C_p_C(self.n_C, self.n_I, self.n_T, self.n_U, self.p_hat[0], self.p_hat[1]),
                                d_p_C_p_I(self.n_C, self.n_I, self.n_T, self.n_U, self.p_hat[0], self.p_hat[1])],
                               [d_p_C_p_I(self.n_C, self.n_I, self.n_T, self.n_U, self.p_hat[0], self.p_hat[1]),
                                d_p_I_p_I(self.n_C, self.n_I, self.n_T, self.n_U, self.p_hat[0], self.p_hat[1])]])

        # hat{var}
        matrix_I = np.identity(self.p_hat.shape[0], dtype = np.float64)
        matrix_x_inv = matrix_I + np.matmul(np.transpose(psi_prime), np.linalg.inv(matrix_I - np.transpose(psi_prime)))
        return np.matmul(np.linalg.inv(matrix_i_Y), matrix_x_inv)