# -*- coding: utf-8 -*-

import sys
sys.path.append(r".\Chapter4\EM\Peppered")
from Standard import Standard
import numpy as np
import sympy as sp
from Funs import Q_prime_m, Q_2prime_m, Q_m
import scipy.optimize as optimize


class Gradient(Standard):
    """
    Implement EM Gradient Algorithm in peppered moth analysis \n
    Args:
        initial_params: (p_C^0, p_I^0) \n
        epsilon: Threshold of the stopping rule \n
        beta: Rate of backtracking. Default: 0.5 \n
        max_iter: Max iteration. Default: 100 \n
        sym: Use sympy. Default: False \n
        auto: Automatically optimize parameters. Default: True
    """
    
    def __init__(self, initial_params, epsilon, beta = 0.5, max_iter = 100, sym = False, auto = True):
        super().__init__(initial_params, epsilon, max_iter)
        self.beta = beta
        self.sym = sym 
        self.auto = auto
        
    def update_equation(self, p_hat):
        """
        Define the update equation
        """
        # Define symbols
        n_C, n_I, n_T, n_U = sp.symbols("n_C, n_I, n_T, n_U")
        p_C, p_I = sp.symbols("p_C, p_I")
        
        # dQ
        d_Q_eq = super().Q_prime()
        d_p_C = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), d_Q_eq[0], "numpy")
        d_p_I = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), d_Q_eq[1], "numpy")
        d_Q = np.array([d_p_C(self.n_C, self.n_I, self.n_T, self.n_U, p_hat[0], p_hat[1]),
                        d_p_I(self.n_C, self.n_I, self.n_T, self.n_U, p_hat[0], p_hat[1])])
        
        # d^2Q
        dd_Q_eq = super().Q_2prime()
        d_p_C_p_C = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), dd_Q_eq[0][0], "numpy")
        d_p_C_p_I = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), dd_Q_eq[0][1], "numpy")
        d_p_I_p_I = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), dd_Q_eq[1][1], "numpy")
        dd_Q = np.array([[d_p_C_p_C(self.n_C, self.n_I, self.n_T, self.n_U, p_hat[0], p_hat[1]),
                          d_p_C_p_I(self.n_C, self.n_I, self.n_T, self.n_U, p_hat[0], p_hat[1])],
                         [d_p_C_p_I(self.n_C, self.n_I, self.n_T, self.n_U, p_hat[0], p_hat[1]),
                          d_p_I_p_I(self.n_C, self.n_I, self.n_T, self.n_U, p_hat[0], p_hat[1])]])
        
        return np.matmul(np.linalg.inv(dd_Q), d_Q)
    
    def update_equation_m(self, p_hat):
        """
        Define the update equation (manually)
        """
        d_Q = Q_prime_m(self.n_C, self.n_I, self.n_T, self.n_U, p_hat[0], p_hat[1])
        dd_Q = Q_2prime_m(self.n_C, self.n_I, self.n_T, self.n_U, p_hat[0], p_hat[1])
        
        return np.matmul(np.linalg.inv(dd_Q), d_Q)
    
    def obj_fun(self, params):
        """
        Q(p|p^{(t)})
        """
        p_C, p_I = params
        p_T = 1 - p_C - p_I
        n_CC = (self.n_C * (p_C ** 2)) / ((p_C ** 2) + 2 * p_C * p_I + 2 * p_C * p_T)
        n_CI = (2 * self.n_C * p_C * p_I) / ((p_C ** 2) + 2 * p_C * p_I + 2 * p_C * p_T)
        n_CT = (2 * self.n_C * p_C * p_T) / ((p_C ** 2) + 2 * p_C * p_I + 2 * p_C * p_T)
        n_II = (self.n_I * (p_I ** 2)) / ((p_I ** 2) + 2 * p_I * p_T) + (self.n_U * (p_I ** 2)) / ((p_I ** 2) + 2 * p_I * p_T + (p_T ** 2))
        n_IT = (2 * self.n_I * p_I * p_T) / ((p_I ** 2) + 2 * p_I * p_T) + (2 * self.n_U * p_I * p_T) / ((p_I ** 2) + 2 * p_I * p_T + (p_T ** 2))
        n_TT = self.n_T + (self.n_U * (p_T ** 2)) / ((p_I ** 2) + 2 * p_I * p_T + (p_T ** 2))
        
        return n_CC * np.log(p_C ** 2) + n_CI * np.log(2 * p_C * p_I) + n_CT * np.log(2 * p_C * p_T) + n_II * np.log(p_I ** 2) + n_IT * np.log(2 * p_I * p_T) + n_TT * np.log(p_T ** 2)
        
    def em_gradient(self):
        """
        EM Gradient Algorithm \n
        Return:
            p_C_history, p_I_history, R_history, iteration        
        """
        # Initiate
        p_C_t = self.p_C
        p_I_t = self.p_I
        p_C_t_sub_1 = p_C_t - self.epsilon - 1
        p_I_t_sub_1 = p_I_t - self.epsilon - 1
        iteration = 0
        R = super().stopping_rule(p_C_t, p_I_t, p_C_t_sub_1, p_I_t_sub_1)
        p_C_history = []
        p_I_history = []
        R_history = []
        p_C_history.append(p_C_t)
        p_I_history.append(p_I_t)
        R_history.append(R)
        
        # Main logic
        while R > self.epsilon and iteration < self.max_iter:
            p_C_t_sub_1 = p_C_t
            p_I_t_sub_1 = p_I_t
            
            t = 1
            if self.auto:
                res = optimize.minimize(self.obj_fun, [p_C_t_sub_1, p_I_t_sub_1], method = "Nelder-Mead")
                p_C_t = res.x[0]
                p_I_t = res.x[1]
            else:
                if self.sym:
                    n_C, n_I, n_T, n_U = sp.symbols("n_C, n_I, n_T, n_U")
                    p_C, p_I = sp.symbols("p_C, p_I")
                    Q_eq = super().Q_equation()
                    Q = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), Q_eq, "numpy")
                    
                    delta_term = self.update_equation([p_C_t_sub_1, p_I_t_sub_1])
                    theta_t = np.array([p_C_t_sub_1, p_I_t_sub_1]) - t * delta_term
                    while (theta_t[0] <= 0) or (theta_t[1] <= 0) or (theta_t[0] >= 1) or (theta_t[1] 
                           >= 1) or (Q(self.n_C, self.n_I, self.n_T, self.n_U, theta_t[0], theta_t[1]) < 
                           Q(self.n_C, self.n_I, self.n_T, self.n_U, p_C_t_sub_1, p_I_t_sub_1)):
                        t = self.beta * t
                        theta_t = np.array([p_C_t_sub_1, p_I_t_sub_1]) - t * delta_term
                else:
                    delta_term = self.update_equation_m([p_C_t_sub_1, p_I_t_sub_1])
                    theta_t = np.array([p_C_t_sub_1, p_I_t_sub_1]) - t * delta_term
                    while (theta_t[0] <= 0) or (theta_t[1] <= 0) or (theta_t[0] >= 1) or (theta_t[1] 
                           >= 1) or (Q_m(self.n_C, self.n_I, self.n_T, self.n_U, theta_t[0], theta_t[1]) < 
                           Q_m(self.n_C, self.n_I, self.n_T, self.n_U, p_C_t_sub_1, p_I_t_sub_1)):
                        t = self.beta * t
                        theta_t = np.array([p_C_t_sub_1, p_I_t_sub_1]) - t * delta_term
            
                p_C_t = theta_t[0]
                p_I_t = theta_t[1]
                
            R = self.stopping_rule(p_C_t, p_I_t, p_C_t_sub_1, p_I_t_sub_1)
            p_C_history.append(p_C_t)
            p_I_history.append(p_I_t)
            R_history.append(R)                    
            iteration += 1
            
        return p_C_history, p_I_history, R_history, iteration