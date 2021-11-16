# -*- coding: utf-8 -*-

import sys
sys.path.append(r".\Chapter4\EM\Peppered")
from Standard import Standard
import sympy as sp
import numpy as np


class Aitken(Standard):
    """
    Implement Aitken accelerated EM Algorithm in peppered moth analysis \n
    Args:
        initial_params: (p_C^0, p_I^0) \n
        epsilon: Threshold of the stopping rule \n
        beta: Rate of backtracking. Default: 0.5 \n
        max_iter: Max iteration. Default: 100 \n
        back: Backtracking. Default: False
    """
    
    def __init__(self, initial_params, epsilon, beta = 0.5, max_iter = 100, back = False):
        super().__init__(initial_params, epsilon, max_iter)
        self.beta = beta
        self.back = back
        
    def update_equation(self, theta, theta_em):
        """
        Define update equation in Aitken Acceleration
        """
        n_C, n_I, n_T, n_U = sp.symbols("n_C, n_I, n_T, n_U")
        p_C, p_I = sp.symbols("p_C, p_I")
        
        # d^2L
        dd_L_eq = super().L_2prime()
        d_p_C_p_C_L = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), dd_L_eq[0][0], "numpy")
        d_p_C_p_I_L = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), dd_L_eq[0][1], "numpy")
        d_p_I_p_I_L = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), dd_L_eq[1][1], "numpy")
        dd_L = np.array([[d_p_C_p_C_L(self.n_C, self.n_I, self.n_T, self.n_U, theta[0], theta[1]),
                          d_p_C_p_I_L(self.n_C, self.n_I, self.n_T, self.n_U, theta[0], theta[1])],
                         [d_p_C_p_I_L(self.n_C, self.n_I, self.n_T, self.n_U, theta[0], theta[1]),
                          d_p_I_p_I_L(self.n_C, self.n_I, self.n_T, self.n_U, theta[0], theta[1])]])
        
        # d^2Q
        dd_Q_eq = super().Q_2prime()
        d_p_C_p_C_Q = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), dd_Q_eq[0][0], "numpy")
        d_p_C_p_I_Q = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), dd_Q_eq[0][1], "numpy")
        d_p_I_p_I_Q = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), dd_Q_eq[1][1], "numpy")
        dd_Q = np.array([[d_p_C_p_C_Q(self.n_C, self.n_I, self.n_T, self.n_U, theta[0], theta[1]),
                          d_p_C_p_I_Q(self.n_C, self.n_I, self.n_T, self.n_U, theta[0], theta[1])],
                         [d_p_C_p_I_Q(self.n_C, self.n_I, self.n_T, self.n_U, theta[0], theta[1]),
                          d_p_I_p_I_Q(self.n_C, self.n_I, self.n_T, self.n_U, theta[0], theta[1])]])
        
        return np.matmul(np.matmul(np.linalg.inv(dd_L), dd_Q), (theta_em - theta))
        
    def Q_eval(self):
        n_C, n_I, n_T, n_U = sp.symbols("n_C, n_I, n_T, n_U")
        p_C, p_I = sp.symbols("p_C, p_I")
        Q_eq = super().Q_equation()
        return sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), Q_eq, "numpy")
    
    def em_aitken(self):
        """
        Aitken accelerated EM \n
        Return:
            p_C_history, p_I_history, R_history, iteration
        """
        # Initiate
        p_C = self.p_C
        p_I = self.p_I
        p_C_t_sub_1 = p_C - self.epsilon - 1
        p_I_t_sub_1 = p_I - self.epsilon - 1
        iteration = 0
        R = super().stopping_rule(p_C, p_I, p_C_t_sub_1, p_I_t_sub_1)
        p_C_np, p_I_np = super().update_equation()
        Q = self.Q_eval()
        p_C_history = []
        p_I_history = []
        R_history = []
        p_C_history.append(p_C)
        p_I_history.append(p_I)
        R_history.append(R)
        
        # Main logic
        while R > self.epsilon and iteration < self.max_iter:
            p_C_t_sub_1 = p_C
            p_I_t_sub_1 = p_I            
            p_C_em = p_C_np(self.n_C, self.n_I, self.n_T, self.n_U, p_C_t_sub_1, p_I_t_sub_1)
            p_I_em = p_I_np(self.n_C, self.n_I, self.n_T, self.n_U, p_C_t_sub_1, p_I_t_sub_1)
            
            t = 1
            theta = np.array([p_C_t_sub_1, p_I_t_sub_1]) + t * self.update_equation(np.array([p_C_t_sub_1, p_I_t_sub_1]), np.array([p_C_em, p_I_em]))
            if self.back:
                while (theta[0] <= 0) or (theta[1] <= 0) or (theta[0] >= 1) or (theta[1] 
                        >= 1) or (Q(self.n_C, self.n_I, self.n_T, self.n_U, theta[0], theta[1]) < 
                        Q(self.n_C, self.n_I, self.n_T, self.n_U, p_C_t_sub_1, p_I_t_sub_1)):
                    t = self.beta * t
                    theta = np.array([p_C_t_sub_1, p_I_t_sub_1]) + t * self.update_equation(np.array([p_C_t_sub_1, p_I_t_sub_1]), np.array([p_C_em, p_I_em]))
            
            p_C = theta[0]
            p_I = theta[1]
            R = self.stopping_rule(p_C, p_I, p_C_t_sub_1, p_I_t_sub_1)
            p_C_history.append(p_C)
            p_I_history.append(p_I)
            R_history.append(R)            
            iteration += 1
        
        return p_C_history, p_I_history, R_history, iteration