# -*- coding: utf-8 -*-

import sympy as sp
import numpy as np


class Standard():
    """
    Implement standard EM in peppered moth analysis \n
    Args:
        initial_params: (p_C^0, p_I^0) \n
        epsilon: Threshold of the stopping rule \n
        max_iter: Max iteration. Default: 100
    """
    
    def __init__(self, initial_params, epsilon, max_iter = 100):
        self.p_C = initial_params[0]
        self.p_I = initial_params[1]
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.n_C = 85
        self.n_I = 196
        self.n_T = 341
        self.n_U = 578
        
    def basic_relations(self):
        """
        Define p_T, n, n_CC, n_CI, n_CT, n_II, n_IT, n_TT
        """
        # Define symbols
        n_C, n_I, n_T, n_U = sp.symbols("n_C, n_I, n_T, n_U")
        p_C, p_I = sp.symbols("p_C, p_I")
        
        # Formulas
        p_T = 1 - p_C - p_I
        n = n_C + n_I + n_T + n_U
        n_CC = (n_C * (p_C ** 2)) / ((p_C ** 2) + 2 * p_C * p_I + 2 * p_C * p_T)
        n_CI = (2 * n_C * p_C * p_I) / ((p_C ** 2) + 2 * p_C * p_I + 2 * p_C * p_T)
        n_CT = (2 * n_C * p_C * p_T) / ((p_C ** 2) + 2 * p_C * p_I + 2 * p_C * p_T)
        n_II = (n_I * (p_I ** 2)) / ((p_I ** 2) + 2 * p_I * p_T) + (n_U * (p_I ** 2)) / ((p_I ** 2) + 2 * p_I * p_T + (p_T ** 2))
        n_IT = (2 * n_I * p_I * p_T) / ((p_I ** 2) + 2 * p_I * p_T) + (2 * n_U * p_I * p_T) / ((p_I ** 2) + 2 * p_I * p_T + (p_T ** 2))
        n_TT = n_T + (n_U * (p_T ** 2)) / ((p_I ** 2) + 2 * p_I * p_T + (p_T ** 2))
        
        return p_T, n, n_CC, n_CI, n_CT, n_II, n_IT, n_TT 
    
    def symbol_equation(self):
        """
        Define equations (symbols) \n
        Returns:
            sympy formulas, (p_C, p_I)
        """
        # Define symbols
        n_C, n_I, n_T, n_U = sp.symbols("n_C, n_I, n_T, n_U")
        p_C, p_I = sp.symbols("p_C, p_I")
        
        # Formulas
        p_T, n, n_CC, n_CI, n_CT, n_II, n_IT, n_TT = self.basic_relations()
        p_C_up = (2 * n_CC + n_CI + n_CT) / (2 * n)
        p_I_up = (2 * n_II + n_IT + n_CI) / (2 * n)
        
        return p_C_up, p_I_up
    
    def Q_equation(self):
        """
        Define Q(p|p^{(t)})
        """
        p_C, p_I = sp.symbols("p_C, p_I")
        
        # Formulas
        p_T, n, n_CC, n_CI, n_CT, n_II, n_IT, n_TT = self.basic_relations()
        Q = n_CC * sp.log(p_C ** 2) + n_CI * sp.log(2 * p_C * p_I) + n_CT * sp.log(2 * p_C * p_T) 
        + n_II * sp.log(p_I ** 2) + n_IT * sp.log(2 * p_I * p_T) + n_TT * sp.log(p_T ** 2)
        return Q
    
    def Q_prime(self):
        """
        Calculate Q'(p|p^{(t)})
        """
        p_C, p_I = sp.symbols("p_C, p_I")        
        Q = self.Q_equation()
        
        d_p_C = sp.diff(Q, p_C)
        d_p_I = sp.diff(Q, p_I)
        
        return [d_p_C, d_p_I]
    
    def Q_2prime(self):
        """
        Calculate Q''(p|p^{(t)})
        """
        p_C, p_I = sp.symbols("p_C, p_I")
        d_Q = self.Q_prime()
        
        d_p_C_p_C = sp.diff(d_Q[0], p_C)
        d_p_C_p_I = sp.diff(d_Q[0], p_I)
        d_p_I_p_I = sp.diff(d_Q[1], p_I)
        
        return [[d_p_C_p_C, d_p_C_p_I],
                [d_p_C_p_I, d_p_I_p_I]]
    
    def L_equation(self):
        """
        Define L(p^{(t)}|x)
        """
        p_C, p_I = sp.symbols("p_C, p_I")
        n_C, n_I, n_T, n_U = sp.symbols("n_C, n_I, n_T, n_U")
        
        # Formulas
        p_T = 1 - p_C - p_I
        L = n_C * sp.log(p_C ** 2 + 2 * p_C * p_I + 2 * p_C * p_T) + n_I * sp.log(p_I ** 2 + 2 * p_I * p_T) + n_T * sp.log(p_T ** 2) + n_U * sp.log(p_I ** 2 + 2 * p_I * p_T + p_T ** 2)
        return L
    
    def L_prime(self):
        """
        Calculate L'(p^{(t)}|x)
        """
        p_C, p_I = sp.symbols("p_C, p_I")        
        L = self.L_equation()
        
        d_p_C = sp.diff(L, p_C)
        d_p_I = sp.diff(L, p_I)
        
        return [d_p_C, d_p_I]        
    
    def L_2prime(self):
        """
        Calculate L''(p^{(t)}|x)
        """
        p_C, p_I = sp.symbols("p_C, p_I")
        d_L = self.L_prime()
        
        d_p_C_p_C = sp.diff(d_L[0], p_C)
        d_p_C_p_I = sp.diff(d_L[0], p_I)
        d_p_I_p_I = sp.diff(d_L[1], p_I)
        
        return [[d_p_C_p_C, d_p_C_p_I],
                [d_p_C_p_I, d_p_I_p_I]]  
    
    def update_equation(self):
        """
        Define unpdate equations \n
        Return:
            sympy formulas (numpy api), (p_C, p_I)
        """
        # Define symbols
        n_C, n_I, n_T, n_U = sp.symbols("n_C, n_I, n_T, n_U")
        p_C, p_I = sp.symbols("p_C, p_I")
        p_C_up, p_I_up = self.symbol_equation()
        
        # To numpy
        p_C_np = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), p_C_up, "numpy")
        p_I_np = sp.lambdify((n_C, n_I, n_T, n_U, p_C, p_I), p_I_up, "numpy")
        
        return p_C_np, p_I_np
    
    def stopping_rule(self, p_C, p_I, p_C_t_sub_1, p_I_t_sub_1):
        """
        Calculate R^{(t)}
        """
        t = np.array([p_C, p_I], dtype = np.float64)
        t_sub_1 = np.array([p_C_t_sub_1, p_I_t_sub_1], dtype = np.float64)
        return np.sqrt(np.sum((t - t_sub_1) ** 2)) / np.sqrt(np.sum(t_sub_1 ** 2))
        
    def em(self):
        """
        Standard EM \n
        Return:
            p_C_history, p_I_history, R_history, iteration
        """
        # Initiate
        p_C = self.p_C
        p_I = self.p_I
        p_C_t_sub_1 = p_C - self.epsilon - 1
        p_I_t_sub_1 = p_I - self.epsilon - 1
        p_C_np, p_I_np = self.update_equation()
        iteration = 0
        R = self.stopping_rule(p_C, p_I, p_C_t_sub_1, p_I_t_sub_1)
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
            p_C = p_C_np(self.n_C, self.n_I, self.n_T, self.n_U, p_C_t_sub_1, p_I_t_sub_1)
            p_I = p_I_np(self.n_C, self.n_I, self.n_T, self.n_U, p_C_t_sub_1, p_I_t_sub_1)
            R = self.stopping_rule(p_C, p_I, p_C_t_sub_1, p_I_t_sub_1)
            p_C_history.append(p_C)
            p_I_history.append(p_I)
            R_history.append(R)            
            iteration += 1
        
        return p_C_history, p_I_history, R_history, iteration