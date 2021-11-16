# -*- coding: utf-8 -*-

import sys
sys.path.append(r".\Chapter4\EM\Gear")
from Funs import z_MC, d_b, dd_b
import numpy as np
import scipy.optimize as optimize


class FamilyEM():
    """
    MCEM + ECM in gear couplings problem \n
    Args:
        a, b: Initial parameters \n
        epsilon: Stopping rule in ECM. \n
        m: para in MCEM. Default: 20 \n
        max_iter: Max iterations. Default: 100 \n
        auto: Automatically optimize parameters. Default: True \n
        skip_Q: Direct maximization of the observed-data likelihood. Default: False
    """
    
    def __init__(self, a, b, epsilon, m = 20, max_iter = 100, auto = True, skip_Q = False):
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.x = np.array([5.50, 4.54, 2.14, 10.24, 4.56, 9.42, 5.64])
        self.censored = np.array([6.94, 3.65, 3.40, 4.38, 4.55, 4.15, 10.23])
        self.m = m
        self.max_iter = max_iter
        self.auto = True
        self.skip_Q = skip_Q
        
    def mcem(self, a, b, iteration):
        """
        E-Step: MCEM
        """
        sample = []
        for i in range(self.m + iteration * 10):
            sample.append(z_MC(a, b, self.censored))
        z = np.transpose(np.array(sample))
        x = np.tile(np.expand_dims(self.x, axis = 1), (1, z.shape[1]))
        return np.concatenate((x, z), axis = 0)
        
    def ecm(self, a, b, y_array):
        """
        M-Step: ECM
        """
        # Optimize a
        iteration_a = 0
        a_t_sub_1 = a - self.epsilon - 1
        while iteration_a < self.max_iter and self.stopping_rule(a, a_t_sub_1) > self.epsilon:
            a_t_sub_1 = a
            a = 2 * a_t_sub_1 - ((a_t_sub_1 ** 2) * np.sum(y_array ** b)) / (14 * y_array.shape[1])
            iteration_a += 1
    
        # Optimize b
        iteration_b = 0
        b_t_sub_1 = b - self.epsilon - 1
        while iteration_b < self.max_iter and self.stopping_rule(b, b_t_sub_1) > self.epsilon:
            b_t_sub_1 = b
            b = b_t_sub_1 - d_b(a, b, y_array) / dd_b(a, b, y_array)
            iteration_b += 1
            
        return a, b
    
    def stopping_rule(self, current_values, previous_values):
        """
        Define stopping rule
        """
        return np.sqrt(np.sum((current_values - previous_values) ** 2) / np.sum(previous_values ** 2))
    
    def obj_fun_a(self, a):
        """
        Q(a, b*)
        """
        return -(14 * (np.log(a) + np.log(self.b_temp)) + ((self.b_temp - 1) * np.sum(np.log(self.y_temp))) / self.y_temp.shape[1] - a * np.sum(self.y_temp ** self.b_temp) / self.y_temp.shape[1])

    def obj_fun_b(self, b):
        """
        Q(a*, b)
        """
        return -(14 * (np.log(self.a_temp) + np.log(b)) + ((b - 1) * np.sum(np.log(self.y_temp))) / self.y_temp.shape[1] - self.a_temp * np.sum(self.y_temp ** b) / self.y_temp.shape[1])

    def obj_fun(self, params):
        """
        Q(a, b)
        """
        a, b = params
        return -(14 * (np.log(a) + np.log(b)) + ((b - 1) * np.sum(np.log(self.y_temp))) / self.y_temp.shape[1] - a * np.sum(self.y_temp ** b) / self.y_temp.shape[1])

    def em_gear(self):
        """
        MCEM + ECM
        Return:
            a_history, b_history, iteration
        """
        # Initiate
        a = self.a
        b = self.b
        a_previous = a - self.epsilon - 1
        b_previous = b - self.epsilon - 1
        iteration = 0
        a_history = []
        b_history = []
        a_history.append(a)
        b_history.append(b)
        
        # Main logic
        while iteration < self.max_iter and self.stopping_rule(np.array([a, b]), np.array([a_previous, b_previous])) > self.epsilon:
            print("Iteration: " + str(iteration + 1))
            a_previous = a
            b_previous = b
            y_array = self.mcem(a_previous, b_previous, iteration)
            if not self.skip_Q:
                if not self.auto:
                    a, b = self.ecm(a_previous, b_previous, y_array)
                else:
                    self.y_temp = y_array
                    self.b_temp = b_previous
                    res_a = optimize.minimize(self.obj_fun_a, a_previous, method = 'Nelder-Mead')
                    a = res_a.x
                    self.a_temp = a
                    res_b = optimize.minimize(self.obj_fun_b, b_previous, method = 'Nelder-Mead')
                    b = res_b.x
                a_history.append(a[0])
                b_history.append(b[0])   
            else:
                self.y_temp = y_array
                res = optimize.minimize(self.obj_fun, [a_previous, b_previous], method = "Nelder-Mead")
                a = res.x[0]
                b = res.x[1]
                a_history.append(a)
                b_history.append(b)   
 
            iteration += 1
            
        return a_history, b_history, iteration