# -*- coding: utf-8 -*-

from Tangent import Tangent
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt


class Rejection(Tangent):
    """
    Find the optimal placement of nodes for the tangent-based envelope.
    
    Args:
        n: Initial nodes. \n
        seed: Random seed. Default: 99.
    """
    
    def __init__(self, n, seed = 99):
        super().__init__(n)
        np.random.seed(seed)
        trials = np.random.uniform(0, 1, n)
        self.x0 = np.abs(trials / (1 - trials))
        
    def e_k_origin(self, x_i, x):
        return np.exp(self.e_k(x_i, x))
    
    def s_k_origin(self, x_i, x_i_plus_1, x):
        return np.exp(self.s_k(x_i, x_i_plus_1, x))
    
    def riemann_inner(self, x_i, x_i_plus_1):
        """
        Riemann rule
        """
        delta = (x_i_plus_1 - x_i) / 2
        lst = list(np.arange(x_i, x_i_plus_1, delta))
        
        total_curr = 0
        for i in lst:
            total_curr += self.s_k_origin(x_i, x_i_plus_1, i) * delta

        re = 0.1
        
        while re > 10 ** (-4):
            total_prev = total_curr
            total_curr = 0
            delta = delta / 2
            lst = list(np.arange(x_i, x_i_plus_1, delta))
            for i in lst:
                total_curr += self.s_k_origin(x_i, x_i_plus_1, i) * delta
            re = np.abs(total_curr - total_prev)

        return total_curr
    
    def riemann_outer(self, x_i_minus_1, x_i, x_i_plus_1):
        """
        Riemann rule
        """
        z_i_minus_1 = self.z_i(x_i_minus_1, x_i)
        z_i = self.z_i(x_i, x_i_plus_1)
        delta = (z_i - z_i_minus_1) / 2
        lst = list(np.arange(z_i_minus_1, z_i, delta))
        
        total_curr = 0
        for i in lst:
            total_curr += self.e_k_origin(x_i, i) * delta

        re = 0.1
        
        while re > 10 ** (-4):
            total_prev = total_curr
            total_curr = 0
            delta = delta / 2
            lst = list(np.arange(z_i_minus_1, z_i, delta))
            for i in lst:
                total_curr += self.e_k_origin(x_i, i) * delta
            re = np.abs(total_curr - total_prev)

        return total_curr
    
    def ratio(self, params):
        s = 0
        e = 0
        T_k = np.sort(np.concatenate([-params[::-1], params]))
        for i in range(T_k.shape[0] - 1):
            s += self.riemann_inner(T_k[i], T_k[i + 1])
            if i != (T_k.shape[0] - 2):
                e += self.riemann_outer(T_k[i], T_k[i + 1], T_k[i + 2])

        e += self.riemann_outer(T_k[0], T_k[0], T_k[1])
        e += self.riemann_outer(T_k[-2], T_k[-1], T_k[-1])
        e += self.riemann_outer(-4, T_k[0], T_k[0])
        e += self.riemann_outer(T_k[-1], T_k[-1], 4)

        return e - s
    
    def placement(self):
        """
        Optimize the placement of nodes
        """    
        result = minimize(self.ratio, self.x0)
        
        if result.success:
            fitted_params = result.x
            print("Best placements: " + str(np.sort(fitted_params)))
            return np.sort(fitted_params)
        else:
            print("Fail to optimize.")
            return 0
        
    def plot_placement(self, solution):
        T_k = np.sort(np.concatenate([-solution[::-1], solution]))
        T_k_y = norm.pdf(T_k, 0, 1)
        base = np.arange(min(T_k) - 1, max(T_k) + 1, 0.01)
        base_y = norm.pdf(base, 0, 1)
        
        s_base = []
        s_base_y = []
        for i in range(T_k.shape[0] - 1):
            k_base = np.arange(T_k[i], T_k[i + 1], 0.01)
            s_base = s_base + list(k_base)
            for j in range(k_base.shape[0]):
                s_base_y.append(self.s_k_origin(T_k[i], T_k[i + 1], k_base[j]))
                    
        e_base = []
        e_base_y = []
        for i in range(T_k.shape[0]):
            if i == 0:
                k_base = np.arange(min(T_k) - 1, self.z_i(T_k[i], T_k[i + 1]), 0.01)
            elif i == (T_k.shape[0] - 1):
                k_base = np.arange(self.z_i(T_k[i - 1], T_k[i]), max(T_k) + 1, 0.01)
            else:
                k_base = np.arange(self.z_i(T_k[i - 1], T_k[i]), self.z_i(T_k[i], T_k[i + 1]), 0.01)
            e_base = e_base + list(k_base)
            for j in range(k_base.shape[0]):
                e_base_y.append(self.e_k_origin(T_k[i], j))
        
        plt.plot(base, base_y, color = "black", lw = 0.5)
        plt.plot(s_base, s_base_y, '-.', lw = 0.5)
        plt.plot(e_base, e_base_y, '-', color = "red", lw = 0.5)
        plt.plot(T_k, T_k_y, 'bo', ms = 1.5)
        plt.show()