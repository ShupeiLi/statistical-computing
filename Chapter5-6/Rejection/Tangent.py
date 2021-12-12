# -*- coding: utf-8 -*-

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


class Tangent():
    """
    Tangent-based envelope. 
    
    Args:
        n: number of nodes. \n
        specify / specify_array: Enter specific initial placement of nodes.
    """
    
    def __init__(self, n, specify = True, specify_array = []):
        self.n = 2 * n
        self.c = norm.ppf(1 - 10 ** (-5))
        self.fc = -norm.logpdf(self.c, 0, 1)
        c_lst = []
        for i in range(n):
            c_lst.append(self.draw())
        c_lst = np.array(c_lst)
        self.c_lst = c_lst
        if specify:
            self.c_lst = specify_array
        self.T_k = np.sort(np.concatenate([-c_lst[::-1], c_lst]))
            
    
    def draw(self):
        trial_u = np.random.uniform(0, 1, 1)[0]
        trial = trial_u / (1 - trial_u)
        while (trial <= -self.c) or (trial >= self.c):
            trial_u = np.random.uniform(0, 1, 1)[0]
            trial = trial_u / (1 - trial_u)
        return trial
    
    def z_i(self, x_i, x_i_plus_1):
        return (x_i + x_i_plus_1) / 2
   
    def e_k(self, x_i, x):
        e_i = -x_i * x - 0.5 * np.log(2 * np.pi) + x_i ** 2 / 2
        return e_i
    
    def s_k(self, x_i, x_i_plus_1, x):
        s_i = -0.5 * (x_i + x_i_plus_1) * x - 0.5 * np.log(2 * np.pi) + 0.5 * x_i * x_i_plus_1
        return s_i
    
    def upper_s(self, x_i = 0, x_i_plus_1 = 0, x_i_plus_2 = 0, index = 0):
        """
        Calculate the area of upper hull.
        """
        if index == 0:
            z_1 = x_i
            z_2 = self.z_i(x_i, x_i_plus_1)
            e_2 = self.e_k(x_i, z_2)
            s = 0.5 * (z_2 - z_1) * e_2
        elif index == (self.n - 1):
            z_1 = self.z_i(x_i, x_i_plus_1)
            z_2 = x_i_plus_1
            e_1 = self.e_k(x_i_plus_1, z_1)
            s = 0.5 * (z_2 - z_1) * e_1
        else:
            z_1 = self.z_i(x_i, x_i_plus_1)
            z_2 = self.z_i(x_i_plus_1, x_i_plus_2)
            e_1 = self.e_k(x_i_plus_1, z_1)
            e_2 = self.e_k(x_i_plus_1, z_2)
            s = 0.5 * (z_2 - z_1) * (e_1 + e_2)
            
        return s
    
    def lower_s(self, x_i, x_i_plus_1):
        """
        Calculate the area of lower hull.
        """
        s = 0.5 * (x_i_plus_1 - x_i) * (self.s_k(x_i, x_i_plus_1, x_i) + self.s_k(x_i, x_i_plus_1, x_i_plus_1))
        return s
        
    def tangent_based(self, threshold = 0.95):
        """
        Optimization.
        """
        # Initiate
        T_k = self.T_k
        c_lst = self.c_lst
        ratio = threshold - 0.1
        
        # Main logic, update self.n
        while ratio < threshold:
            upper_s_i = []
            lower_s_i = []
            for i in range(len(T_k) - 1):
                if i == 0:
                    upper_s_i.append(0.5 * (norm.logpdf(T_k[i]) + self.e_k(T_k[i], self.z_i(T_k[i], T_k[i + 1])) + 2 * self.fc) * (T_k[i + 1] - T_k[i]))
                elif i == (len(T_k) - 2):
                    upper_s_i.append(0.5 * (norm.logpdf(T_k[i + 1]) + self.e_k(T_k[i + 1], self.z_i(T_k[i], T_k[i + 1])) + 2 * self.fc) * (T_k[i + 1] - T_k[i]))
                else:
                    upper_s_i.append(0.5 * (self.e_k(T_k[i], self.z_i(T_k[i - 1], T_k[i])) + self.e_k(T_k[i], self.z_i(T_k[i], T_k[i + 1])) + 2 * self.fc) * (self.z_i(T_k[i + 1], T_k[i + 2]) - self.z_i(T_k[i], T_k[i + 1])))
                    
                lower_s_i.append(0.5 * (norm.logpdf(T_k[i]) + norm.logpdf(T_k[i + 1]) + 2 * self.fc) * (T_k[i + 1] - T_k[i]))
        
            upper_s = sum(upper_s_i)
            lower_s = sum(lower_s_i)
            ratio = lower_s / upper_s
            c_lst = list(c_lst)
            c_lst.append(self.draw())
            c_lst = np.array(c_lst)
            T_k = np.sort(np.concatenate([-c_lst[::-1], c_lst]))
        
        return T_k
    
    def z_is(self, T_k):
        T_k = list(T_k)
        T = []
        for i in range(len(T_k) - 1):
            T.append(self.z_i(T_k[i], T_k[i + 1]))
        T.append(T_k[len(T_k) - 1] + 0.1)
        
        T = np.array(T)
        return T
    
    def plot_hull(self, T_k):
        z_s = self.z_is(T_k)
        f_z_s = norm.logpdf(T_k) - (z_s - T_k) * T_k
        z_s = [T_k[0] - 0.1] + list(z_s)
        z_s = np.array(z_s)
        f_z_s = [norm.logpdf(T_k[0]) - (z_s[0] - T_k[0]) * T_k[0]] + list(f_z_s)
        f_z_s = np.array(f_z_s)
        
        T_k = list(T_k)
        T_k = [T_k[0]] + T_k + [T_k[-1]]
        T_k = np.array(T_k)
        f_T_k = norm.logpdf(T_k, 0, 1)
        f_T_k[0] = min(f_T_k) - 0.5
        f_T_k[-1] = f_T_k[0]
        
        base = np.arange(min(T_k) - 0.1, max(T_k) + 0.1, 0.01)
        base_y = norm.logpdf(base, 0, 1)
        
        plt.plot(base, base_y, color = "black", lw = 0.5)
        plt.plot(T_k, f_T_k, '-.', lw = 0.5)
        plt.plot(z_s, f_z_s, '-', color = "red", lw = 0.5)
        plt.plot(T_k, f_T_k, 'bo', ms = 1.5)
        plt.show()