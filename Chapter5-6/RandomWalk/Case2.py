# -*- coding: utf-8 -*-

from Case1 import Case1
import numpy as np


class Case2(Case1):
    """
    Random walk. f ~ exp, g prop to f.
    
    Args:
        n: Times of sampling. \n
        t: Max time.Default: 30. \n
        t_sd: Estimate the sd. Default: 30.
    """
    
    def __init__(self, n, t = 30, t_sd = 30):
        super().__init__(n, t, t_sd)
        
    def g_t_con(self, x_t):
        """
        x_t: x_{1:t}
        """
        current = x_t[x_t.shape[0] - 1, :]
        possibility = []
        for i in range(len(self.choices)):
            trial = np.array(self.choices[i]) + current
            trials = np.insert(x_t, x_t.shape[0], trial, 0)
            possibility.append(self.f_t(trials))
        possibility = np.array(possibility)
        possibility = possibility / np.sum(possibility)
        choice = self.choices[np.random.choice(4, 1, p = possibility)[0]]
        update = np.array(choice) + current
        return update    

    def sis(self):
        """
        Sequential importance sampling.
        """
        # Intiate
        x_t = self.x0
        w_t = 1
        
        # Main logic
        for i in range(self.t):
            f_t_minus_1 = self.f_t(x_t)
            x_t = np.insert(x_t, x_t.shape[0], self.g_t_con(x_t), 0)
            w_t *= 1 / f_t_minus_1
            
        return x_t, w_t