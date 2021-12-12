# -*- coding: utf-8 -*-

import numpy as np
from Verify import Verify


class Posterior(Verify):
    """
    Implement methods in in Problem 5.3. b.
    """
    
    def __init__(self):
        super().__init__()
        self.correct = 0.99605
        self.k = 7.84654
        self.lower = 2
        self.upper = 8
        
    def riemann(self):
        """
        Riemann Rule

        Returns:
            Integral of the posterior probability (history), bias (history).
        """
        delta = (self.upper - self.lower) / 2
        lst = list(np.arange(self.lower, self.upper, delta))
        
        total_curr = 0
        for i in lst:
            total_curr += self.integrand(i) * delta
        re = 0.1
        history_total = []
        history_bias = []
        history_total.append(total_curr * self.k)
        history_bias.append(np.abs(total_curr * self.k - self.correct))
        
        while re > 10 ** (-4):
            total_prev = total_curr
            total_curr = 0
            delta = delta / 2
            lst = list(np.arange(self.lower, self.upper, delta))
            for i in lst:
                total_curr += self.integrand(i) * delta
            re = self.stop_rule(total_prev, total_curr)
            history_total.append(total_curr * self.k)
            history_bias.append(np.abs(total_curr * self.k - self.correct))

        return history_total, history_bias
    
    def trapezoidal(self):
        """
        Trapezoidal Rule

        Returns:
            Integral of the posterior probability (history), bias (history).        
        """
        delta = (self.upper - self.lower) / 2
        lst = list(np.arange(self.lower, self.upper + 10 ** (-8), delta))
        
        total_curr = 0
        for i in range(len(lst)):
            if i == 0 or i == (len(lst) - 1):
                total_curr += self.integrand(lst[i]) * (delta / 2)
            else:
                total_curr += self.integrand(lst[i]) * delta
        re = 0.1
        history_total = []
        history_bias = []
        history_total.append(total_curr * self.k)
        history_bias.append(np.abs(total_curr * self.k - self.correct))        
        
        while re > 10 ** (-4):
            total_prev = total_curr
            total_curr = 0
            delta = delta / 2
            lst = list(np.arange(self.lower, self.upper + 10 ** (-8), delta))
            for i in range(len(lst)):
                if i == 0 or i == (len(lst) - 1):
                    total_curr += self.integrand(lst[i]) * (delta / 2)
                else:
                    total_curr += self.integrand(lst[i]) * delta
            re = self.stop_rule(total_prev, total_curr)
            history_total.append(total_curr * self.k)
            history_bias.append(np.abs(total_curr * self.k - self.correct))

        return history_total, history_bias        
        
    def simpson(self):
        """
        Simpson’s Rule

        Returns:
            Integral of the posterior probability (history), bias (history).            
        """
        inter = 2
        delta = (self.upper - self.lower) / inter
        lst = list(np.arange(self.lower, self.upper + 10 ** (-8), delta))
        
        f_lst = []
        for item in lst:
            f_lst.append(self.integrand(item))
            
        total_curr = 0
        for i in range(1, int(inter / 2) + 1):
            total_curr += (delta / 3) * (f_lst[2 * i - 2] + 4 * f_lst[2 * i - 1] + f_lst[2 * i])
        
        re = 0.1
        history_total = []
        history_bias = []
        history_total.append(total_curr * self.k)
        history_bias.append(np.abs(total_curr * self.k - self.correct))   
        
        while re > 10 ** (-4):
            total_prev = total_curr
            inter = int(inter * 2)
            delta = (self.upper - self.lower) / inter
            lst = list(np.arange(self.lower, self.upper + 10 ** (-8), delta))
            
            f_lst = []
            for item in lst:
                f_lst.append(self.integrand(item))
                
            total_curr = 0
            for i in range(1, int(inter / 2) + 1):
                total_curr += (delta / 3) * (f_lst[2 * i - 2] + 4 * f_lst[2 * i - 1] + f_lst[2 * i])
                        
            re = self.stop_rule(total_prev, total_curr)
            history_total.append(total_curr * self.k)
            history_bias.append(np.abs(total_curr * self.k - self.correct))

        return history_total, history_bias               