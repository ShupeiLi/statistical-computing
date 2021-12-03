# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm, cauchy


class Verify():
    """
    Verify k in Problem 5.3. a.
    """
    
    def __init__(self):
        self.obs = np.array([6.52, 8.32, 0.31, 2.82, 9.96, 0.14, 9.64])
        self.x_bar = np.mean(self.obs)
        
    def integrand(self, mu):
        """
        Integrand function in Problem 5.3 a.
        
        Args:
            mu: A sample point.
        """
        return norm.pdf(mu, self.x_bar, 3 / np.sqrt(7)) * cauchy.pdf(mu, 5, 2)
    
    def stop_rule(self, prev, curr):
        """
        Relative convergence
        """
        return np.abs(curr - prev) / prev
    
    def riemann(self):
        """
        Riemann Rule
        
        Returns:
            constant k, iterations
        """
        
        lower = self.x_bar - 50
        upper = self.x_bar + 50
        delta = (upper - lower) / 2
        lst = list(np.arange(lower, upper, delta))
        total_curr = 0
        for i in lst:
            total_curr += self.integrand(i) * delta
        re = 0.1
        iterations = 0
        
        while re > 10 ** (-4):
            total_prev = total_curr
            total_curr = 0
            delta = delta / 2
            lst = list(np.arange(lower, upper, delta))
            for i in lst:
                total_curr += self.integrand(i) * delta

            re = self.stop_rule(total_prev, total_curr)
            iterations += 1
                
        return 1 / total_curr, iterations