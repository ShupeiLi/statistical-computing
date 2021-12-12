# -*- coding: utf-8 -*-

import numpy as np
from Posterior import Posterior
from scipy.stats import norm, cauchy


class Improper(Posterior):
    """
    Calculate the integral in in Problem 5.3. c.
    """
    
    def __init__(self):
        super().__init__()
        self.correct = 0.99086
        self.k = 7.84654
        self.lower = np.exp(3) / (1 + np.exp(3))
        self.upper = 1 - 10 ** (-5)
        
    def integrand(self, u):
        """
        Transformed integrand function in Problem 5.3 c.
        
        Args:
            u: A sample point.      
        """
        trans = np.log(u / (1 - u))
        return norm.pdf(trans, self.x_bar, 3 / np.sqrt(7)) * cauchy.pdf(trans, 5, 2) * (1 / (u * (1 - u)))