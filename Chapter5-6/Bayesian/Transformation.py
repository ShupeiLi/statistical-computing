# -*- coding: utf-8 -*-

import numpy as np
from Posterior import Posterior
from scipy.stats import norm, cauchy


class Transformation(Posterior):
    """
    Calculate the integral in in Problem 5.3. d.
    """
    
    def __init__(self):
        super().__init__()
        self.correct = 0.99086
        self.k = 7.84654
        self.lower = 0 + 10 ** (-5)
        self.upper = 1 / 3
        
    def integrand(self, u):
        """
        Transformed integrand function in Problem 5.3 c.
        
        Args:
            u: A sample point.      
        """
        trans = 1 / u
        return norm.pdf(trans, self.x_bar, 3 / np.sqrt(7)) * cauchy.pdf(trans, 5, 2) * (1 / (u ** 2))    