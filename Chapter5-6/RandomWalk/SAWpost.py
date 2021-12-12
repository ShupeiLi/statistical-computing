# -*- coding: utf-8 -*-

from Case1 import Case1
import numpy as np


class SAWpost(Case1):
    """
    Self-avoiding walk. Eliminating any self intersecting paths post hoc.
    
    Args:
        n: Times of sampling. \n
        t: Max time.Default: 30. \n
        t_sd: Estimate the sd. Default: 30.
    """
    
    def __init__(self, n, t = 30, t_sd = 30):
        super().__init__(n, t, t_sd)

    def path_generator(self):
        """
        Generate one path.
        """
        # Intiate
        x_t = self.x0
        iteration = 0
        
        # Main logic
        while (x_t.shape[0] < self.t):
            for i in range(x_t.shape[0], self.t):
                x_t = np.insert(x_t, x_t.shape[0], self.g_t_con(x_t), 0)
                iteration += 1
            x_t = self.delete_intersection(x_t)
        
        return x_t, iteration
            
    def delete_intersection(self, x_t):
        """
        Eliminating any selfintersecting paths.
        """
        condition = True
        
        while condition:
            values, counts = np.unique(x_t, axis = 0, return_counts = True)
            index = -1
            for i in range(counts.shape[0]):
                if counts[i] > 1:
                    index = i
                    break
            if index == -1:
                condition = False
            else:
                begin_index = -1
                end_index = -1
                target = values[index, :]
                for i in range(x_t.shape[0]):
                    if (x_t[i, 0] == target[0]) and (x_t[i, 1] == target[1]):
                        if begin_index == -1:
                            begin_index = i
                        else:
                            end_index = i
                x_t = np.delete(x_t, range(begin_index, end_index), axis = 0)
            
        return x_t