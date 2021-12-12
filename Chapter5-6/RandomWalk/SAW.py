# -*- coding: utf-8 -*-

from Case1 import Case1
import numpy as np
from Measure import D_x


class SAW(Case1):
    """
    Self-avoiding walk.
    
    Args:
        n: Times of sampling. \n
        t: Max time.Default: 30. \n
        t_sd: Estimate the sd. Default: 30.
    """
    
    def __init__(self, n, t = 30, t_sd = 30):
        super().__init__(n, t, t_sd)
        
    def unvisited(self, x_t):
        """
        x_t: x_{1:t}
        """
        choices = []
        current = x_t[x_t.shape[0] - 1, :]

        for i in range(len(self.choices)):
            trial = np.array(self.choices[i]) + current
            check = True
            for j in range(x_t.shape[0]):
                if (x_t[j, 0] == trial[0]) and (x_t[j, 1] == trial[1]):
                    check = False
            if check:
                choices.append(self.choices[i])
            
        return tuple(choices)
    
    def g_t_con(self, x_t):
        """
        x_t: x_{1:t}
        """
        current = x_t[x_t.shape[0] - 1, :]
        choices = self.unvisited(x_t)
        
        if len(choices) == 0:
            return 0
        else:
            choice = choices[np.random.choice(len(choices), 1)[0]]
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
            sample_x = self.g_t_con(x_t)
            if type(sample_x) == type(0):
                print("Trapped at iteration " + str(i))
                break
            else:
                x_t = np.insert(x_t, x_t.shape[0], sample_x, 0)
                w_t *= len(self.unvisited(x_t))
        
        return x_t, w_t

    def estimate_mean(self):
        w_t_history = np.empty((0, ))
        d = []
        r = []
        for i in range(self.n):
            x, w = self.sis()
            d.append(D_x(x))
            r.append(self.M_x(x))
            w_t_history = np.insert(w_t_history, w_t_history.shape[0], w, 0)

        d = np.array(d)
        r = np.array(r)
        e_d = np.sum(d * w_t_history) / np.sum(w_t_history)
        e_r = np.sum(r * w_t_history) / np.sum(w_t_history)
        
        return e_d, e_r