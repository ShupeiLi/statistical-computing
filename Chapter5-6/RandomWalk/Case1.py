# -*- coding: utf-8 -*-

from Measure import D_x, R_x
import numpy as np


class Case1():
    """
    Random walk. f ~ exp, g ~ U.
    
    Args:
        n: Times of sampling. \n
        t: Max time.Default: 30. \n
        t_sd: Estimate the sd. Default: 30.
    """
    
    def __init__(self, n, t = 30, t_sd = 30):
        self.t = t
        self.choices = ((0, 1), (0, -1), (1, 0), (-1, 0))
        self.x0 = np.array([[0, 0]])
        self.n = n
        self.t_sd = t_sd
        
    def f_t(self, x_t):
        """
        x_t: x_{1:t}
        """
        return np.exp(-(D_x(x_t) + 0.5 * R_x(x_t)))
    
    def g_t_con(self, x_t):
        """
        x_t: x_{1:t}
        """
        current = x_t[x_t.shape[0] - 1, :]
        choice = self.choices[np.random.choice(4, 1)[0]]
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
            f_t = self.f_t(x_t)
            w_t *= 4 * (f_t / f_t_minus_1)
            
        return x_t, w_t

    def samplings(self):
        """
        Sampling with sis
        """
        x_t_history = np.empty((0, self.t + 1, 2))
        w_t_history = np.empty((0, ))
        
        for i in range(self.n):
            x, w = self.sis()
            x_t_history = np.insert(x_t_history, x_t_history.shape[0], x, 0)
            w_t_history = np.insert(w_t_history, w_t_history.shape[0], w, 0)

        return x_t_history, w_t_history
    
    def M_x(self, x_t):
        """
        x_t: x_{1:t}
        """
        values, counts = np.unique(x_t, axis = 0, return_counts = True)
        return np.max(counts)
    
    def estimate_mean(self):
        x_t, w_t = self.samplings()
        
        d = []
        r = []
        for i in range(self.n):
            d.append(D_x(x_t[i,:,:]))
            r.append(self.M_x(x_t[i,:,:]))
        d = np.array(d)
        r = np.array(r)
        
        e_d = np.sum(d * w_t) / np.sum(w_t)
        e_r = np.sum(r * w_t) / np.sum(w_t)
        
        return e_d, e_r

    def estimate_sd(self):
        e_d_lst = []
        e_r_lst = []
        
        for i in range(self.t_sd):
            e_d, e_r = self.estimate_mean()
            e_d_lst.append(e_d)
            e_r_lst.append(e_r)
        e_d_lst = np.array(e_d_lst)
        e_r_lst = np.array(e_r_lst)
        
        return np.std(e_d_lst), np.std(e_r_lst)