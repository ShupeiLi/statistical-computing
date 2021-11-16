# -*- coding: utf-8 -*-

import sys
sys.path.append(r".\Chapter4\EM\Peppered")
from Standard import Standard
import numpy as np
from tqdm import tqdm


class Bootstrapping(Standard):
    """
    Implement bootstrapping method to estimate variance in peppered moth analysis \n
    Args:
        initial_params: (p_C^0, p_I^0) \n
        epsilon: Threshold of the stopping rule in EM \n
        epoch: Epochs of bootstrapping \n
        max_iter: Max iteration. Default: 100
    """
    
    def __init__(self, initial_params, epsilon, epoch, max_iter = 100):
        super().__init__(initial_params, epsilon, max_iter)
        self.epoch = epoch
        
    def bootstrap(self):
        """
        Generate pseudo data
        """
        n_size = self.n_C + self.n_I + self.n_T + self.n_U
        n_sample = np.ones(self.n_C).tolist() + (np.ones(self.n_I) * 2).tolist() + (np.ones(self.n_T) * 3).tolist() + (np.ones(self.n_U) * 4).tolist()
        n_sample = np.array(n_sample, dtype = np.int64)
        trial = np.random.choice(n_sample, n_size)
        unique, counts = np.unique(trial, return_counts=True)
        n_dict = dict(zip([str(i) for i in unique], counts))
        if "1" in n_dict:
            self.n_C = n_dict["1"]
        else:
            self.n_C = 0
        if "2" in n_dict:
            self.n_I = n_dict["2"]
        else:
            self.n_I = 0
        if "3" in n_dict:
            self.n_T = n_dict["3"]
        else:
            self.n_T = 0
        if "4" in n_dict:
            self.n_U = n_dict["4"]
        else:
            self.n_U = 0
        
    def em_bootstrap(self):
        """
        EM algorithm with bootstrapping method \n
        Return:
            theta_history: [(p_C, p_I)]
        """
        # Initiate
        print("Loading...")
        theta_history = []
        p_C_history, p_I_history, _, _ = super().em()
        theta_history.append([p_C_history[-1], p_I_history[-1]])
        
        # Main logic
        for i in tqdm(range(1, self.epoch)):
            self.bootstrap()
            p_C_history, p_I_history, _, _ = super().em()
            theta_history.append([p_C_history[-1], p_I_history[-1]])            
        print("\nBootstrapping completed.")
        
        return np.array(theta_history)