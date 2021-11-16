# -*- coding: utf-8 -*-

import sys
sys.path.append(r".\Chapter4\EM\HIV")
from Standard import Standard
import numpy as np
from tqdm import tqdm


class Bootstrapping(Standard):
    """
    Implement bootstrapping method to estimate variance in HIV infection analysis \n
    Args:
        initial_params: (alpha_0, beta_0, mu_0, lambda_0) \n
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
        n_sample = []
        index = np.arange(17)
        for i in index:
            n_sample = n_sample + (np.ones(int(self.n_i[i])) * i).tolist()
        n_sample = np.array(n_sample, dtype = np.int64)
        trial = np.random.choice(n_sample, self.N)
        unique, counts = np.unique(trial, return_counts=True)
        n_dict = dict(zip([str(i) for i in unique], counts))      
        total_dict = dict(zip([str(i) for i in index.tolist()], np.zeros(17).tolist()))
        for key in n_dict.keys():
            if key in total_dict.keys():
                total_dict[key] = n_dict[key]
        self.n_i = np.array(list(total_dict.values()))
        
    def em_bootstrap(self):
        """
        EM algorithm with bootstrapping method \n
        Return:
            theta_history: [(alpha, beta, mu, lambda)]        
        """
        # Initiate
        print("Loading...")
        theta_history = []
        alpha_history, beta_history, mu_history, lambda_history, _, _ = super().em()
        theta_history.append([alpha_history[-1], beta_history[-1], mu_history[-1], lambda_history[-1]])        
        
        # Main logic
        for i in tqdm(range(1, self.epoch)):
            self.bootstrap()
            alpha_history, beta_history, mu_history, lambda_history, _, _ = super().em()
            theta_history.append([alpha_history[-1], beta_history[-1], mu_history[-1], lambda_history[-1]])             
        print("\nBootstrapping completed.")
        
        return np.array(theta_history)