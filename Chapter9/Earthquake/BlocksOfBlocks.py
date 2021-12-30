# -*- coding: utf-8 -*-

from ModelBased import ModelBased
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class BlocksOfBlocks(ModelBased):
    """
    Estimate the lag-1 autocorrelation of the annual change via blocks-of-blocks strategy.

    Args:
        B: Bootstrap times. Default: 10000.
    """
    
    def __init__(self, B=10000):
        super().__init__(B=B)
        self.M = np.mean(self.data)
        
    def bootstrap(self):
        """
        Bootstrap method.
        """
        estimates = []        
        
        for i in tqdm(range(self.B)):
            sample_index = np.random.randint(0, self.x_t.shape[0] - 1, size=49)
            block_sample_x_t = self.x_t[sample_index[0]:(sample_index[0] + 1)]
            block_sample_x_t_minus_1 = self.x_t_minus_1[sample_index[0]:(sample_index[0] + 1)]
            for j in range(1, sample_index.shape[0]):
                block_sample_x_t = np.concatenate((block_sample_x_t, self.x_t[sample_index[j]:(sample_index[j] + 2)]))
                block_sample_x_t_minus_1 = np.concatenate((block_sample_x_t_minus_1, self.x_t_minus_1[sample_index[j]:(sample_index[j] + 2)]))
            auto_cor = np.sum((block_sample_x_t - self.M) * (block_sample_x_t_minus_1 - self.M)) / np.sum((self.data - self.M) ** 2)
            estimates.append(auto_cor)            
            
        return np.array(estimates)     
    
    def bootstrap_summary(self):
        """
        C.I., st.d., histogram
        """
        print("Estimate: " + str(self.alpha))
        estimates = self.bootstrap()
        print("Bias: " + str(np.mean(estimates) - self.alpha))
        std = np.std(estimates)
        print("st.d.: " + str(std))
        print("95% C.I. Left: " + str(np.percentile(estimates, 2.5)))
        print("95% C.I. Right: " + str(np.percentile(estimates, 97.5)))
        
        counts, bins = np.histogram(estimates)
        plt.hist(bins[:-1], bins, weights=counts)