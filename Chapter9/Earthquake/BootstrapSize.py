# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm


data_path = r"D:\Document\大学资料\大四上课件\统计计算\Ref\datasets\earthquake.dat"


class BootstrapSize():
    """
    Bootstrap block size.
    
    Args:
        B: Bootstrap times. Default: 10000. \n
        m: Size of the subset. Default: 24. \n
        l_0: Initial block lenth. Default: 5.
    """
    
    def __init__(self, B=10000, m=24, l_0=5):
        df = pd.read_table(data_path)
        self.data = np.diff(df.iloc[:, 0])
        self.B = B
        self.m = m
        self.l_0 = l_0
        
    def subset_gen(self, index, lenth):
        """
        Generate X_i^(m).
        """
        return self.data[index:(index + lenth)]

    def phi_hat(self, block, length):
        """
        Estimate phi hat via bootstrap.
        """
        block_mean = np.mean(block)
        estimates = []
        
        for i in range(self.B):
            sample_index = np.random.randint(0, block.shape[0] - length + 1, size=int(block.shape[0] / length))
            block_sample = block[sample_index[0]:(sample_index[0] + length)]
            if int(block.shape[0] / length) > 1:
                for j in range(1, sample_index.shape[0]):
                    block_sample = np.concatenate((block_sample, block[sample_index[j]:(sample_index[j] + length)]))
            estimates.append(np.mean(block_sample))
        
        estimates_array = np.array(estimates)
        return np.mean(estimates_array - block_mean)
    
    def phi_hat_zero(self):
        """
        Estimate phi_0 via bootstrap.
        """
        return self.phi_hat(self.data, self.l_0)
    
    def mse_hat(self, length):
        """
        Calculate mse hat.
        """
        phi_hat = []
        phi_hat_zero = self.phi_hat_zero()
        
        for i in range(self.data.shape[0] - self.m + 1):
            block = self.data[i:(i + self.m)]
            phi_hat.append(self.phi_hat(block, length))
            print(str(i) + " move.")
            
        phi_hat = np.array(phi_hat)
        return np.mean((phi_hat - phi_hat_zero) ** 2)
    
    def opt_mse(self, l_list=[1, 2, 3, 4, 6, 8, 12, 24]):
        """
        Find l_opt^{'(m)}
        """
        results = []
        
        for i in tqdm(range(len(l_list))):
            length = l_list[i]
            result = self.mse_hat(length)
            results.append((length, result))
            
        results_df = pd.DataFrame(results, columns=['length', 'mse'])
        return results_df.sort_values(by=['mse'])