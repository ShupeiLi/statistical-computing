# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


data_path = r"D:\Document\大学资料\大四上课件\统计计算\Ref\datasets\earthquake.dat"


class MovingBlock():
    """
    Estimate the 90th percentile of the annual change via moving block bootstrap.
    
    Args:
        B: Bootstrap times. Default: 10000.
    """
    
    def __init__(self, B=10000):
        df = pd.read_table(data_path)
        self.data = np.diff(df.iloc[:, 0])
        self.B = B

    def bootstrap(self):
        """
        Bootstrap method.
        """
        estimates = []        
        
        for i in tqdm(range(self.B)):
            sample_index = np.random.randint(0, self.data.shape[0] - 1, size=49)
            block_sample = self.data[sample_index[0]:(sample_index[0] + 2)]
            for j in range(1, sample_index.shape[0]):
                block_sample = np.concatenate((block_sample, self.data[sample_index[j]:(sample_index[j] + 2)]))
            estimates.append(np.percentile(block_sample, 90))

        return np.array(estimates)        
    
    def bootstrap_summary(self):
        """
        C.I., st.d., histogram
        """
        print("Estimate: " + str(np.percentile(self.data, 90)))
        estimates = self.bootstrap()
        std = np.std(estimates)
        print("st.d.: " + str(std))
        print("95% C.I. Left: " + str(np.percentile(estimates, 2.5)))
        print("95% C.I. Right: " + str(np.percentile(estimates, 97.5)))
        
        counts, bins = np.histogram(estimates)
        plt.hist(bins[:-1], bins, weights=counts)