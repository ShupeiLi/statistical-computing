# -*- coding: utf-8 -*-

from scipy.stats import cauchy, uniform
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class Failure():
    """
    Failures of bootstrap.
    
    Args:
        n: The number of samples. Default: 50. \n
        B: Bootstrap times. Default: 10000. \n
        seed: Set random_state. Default: 1234. \n
        distribution: "cauchy" or "uniform". Default: "cauchy".
    """
    
    def __init__(self, n=50, B=10000, seed=1234, distribution="cauchy"):
        self.n = n
        self.B = B
        if distribution == "cauchy":
            self.obs = cauchy.rvs(size=n, random_state=seed)
        else:
            self.obs = uniform.rvs(size=n, random_state=seed)
        
    def bootstrap(self):
        """
        Bootstrap method.
        """
        estimates = []        
        for i in tqdm(range(self.B)):
            samples = np.random.choice(self.obs, size=self.n, replace=True)
            estimates.append(np.mean(samples))
        return np.array(estimates)
    
    def bootstrap_summary(self):
        """
        C.I., st.d., histogram
        """
        estimates = self.bootstrap()
        std = np.std(estimates)
        print("st.d.: " + str(std))
        print("95% C.I. Left: " + str(np.percentile(estimates, 2.5)))
        print("95% C.I. Right: " + str(np.percentile(estimates, 97.5)))
        
        counts, bins = np.histogram(estimates)
        plt.hist(bins[:-1], bins, weights=counts)