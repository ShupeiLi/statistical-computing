# -*- coding: utf-8 -*-

import numpy as np
from MovingBlock import MovingBlock
from tqdm import tqdm
import statsmodels.api as sm


class ModelBased(MovingBlock):
    """
    Estimate st.d. of the 90th percentile estimator based on AR(1) assumption.
    
    Args:
        B: Bootstrap times. Default: 10000.
    """
    
    def __init__(self, B=10000):
        super().__init__(B=B)
        ar = sm.tsa.arima.ARIMA(self.data, order=(1, 0, 0))
        res = ar.fit()
        self.alpha = res.arparams[0]
        self.x_t_minus_1 = self.data[:-1]
        self.x_t = self.data[1:]
        e_t_hat = self.x_t - self.alpha * self.x_t_minus_1
        self.epsilon_hat = e_t_hat - np.mean(e_t_hat)

    def bootstrap(self):
        """
        Bootstrap method.
        """
        estimates = []        
        
        for i in tqdm(range(self.B)):
            sample_epsilon = np.random.choice(self.epsilon_hat, size=(self.data.shape[0] + 1), replace=True)
            sample_data = [sample_epsilon[0]]
            for j in range(1, sample_epsilon.shape[0]):
                sample_x_t = self.alpha * sample_data[j - 1] + sample_epsilon[j]
                sample_data.append(sample_x_t)
            sample_data = np.array(sample_data)
            estimates.append(np.percentile(sample_data, 90))

        return np.array(estimates)      