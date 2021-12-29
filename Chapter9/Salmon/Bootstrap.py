# -*- coding: utf-8 -*-

import pandas as pd
import statsmodels.api as sm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


data_path = r".\salmon.dat"


class Bootstrap():
    """
    Bootstrap methods for fishery data.
    
    Args:
        B: Bootstrap times. Default: 10000.
    """
    
    def __init__(self, B=10000):
        df = pd.read_table(data_path, sep="\s", engine="python")
        self.variables = df.iloc[:, 1:]
        self.R = df.iloc[:, 1]
        self.S = df.iloc[:, 2]
        self.B = B
   
    def beverton_holt_fit(self):
        """
        1 / R = beta_1 + beta_2 / S
        """
        y = 1 / self.R
        x = 1 / self.S
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        return model
   
    def stable_estimate(self):
        """
        R = S -> 1 / R = 1 / S
        """
        a = np.array([[1, -1], [1, -0.6978]])
        b = np.array([0, 0.002])
        x = np.linalg.solve(a, b)
        print("1 / R = 1 / S = " + str(x[0]))
        print("R = S = " + str(1 / x[0]))
        
    def bootstrap_residual(self):
        """
        Bootstrapping the residuals.
        """
        ols = self.beverton_holt_fit()
        residuals = ols.resid
        fitted_values = np.array(ols.fittedvalues)
        x = sm.add_constant(np.array(1 / self.S, dtype=np.float64))
        estimates = []
        
        for i in tqdm(range(self.B)):
            residuals_sample = np.array(residuals.sample(n=residuals.shape[0], replace=True))
            y = fitted_values + residuals_sample
            model = sm.OLS(y, x).fit()
            estimates.append((1 - model.params[1]) / model.params[0])
            
        return np.array(estimates)
        
    def bootstrap_cases(self):
        """
        Bootstrapping the cases.
        """
        variables = self.variables
        estimates = []
        
        for i in tqdm(range(self.B)):
            variables_sample = variables.sample(n=variables.shape[0], replace=True)
            x = sm.add_constant(1 / variables_sample.iloc[:, 1])
            y = 1 / variables_sample.iloc[:, 0]
            model = sm.OLS(y, x).fit()
            estimates.append((1 - model.params[1]) / model.params[0])
            
        return np.array(estimates)
    
    def bootstrap_corrected(self):
        """
        Corrected estimator.
        """
        estimates = self.bootstrap_cases()
        estimated_bias = np.mean(estimates) - 151.1
        print("Bias: " + str(estimated_bias))
        print("Corrected: " + str(151.1 - estimated_bias))
        self.bootstrap_summary(estimates - estimated_bias)
    
    def bootstrap_summary(self, estimates):
        """
        C.I., st.d., histogram
        """
        std = np.std(estimates)
        print("st.d.: " + str(std))
        print("95% C.I. Left: " + str(np.percentile(estimates, 2.5)))
        print("95% C.I. Right: " + str(np.percentile(estimates, 97.5)))
        
        counts, bins = np.histogram(estimates)
        plt.hist(bins[:-1], bins, weights=counts)