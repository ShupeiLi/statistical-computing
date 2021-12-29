# -*- coding: utf-8 -*-

from Bootstrap import Bootstrap


if __name__ == '__main__':
    # a.
    # Fit the model
    model = Bootstrap()
    ols = model.beverton_holt_fit()
    print(ols.summary())
    
    # Point estimate
    model.stable_estimate()
    
    # Bootstrapping the residuals
    bootstrap_resids = model.bootstrap_residual()
    model.bootstrap_summary(bootstrap_resids)
    
    # Bootstrapping the cases
    bootstrap_cases = model.bootstrap_cases()
    model.bootstrap_summary(bootstrap_cases)
    
    # b.
    model.bootstrap_corrected()