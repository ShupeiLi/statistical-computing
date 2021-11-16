# -*- coding: utf-8 -*-

import numpy as np


class Standard():
    """
    Implement standard EM in HIV infection analysis \n
    Args:
        initial_params: (alpha_0, beta_0, mu_0, lambda_0) \n
        epsilon: Threshold of the stopping rule \n
        max_iter: Max iteration. Default: 100
    """
    
    def __init__(self, initial_params, epsilon, max_iter = 100):
        self.alpha = initial_params[0]
        self.beta = initial_params[1]
        self.mu = initial_params[2]
        self.Lambda = initial_params[3]
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.n_i = np.array([379, 299, 222, 145, 109, 95, 73, 59, 45, 
                             30, 24, 12, 4, 2, 0, 1, 1])
        self.N = np.sum(self.n_i)
        
    def basic_relations(self, alpha, beta, mu, Lambda):
        """
        Calculate pi_i, z_0, t_i, p_i
        """
        pi_0 = alpha + beta * np.exp(-mu) + (1 - alpha - beta) * np.exp(-Lambda)
        index = np.arange(1, 17)
        pi_i = beta * (mu ** index) * np.exp(-mu) + (1 - alpha - beta) * (Lambda ** index) * np.exp(-Lambda)
        pi_i_c = np.concatenate((np.expand_dims(pi_0, axis = 0), pi_i))
        z_0 = alpha / pi_0
        index_c = np.arange(17)
        t_i = (beta * (mu ** index_c) * np.exp(-mu)) / pi_i_c
        p_i = (1 - alpha - beta) * (Lambda ** index_c) * np.exp(-Lambda) / pi_i_c
        return pi_i_c, z_0, t_i, p_i
    
    def update_equation(self, alpha, beta, mu, Lambda):
        """
        Calculate alpha^{(t+1)}, beta^{(t+1)}, mu^{(t+1)}, lambda^{(t+1)}
        """
        pi_i, z_0, t_i, p_i = self.basic_relations(alpha, beta, mu, Lambda)
        i = np.arange(17)
        alpha_up = (self.n_i[0] * z_0) / self.N
        beta_up = np.sum(self.n_i * t_i) / self.N
        mu_up = np.sum(i * self.n_i * t_i) / np.sum(self.n_i * t_i)
        Lambda_up = np.sum(i * self.n_i * p_i) / np.sum(self.n_i * p_i)
        return alpha_up, beta_up, mu_up, Lambda_up

    def stopping_rule(self, params_current, params_previous):
        """
        Calculate R^{(t)}
        """
        return np.sqrt(np.sum((params_current - params_previous) ** 2)) / np.sqrt(np.sum(params_previous ** 2))

    def em(self):
        """
        Standard EM \n
        Return:
            alpha_history, beta_history, mu_history, lambda_history, R_history, iteration
        """
        # Initiate
        alpha = self.alpha
        beta = self.beta
        mu = self.mu
        Lambda = self.Lambda
        alpha_t_sub_1 = alpha - self.epsilon - 1
        beta_t_sub_1 = beta - self.epsilon - 1
        mu_t_sub_1 = mu - self.epsilon - 1
        Lambda_t_sub_1 = Lambda - self.epsilon - 1
        R = self.stopping_rule(np.array([alpha, beta, mu, Lambda]), 
                               np.array([alpha_t_sub_1, beta_t_sub_1, mu_t_sub_1, Lambda_t_sub_1]))
        alpha_history = []
        beta_history = []
        mu_history = []
        lambda_history = []
        R_history = []
        alpha_history.append(alpha)
        beta_history.append(beta)
        mu_history.append(mu)
        lambda_history.append(Lambda)
        R_history.append(R)
        iteration = 0
        
        # Main logic
        while R > self.epsilon and iteration < self.max_iter:
            alpha_t_sub_1 = alpha
            beta_t_sub_1 = beta
            mu_t_sub_1 = mu
            Lambda_t_sub_1 = Lambda
            alpha, beta, mu, Lambda = self.update_equation(alpha_t_sub_1, beta_t_sub_1, 
                                                           mu_t_sub_1, Lambda_t_sub_1)
            R = self.stopping_rule(np.array([alpha, beta, mu, Lambda]), 
                                   np.array([alpha_t_sub_1, beta_t_sub_1, mu_t_sub_1, Lambda_t_sub_1]))
            alpha_history.append(alpha)
            beta_history.append(beta)
            mu_history.append(mu)
            lambda_history.append(Lambda)
            R_history.append(R)          
            iteration += 1
        
        return alpha_history, beta_history, mu_history, lambda_history, R_history, iteration