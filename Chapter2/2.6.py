# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp


# 2.6
# Initiate
filepath = r".\flourbeetles.dat"
dataset = pd.read_table(filepath, sep = " ").to_numpy()
t = dataset[:, 0]
N = dataset[:, 1]
theta = np.array([1100, 0.1], dtype = np.float64)
theta_mle = np.array([100, 0.1, 1], dtype = np.float64)

# Parameters
epsilon = 10 ** (-6) 
max_iteration = 200
alpha = 0.3
beta = 0.5


class Beetles():
    """
    Question 2.6
    """
    
    def __init__(self, N, t, epsilon, max_iteration, 
                 theta = np.array([1100, 0.1]), 
                 theta_mle = np.array([1100, 0.1, 1]),
                 alpha = 0.3, beta = 0.5):
        """
        Args:
            theta = [K, r]
            N, b1, b2: Given in the question
            epsilon, max_iteration: Stop rule
        """
        self.theta = theta
        self.theta_mle = theta_mle
        self.N = N
        self.t = t
        self.N0 = N[0]
        self.n = t.shape[0]
        self.epsilon = epsilon
        self.max_iteration = max_iteration
        self.alpha = 0.3
        self.beta = 0.5
            
    def l_fun(self, theta):
        """
        Return L
        """
        return (theta[0] * self.N0) / (self.N0 + (theta[0] - self.N0) * np.exp(-theta[1] * self.t))
    
    def l_prime(self, theta):
        """
        Return L'
        """
        d_K = ((self.N0 ** 2) * (1 - np.exp(-theta[1] * self.t))) / (self.N0 + (theta[0] - self.N0) * np.exp(-theta[1] * self.t)) ** 2
        d_r = (self.t * theta[0] * self.N0 * (theta[0] - self.N0) * np.exp(-theta[1] * self.t)) / (self.N0 + (theta[0] - self.N0) * np.exp(-theta[1] * self.t)) ** 2
        return np.array([d_K, d_r])
    
    def l_2prime(self, theta):
        """
        Return L''
        """
        dividor = (self.N0 + (theta[0] - self.N0) * np.exp(-theta[1] * self.t)) ** 3
        d_K_K = (-2 * (self.N0 ** 2) * (1 - np.exp(-theta[1] * self.t)) * np.exp(-theta[1] * self.t)) / dividor
        d_K_r = (self.t * (self.N0 ** 2) * np.exp(-theta[1] * self.t) * (2 * theta[0] - self.N0 - np.exp(-theta[1] * self.t) * (theta[0] - self.N0))) / dividor
        d_r_r = ((self.t ** 2) * theta[0] * self.N0 * (theta[0] - self.N0) * np.exp(-theta[1] * self.t) * (-self.N0 + np.exp(-theta[1] * self.t) * (theta[0] - self.N0))) / dividor
        return np.array([[d_K_K, d_K_r], [d_K_r, d_r_r]])
    
    def g_prime(self, theta):
        """
        Retuen g'
        """
        l_prime_term = self.l_prime(theta)
        l_term = self.l_fun(theta)
        d_K = 2 * np.sum((self.N - l_term) * l_prime_term[0])
        d_r = 2 * np.sum((self.N - l_term) * l_prime_term[1])
        return np.array([d_K, d_r])
    
    def g_2prime(self, theta):
        """
        Return g''
        """
        l_prime_term = self.l_prime(theta)
        l_term = self.l_fun(theta)
        l_2prime_term = self.l_2prime(theta)
        d_K_K = 2 * np.sum(-(l_prime_term[0] ** 2) + (self.N - l_term) * l_2prime_term[0, 0])
        d_K_r = 2 * np.sum(-l_prime_term[0] * l_prime_term[1] + (self.N - l_term) * l_2prime_term[0, 1])
        d_r_r = 2 * np.sum(-(l_prime_term[1] ** 2) + (self.N - l_term) * l_2prime_term[1, 1])
        return np.array([[d_K_K, d_K_r], [d_K_r, d_r_r]])
    
    def gauss_newton(self):
        """
        Implement Gauss–Newton method
        
        Returns:
            theta_t_plus_1: Estimated values
            iteration: Stopped iteration
        """
        theta_t = self.theta + self.epsilon + 1
        theta_t_plus_1 = self.theta
        iteration = 0
        
        while np.sum(np.abs(theta_t_plus_1 - theta_t)) > self.epsilon and iteration < self.max_iteration:
            theta_t = theta_t_plus_1
            A_t = np.transpose(self.l_prime(theta_t))
            x_t = self.N - self.l_fun(theta_t)
            para_t = np.linalg.solve(np.matmul(np.transpose(A_t), A_t), np.matmul(np.transpose(A_t), x_t))
            theta_t_plus_1 = theta_t + para_t
            iteration += 1
        
        if iteration == self.max_iteration:
            print("Warning: Fail to converge.")
        
        return theta_t_plus_1, iteration
    
    def newtons_method(self):
        """
        Implement Newton's method
        
        Returns:
            theta_t_plus_1: Estimated values
            iteration: Stopped iteration        
        """
        theta_t = self.theta + self.epsilon + 1
        theta_t_plus_1 = self.theta
        iteration = 0
        
        while np.sum(np.abs(theta_t_plus_1 - theta_t)) > self.epsilon and iteration < self.max_iteration:
            theta_t = theta_t_plus_1
            g_prime_term = self.g_prime(theta_t)
            g_2prime_term = self.g_2prime(theta_t)
            theta_t_plus_1 = theta_t - np.linalg.solve(g_2prime_term, g_prime_term)
            iteration += 1
        
        if iteration == self.max_iteration:
            print("Warning: Fail to converge.")
        
        return theta_t_plus_1, iteration
    
    def f_ti(self):
        """
        Return f_ti (symbol)
        """
        K, N0, r, t = sp.symbols("K, N0, r, t")
# =============================================================================
#         return (K * N0) / (N0 + (K - N0) * sp.exp(-r * t))
# =============================================================================
        return (K * N0) / (2 * N0 + 2 * (K - N0) * sp.tanh(0.5 * r * t))
        
    def L_i(self):
        """
        Return L_i (symbol)
        """
        sigma, N_ti = sp.symbols("sigma, N_ti")
        f_ti = beetles.f_ti()
        return -sp.log(sigma * sp.sqrt(2 * sp.pi)) - 0.5 * ((sp.log(N_ti) - sp.log(f_ti)) / sigma) ** 2
        
    def L_i_prime(self):
        """
        Return L_i' (symbol)
        """
        L_i_term = self.L_i()
        K, r, sigma = sp.symbols("K, r, sigma")
        d_K = sp.diff(L_i_term, K)
        d_r = sp.diff(L_i_term, r)
        d_sigma = sp.diff(L_i_term, sigma)
        return [d_K, d_r, d_sigma]
    
    def L_i_2prime(self):
        """
        Return L_i'' (symbol)
        """
        L_i_prime_lst = self.L_i_prime()
        K, r, sigma = sp.symbols("K, r, sigma")
        d_K_K = sp.diff(L_i_prime_lst[0], K)
        d_K_r = sp.diff(L_i_prime_lst[0], r)
        d_K_sigma = sp.diff(L_i_prime_lst[0], sigma)
        d_r_r = sp.diff(L_i_prime_lst[1], r)
        d_r_sigma = sp.diff(L_i_prime_lst[1], sigma)
        d_sigma_sigma = sp.diff(L_i_prime_lst[2], sigma)
        return [[d_K_K, d_K_r, d_K_sigma],
                [d_K_r, d_r_r, d_r_sigma],
                [d_K_sigma, d_r_sigma, d_sigma_sigma]]
    
    def L_i_eval(self, theta_mle):
        """
        Return L_i
        """
        L_i = self.L_i()
        K, N0, r, t, sigma, N_ti = sp.symbols("K, N0, r, t, sigma, N_ti")
        f_L_i = sp.lambdify((K, N0, r, t, sigma, N_ti), L_i, "numpy")
        return np.array(f_L_i(theta_mle[0], self.N0, theta_mle[1], self.t, theta_mle[2], self.N))
    
    def L_i_prime_eval(self, theta_mle):
        """
        Return L_i'
        """
        L_i_prime_lst = self.L_i_prime()
        K, N0, r, t, sigma, N_ti = sp.symbols("K, N0, r, t, sigma, N_ti")
        f_d_K = sp.lambdify((K, N0, r, t, sigma, N_ti), L_i_prime_lst[0], "numpy")
        f_d_r = sp.lambdify((K, N0, r, t, sigma, N_ti), L_i_prime_lst[1], "numpy")
        f_d_sigma = sp.lambdify((K, N0, r, t, sigma, N_ti), L_i_prime_lst[2], "numpy")
        return np.array([f_d_K(theta_mle[0], self.N0, theta_mle[1], self.t, theta_mle[2], self.N),
                         f_d_r(theta_mle[0], self.N0, theta_mle[1], self.t, theta_mle[2], self.N),
                         f_d_sigma(theta_mle[0], self.N0, theta_mle[1], self.t, theta_mle[2], self.N)])
    
    def L_i_2prime_eval(self, theta_mle):
        """
        Return L_i''
        """
        L_i_2prime_lst = self.L_i_2prime()
        K, N0, r, t, sigma, N_ti = sp.symbols("K, N0, r, t, sigma, N_ti")
        f_d_K_K = sp.lambdify((K, N0, r, t, sigma, N_ti), L_i_2prime_lst[0][0], "numpy")
        f_d_K_r = sp.lambdify((K, N0, r, t, sigma, N_ti), L_i_2prime_lst[0][1], "numpy")
        f_d_K_sigma = sp.lambdify((K, N0, r, t, sigma, N_ti), L_i_2prime_lst[0][2], "numpy")
        f_d_r_r = sp.lambdify((K, N0, r, t, sigma, N_ti), L_i_2prime_lst[1][1], "numpy")
        f_d_r_sigma = sp.lambdify((K, N0, r, t, sigma, N_ti), L_i_2prime_lst[1][2], "numpy")
        f_d_sigma_sigma = sp.lambdify((K, N0, r, t, sigma, N_ti), L_i_2prime_lst[2][2], "numpy")
        return np.array([[f_d_K_K(theta_mle[0], self.N0, theta_mle[1], self.t, theta_mle[2], self.N),
                          f_d_K_r(theta_mle[0], self.N0, theta_mle[1], self.t, theta_mle[2], self.N),
                          f_d_K_sigma(theta_mle[0], self.N0, theta_mle[1], self.t, theta_mle[2], self.N)],
                         [f_d_K_r(theta_mle[0], self.N0, theta_mle[1], self.t, theta_mle[2], self.N),
                          f_d_r_r(theta_mle[0], self.N0, theta_mle[1], self.t, theta_mle[2], self.N),
                          f_d_r_sigma(theta_mle[0], self.N0, theta_mle[1], self.t, theta_mle[2], self.N)],
                         [f_d_K_sigma(theta_mle[0], self.N0, theta_mle[1], self.t, theta_mle[2], self.N),
                          f_d_r_sigma(theta_mle[0], self.N0, theta_mle[1], self.t, theta_mle[2], self.N),
                          f_d_sigma_sigma(theta_mle[0], self.N0, theta_mle[1], self.t, theta_mle[2], self.N)]])
        
    def L_eval(self, theta_mle):
        """
        Return L
        """
        return np.sum(self.L_i_eval(theta_mle), axis = 0)
    
    def L_prime_eval(self, theta_mle):
        """
        Return L'
        """
        return np.sum(self.L_i_prime_eval(theta_mle), axis = 1)
    
    def L_2prime_eval(self, theta_mle):
        """
        Return L''
        """
        return np.sum(self.L_i_2prime_eval(theta_mle), axis = 2)
    
    def gauss_newton_mle(self):
        """
        Implement Gauss–Newton method to find MLEs
        
        Returns:
            theta_t_plus_1: Estimated values
            iteration: Stopped iteration        
        """
        theta_t = self.theta_mle + self.epsilon + 1
        theta_t_plus_1 = self.theta_mle
        iteration = 0
        
        while np.sum(np.abs(theta_t_plus_1 - theta_t)) > self.epsilon and iteration < self.max_iteration:
            theta_t = theta_t_plus_1
            A_t = np.transpose(self.L_i_prime_eval(theta_t))
            x_t = np.log(self.N) - self.L_i_eval(theta_t)
            print(A_t)
            print(x_t)
            para_t = np.matmul(np.linalg.inv(np.matmul(np.transpose(A_t), A_t) + 10 ** (-10) * np.diag(np.ones(3))), np.matmul(np.transpose(A_t), x_t))
            print(para_t)
            theta_t_plus_1 = theta_t + para_t
            print(theta_t_plus_1)
            iteration += 1
        
        if iteration == self.max_iteration:
            print("Warning: Fail to converge.")
        
        return theta_t_plus_1, iteration        
        
    def newtons_method_mle(self, backtracking = True):
        """
        Implement Newton's method to find MLEs
        
        Returns:
            theta_t_plus_1: Estimated values
            iteration: Stopped iteration    
        """
        theta_t = self.theta_mle + self.epsilon + 1
        theta_t_plus_1 = self.theta_mle
        iteration = 0
        
        while np.sum(np.abs(theta_t_plus_1 - theta_t)) > self.epsilon and iteration < self.max_iteration:
            theta_t = theta_t_plus_1
            g_prime_term = self.L_prime_eval(theta_t)
            g_2prime_term = self.L_2prime_eval(theta_t)
            t = 1
            if backtracking:
                while self.L_eval(theta_t - t * np.matmul(np.linalg.inv(g_2prime_term), g_prime_term)) < self.L_eval(theta_t):
                    t = self.beta * t
            theta_t_plus_1 = theta_t - t * np.matmul(np.linalg.inv(g_2prime_term), g_prime_term)
            print(theta_t_plus_1)
            print(iteration)
            iteration += 1
        
        if iteration == self.max_iteration:
            print("Warning: Fail to converge.")
        
        return theta_t_plus_1, iteration        
    

def plot_fitted(theta_hat):
    """
    Visualize the fitted model
    """
    beetles = Beetles(theta, N, t, epsilon, max_iteration)
    plt.plot(t, N, ls = "solid", color = "black")
    plt.plot(t, beetles.l_fun(theta_hat), ls = "dashed", color = "blue")
    plt.show()


if __name__ == '__main__':
    # 初始化实例
    beetles = Beetles(N, t, epsilon, max_iteration, theta, theta_mle, alpha, beta)
    
    # a.
    result_GN, iterations_GN = beetles.gauss_newton()
    print(result_GN)
    print(iterations_GN)
    plot_fitted(result_GN)
    
    # b.
    result_N, iterations_N = beetles.newtons_method()
    print(result_N)
    print(iterations_N)
    plot_fitted(result_N)
    
    # c.
    result_GN_mle, iterations_GN_mle = beetles.gauss_newton_mle()
    print(result_GN_mle)
    print(iterations_GN_mle)
    plot_fitted(result_GN_mle)
    
    result_N_mle, iterations_N_mle = beetles.newtons_method_mle()
    print(result_N_mle)
    print(iterations_N_mle)
    plot_fitted(result_N_mle)
