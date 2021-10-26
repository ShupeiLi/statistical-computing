# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize
from tqdm import tqdm
import matplotlib.pyplot as plt


# 2.3
# Initiate
theta = np.array([1, 0, 0], dtype = np.float64)
t = np.array([6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20,
              22, 23, 25, 32, 32, 34, 35, 1, 1, 2, 2, 3, 4, 4,
              5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23])
w = np.array([0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 
              1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
delta = np.concatenate((np.repeat(1, 21), np.repeat(0, 21)))

# Parameters
epsilon = 10 ** (-6) 
max_iteration = 200


class Newtons():
    """
    Newton's method and its variants. Question 2.3
    """
    
    def __init__(self, theta, t, w, delta, epsilon, max_iteration):
        """
        Args:
            theta = [alpha, beta_0, beta_1]
            t, delta, w: Given in the question
            epsilon, max_iteration: Stop rule
        """
        self.theta = theta
        self.t = t
        self.w = w
        self.delta = delta
        self.epsilon = epsilon
        self.max_iteration = max_iteration
        
    def mu_term(self, theta):
        """
        Return term mu
        """
        return (self.t ** theta[0]) * np.exp(theta[1] + self.delta * theta[2])
    
    def l_fun_minus(self, theta):
        """
        Return -L
        """
        mu = self.mu_term(theta)
        return -np.sum(self.w * np.log(mu) - mu + self.w * np.log(theta[0] / self.t))
    
    def l_prime(self, theta):
        """
        Return L'
        """
        mu = self.mu_term(theta)
        d_alpha = np.sum((self.w - mu) * np.log(self.t) + self.w / theta[0])
        d_beta0 = np.sum(self.w - mu)
        d_beta1 = np.sum((self.w - mu) * self.delta)
        return np.array([d_alpha, d_beta0, d_beta1])
        
    def l_2prime(self, theta):
        """
        Return L''
        """
        mu = self.mu_term(theta)
        d_alpha2 = -np.sum(mu * (np.log(self.t) ** 2) + self.w / (theta[0] ** 2))
        d_alpha_beta0 = -np.sum(mu * np.log(self.t))
        d_alpha_beta1 = -np.sum(mu * self.delta * np.log(self.t))
        d_beta02 = -np.sum(mu)
        d_beta0_beta1 = -np.sum(mu * self.delta)
        d_delta12 = -np.sum(mu * (self.delta ** 2))
        return np.array([[d_alpha2, d_alpha_beta0, d_alpha_beta1],
                         [d_alpha_beta0, d_beta02, d_beta0_beta1],
                         [d_alpha_beta1, d_beta0_beta1, d_delta12]])
    
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
            theta_t_plus_1 = theta_t - np.matmul(np.linalg.inv(self.l_2prime(theta_t)), self.l_prime(theta_t))
            iteration += 1
        
        if iteration == self.max_iteration:
            print("Warning: Fail to converge.")
        
        return theta_t_plus_1, iteration
    
    def uni_newtons_method(self, theta, element_index, max_iteration):
        """
        Implement Newton's method (Univariate) with specified iterations
        
        Args:
            element_index: The ith element to be optimized
            max_iteration: Specified iterations
            
        Returns:
            theta_t: Estimated values
        """
        theta_t = theta
        iteration = 0
        
        while iteration < max_iteration:
            theta_t[element_index] = theta_t[element_index] - self.l_prime(theta_t)[element_index] / self.l_2prime(theta_t)[element_index, element_index]
            iteration += 1
        
        return theta_t
    
    def gauss_seidel(self):
        """
        Implement Gauss-Seidel method
        
        Returns:
            theta_t_plus_1: Estimated values
            iteration: Stopped iteration
        """
        theta_t = self.theta + self.epsilon + 1
        theta_t_plus_1 = self.theta
        iteration = 0
        
        while np.sum(np.abs(theta_t_plus_1 - theta_t)) > self.epsilon and iteration < self.max_iteration:
            theta_t = theta_t_plus_1
            theta_temp = np.copy(theta_t)
            for index in range(len(self.theta)):
                theta_temp = self.uni_newtons_method(theta_temp, index, 5)
            theta_t_plus_1 = np.copy(theta_temp)
            iteration += 1

        if iteration == self.max_iteration:
            print("Warning: Fail to converge.")
        
        return theta_t_plus_1, iteration
    
    def discrete_newton(self):
        """
        Implement discrete Newton method
        
        Returns:
            theta_t_plus_1: Estimated values
            iteration: Stopped iteration
        """
        theta_t = self.theta + self.epsilon + 0.1
        theta_t_plus_1 = self.theta
        n = len(self.theta)
        iteration = 0
        
        while np.sum(np.abs(theta_t_plus_1 - theta_t)) > self.epsilon and iteration < self.max_iteration:
            theta_t_minus_1 = np.copy(theta_t)
            theta_t = np.copy(theta_t_plus_1)
            h_t = np.reshape(np.tile((theta_t - theta_t_minus_1), n), (n, n))
            l_prime_delta = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    e_j = np.zeros(n)
                    e_j[j] = 1
                    l_prime_delta[i, j] = self.l_prime(theta_t + h_t[i, j] * e_j)[i] - self.l_prime(theta_t)[i]
            M_t_orig = l_prime_delta / h_t
            M_t_transpose = np.transpose(M_t_orig)
            M_t = (M_t_orig + M_t_transpose) / 2
            theta_t_plus_1 = theta_t - np.matmul(np.linalg.inv(M_t), self.l_prime(theta_t))
            iteration += 1

        if iteration == self.max_iteration:
            print("Warning: Fail to converge.")
        
        return theta_t_plus_1, iteration
            

def trial_and_error(method):
    """
    Plot 3D scatter
    
    Args:
        method: Enter 'newtons', 'GS', 'DN'
    """
    theta = np.zeros(3)
    newton = Newtons(theta, t, w, delta, epsilon, max_iteration)
    
    alpha = np.array([(i - 5) for i in range(10)])
    beta_0 = np.array([(i - 5) for i in range(10)])
    beta_1 = np.array([(i - 5) for i in range(10)])
    theta = np.zeros((10 ** 3, 3))
    nrow = 0
    
    for i in range(alpha.shape[0]):
        for j in range(beta_0.shape[0]):
            for k in range(beta_1.shape[0]):
                theta[nrow, 0] = alpha[i]
                theta[nrow, 1] = beta_0[j]
                theta[nrow, 2] = beta_1[k]
                nrow += 1
                
    theta_hat = np.zeros((0, 3))
    for index in tqdm(range(theta.shape[0])):
        try:
            one_theta = theta[index, :]
            newton.theta = one_theta
            if method == "newtons":
                one_theta_hat, _ = newton.newtons_method()
            elif method == "GS":
                one_theta_hat, _ = newton.gauss_seidel()
            elif method == "DN":
                one_theta_hat, _ = newton.discrete_newton()
            else:
                print("Error Args!")
                return 1
            theta_hat = np.vstack([theta_hat, one_theta_hat])
        except Exception:
            continue
    
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    ax.scatter3D(theta_hat[:, 0], theta_hat[:, 1], theta_hat[:, 2])
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta 0')
    ax.set_zlabel('beta 1')
    plt.show()
    
            
if __name__ == '__main__':    
    # 初始化实例
    newton = Newtons(theta, t, w, delta, epsilon, max_iteration)
    
    # b.
    result, iterations = newton.newtons_method()
    print(result)
    print(iterations)
    
    # c.
    optimize.minimize(newton.l_fun_minus, theta, method = 'BFGS', args = ())
    
    # d.
    result_var = np.sqrt(np.diag(np.linalg.inv(-newton.l_2prime(result))))
    dividor = np.diag(1 / result_var)
    print(result_var)
    np.matmul(np.matmul(dividor, np.linalg.inv(-newton.l_2prime(result))), dividor)
        
    # e.
    result_GS, iterations_GS = newton.gauss_seidel()
    print(result_GS)
    print(iterations_GS)
    
    # f.
    trial_and_error("newtons")
    trial_and_error("GS")
    trial_and_error("DN")
