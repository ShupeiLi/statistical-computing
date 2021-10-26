# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# 2.5
# Initiate
filepath = r".\oilspills.dat"
dataset = pd.read_table(filepath, sep = " ").iloc[:, 1:].to_numpy()
N = dataset[:, 0]
b1 = dataset[:, 1]
b2 = dataset[:, 2]
theta = np.array([1.085, 1.085], dtype = np.float64)

# Parameters
epsilon = 10 ** (-6) 
max_iteration = 200


class OilSpills():
    """
    Question 2.5
    """
    
    def __init__(self, theta, N, b1, b2, epsilon, max_iteration):
        """
        Args:
            theta = [alpha1, alpha2]
            N, b1, b2: Given in the question
            epsilon, max_iteration: Stop rule
        """
        self.theta = theta
        self.N = N
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.max_iteration = max_iteration
        
    def lambda_term(self, theta):
        """
        Calculate lambda term
        """
        return theta[0] * self.b1 + theta[1] * self.b2
    
    def l_fun_kernel(self, theta):
        """
        Return L kernel
        """
        lambda_t = self.lambda_term(theta)
        return np.sum(self.N * np.log(lambda_t) - lambda_t)
        
    def l_prime(self, theta):
        """
        Return L'
        """
        lambda_t = self.lambda_term(theta)
        d_alpha1 = np.sum(self.b1 * (self.N / lambda_t - 1))
        d_alpha2 = np.sum(self.b2 * (self.N / lambda_t - 1))
        return np.array([d_alpha1, d_alpha2])
        
    def l_2prime(self, theta):
        """
        Return L''
        """
        lambda_t = self.lambda_term(theta)
        d_alpha1_alpha1 = -np.sum(self.N * (self.b1 / lambda_t) ** 2)
        d_alpha1_alpha2 = -np.sum((self.N * self.b1 * self.b2) / (lambda_t) ** 2)
        d_alpha2_alpha2 = -np.sum(self.N * (self.b2 / lambda_t) ** 2)
        return np.array([[d_alpha1_alpha1, d_alpha1_alpha2],
                         [d_alpha1_alpha2, d_alpha2_alpha2]])
    
    def fisher_info(self, theta):
        """
        Return I
        """
        lambda_t = self.lambda_term(theta)
        d_alpha1_alpha1 = np.sum(self.b1 ** 2 / lambda_t)
        d_alpha1_alpha2 = np.sum((self.b1 * self.b2) / lambda_t)
        d_alpha2_alpha2 = np.sum(self.b2 ** 2 / lambda_t)
        return np.array([[d_alpha1_alpha1, d_alpha1_alpha2],
                         [d_alpha1_alpha2, d_alpha2_alpha2]])        
    
    def newtons_method(self):
        """
        Implement Newton's method
        
        Returns:
            theta_t_plus_1: Estimated values
            iteration: Stopped iteration
            estimates: Estimated values at each iteration
        """
        theta_t = self.theta + self.epsilon + 1
        theta_t_plus_1 = self.theta
        estimates =np.zeros((0, 2))
        estimates = np.vstack([estimates, theta_t_plus_1])
        iteration = 0
        
        while np.sum(np.abs(theta_t_plus_1 - theta_t)) > self.epsilon and iteration < self.max_iteration:
            theta_t = theta_t_plus_1
            theta_t_plus_1 = theta_t - np.matmul(np.linalg.pinv(self.l_2prime(theta_t)), self.l_prime(theta_t))
            estimates = np.vstack([estimates, theta_t_plus_1])
            iteration += 1
        
        if iteration == self.max_iteration:
            print("Warning: Fail to converge.")
        
        return theta_t_plus_1, iteration, estimates

    def fisher_scoring_method(self):
        """
        Implement Fisher scoring method
        
        Returns:
            theta_t_plus_1: Estimated values
            iteration: Stopped iteration
            estimates: Estimated values at each iteration
        """        
        theta_t = self.theta + self.epsilon + 1
        theta_t_plus_1 = self.theta
        estimates =np.zeros((0, 2))
        estimates = np.vstack([estimates, theta_t_plus_1])
        iteration = 0
        
        while np.sum(np.abs(theta_t_plus_1 - theta_t)) > self.epsilon and iteration < self.max_iteration:
            theta_t = theta_t_plus_1
            theta_t_plus_1 = theta_t + np.matmul(np.linalg.inv(self.fisher_info(theta_t)), self.l_prime(theta_t))
            estimates = np.vstack([estimates, theta_t_plus_1])
            iteration += 1
        
        if iteration == self.max_iteration:
            print("Warning: Fail to converge.")
        
        return theta_t_plus_1, iteration, estimates       
    
    def steepest_ascent(self, backtracking = True):
        """
        Implement steepest ascent method with / without step-halving backtracking
        
        Returns:
            theta_t_plus_1: Estimated values
            iteration: Stopped iteration
            estimates: Estimated values at each iteration
        """
        theta_t = self.theta + self.epsilon + 1
        theta_t_plus_1 = self.theta
        estimates =np.zeros((0, 2))
        estimates = np.vstack([estimates, theta_t_plus_1])
        iteration = 0
        
        while np.sum(np.abs(theta_t_plus_1 - theta_t)) > self.epsilon and iteration < self.max_iteration:
            theta_t = theta_t_plus_1
            alpha_t = 1
            if backtracking:
                while self.l_fun_kernel(theta_t + alpha_t * self.l_prime(theta_t)) < self.l_fun_kernel(theta_t):
                    alpha_t = alpha_t * 0.5
            theta_t_plus_1 = theta_t + alpha_t * self.l_prime(theta_t)
            estimates = np.vstack([estimates, theta_t_plus_1])
            iteration += 1
        
        if iteration == self.max_iteration:
            print("Warning: Fail to converge.")
        
        return theta_t_plus_1, iteration, estimates       
        
    def quasi_newton(self, backtracking = True):
        """
        Implement Quasi-Newton method with / without step-halving backtracking
        
        Returns:
            theta_t_plus_1: Estimated values
            iteration: Stopped iteration
            estimates: Estimated values at each iteration
        """
        theta_t = self.theta + self.epsilon + 1
        theta_t_plus_1 = self.theta
        estimates =np.zeros((0, 2))
        estimates = np.vstack([estimates, theta_t_plus_1])
        iteration = 0
        M_t = np.diag(-np.ones((2)))
        
        while np.sum(np.abs(theta_t_plus_1 - theta_t)) > self.epsilon and iteration < self.max_iteration:
            theta_t = theta_t_plus_1
            alpha_t = 1
            if backtracking:
                while self.l_fun_kernel(theta_t - alpha_t * np.matmul(np.linalg.inv(M_t), self.l_prime(theta_t))) < self.l_fun_kernel(theta_t):
                    alpha_t = alpha_t * 0.5
            theta_t_plus_1 = theta_t - alpha_t * np.matmul(np.linalg.inv(M_t), self.l_prime(theta_t))
            estimates = np.vstack([estimates, theta_t_plus_1])
            z_t = theta_t_plus_1 - theta_t
            y_t = self.l_prime(theta_t_plus_1) - self.l_prime(theta_t)
            v_t = y_t - np.matmul(M_t, z_t)
            c_t = 1 / (np.matmul(np.transpose(v_t), z_t))
            M_t = M_t + np.matmul(c_t * v_t, np.transpose(v_t))
            iteration += 1
        
        if iteration == self.max_iteration:
            print("Warning: Fail to converge.")
        
        return theta_t_plus_1, iteration, estimates               
    
        
def trial_and_error_contour(method, range_value):
    """
    Plot contour map
    
    Args:
        method: Enter 'newtons', 'fisher', 'steepest'
        range_value: Specify the range of alpha 0 (list)
    """
    theta = np.zeros(2)
    oilspills = OilSpills(theta, N, b1, b2, epsilon, max_iteration)
    
    alpha1, alpha2 = np.array(np.meshgrid(np.array([(i - range_value[0]) * range_value[1] for i in range(10)], dtype = np.float64), 
                                          np.array([(i - range_value[0]) * range_value[1] for i in range(10)], dtype = np.float64)))
    alpha1_re = alpha1.reshape(100)
    alpha2_re = alpha2.reshape(100)
    theta = np.zeros((100, 2))
    nrow = 0
    
    for i in range(alpha1_re.shape[0]):
        theta[nrow, 0] = alpha1_re[i]
        theta[nrow, 1] = alpha2_re[i]
        nrow += 1
                
    theta_hat = np.zeros((100, 2))
    for index in tqdm(range(theta.shape[0])):
        try:
            one_theta = theta[index, :]
            oilspills.theta = one_theta
            if method == "newtons":
                one_theta_hat, _, _ = oilspills.newtons_method()
            elif method == "fisher":
                one_theta_hat, _, _ = oilspills.fisher_scoring_method()
            elif method == "steepest":
                one_theta_hat, _, _ = oilspills.steepest_ascent()
            else:
                print("Error Args!")
                return 1
            theta_hat[index, :] = one_theta_hat
        except Exception:
            continue
    
    alpha1_hat = theta_hat[:, 0].reshape((10, 10))
    alpha2_hat = theta_hat[:, 1].reshape((10, 10))
    
    plt.colorbar(plt.contourf(alpha1, alpha2, alpha1_hat))
    plt.title("alpha 1 hat")
    plt.show()
    
    plt.colorbar(plt.contourf(alpha1, alpha2, alpha2_hat))
    plt.title("alpha 2 hat")
    plt.show()


def algorithm_contour(estimates_N, estimates_fisher, estimates_SA, estimates_QN):
    """
    Plot contour map of selected algorithms
    """
    theta = np.zeros(2)
    oilspills = OilSpills(theta, N, b1, b2, epsilon, max_iteration)
    alpha1, alpha2 = np.array(np.meshgrid(np.array([(i + 4000) / 4000  for i in range(800)], dtype = np.float64), 
                                          np.array([(i + 3600) / 4000 for i in range(800)], dtype = np.float64)))
    alpha1_re = alpha1.reshape(800 ** 2)
    alpha2_re = alpha2.reshape(800 ** 2)
    theta = np.zeros(800 ** 2)

    for i in tqdm(range(alpha1_re.shape[0])):
        theta[i] = oilspills.l_fun_kernel(np.array([alpha1_re[i], alpha2_re[i]])) 
 
    
    theta = theta.reshape((800, 800))
    xmin = np.min(np.array([(i + 4000) / 4000  for i in range(800)]))
    xmax = np.max(np.array([(i + 4000) / 4000  for i in range(800)]))
    ymin = np.min(np.array([(i + 3600) / 4000  for i in range(800)]))
    ymax = np.max(np.array([(i + 3600) / 4000  for i in range(800)]))

    plt.colorbar(plt.contourf(alpha1, alpha2, theta, 50))
    plt.title("theta hat")
    plt.plot(estimates_N[:, 0], estimates_N[:, 1], color = "blue")
    plt.plot(estimates_fisher[:, 0], estimates_fisher[:, 1], color = "purple")
    plt.plot(estimates_SA[:, 0], estimates_SA[:, 1], color = "red")
    plt.plot(estimates_QN[:, 0], estimates_QN[:, 1], color = "orange")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()
    
    plt.colorbar(plt.contourf(alpha1, alpha2, theta, 50))
    plt.title("theta hat")
    plt.plot(estimates_N[:, 0], estimates_N[:, 1], color = "blue")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()
    
    plt.colorbar(plt.contourf(alpha1, alpha2, theta, 50))
    plt.title("theta hat")
    plt.plot(estimates_fisher[:, 0], estimates_fisher[:, 1], color = "purple")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()
    
    plt.colorbar(plt.contourf(alpha1, alpha2, theta, 50))
    plt.title("theta hat")
    plt.plot(estimates_SA[:, 0], estimates_SA[:, 1], color = "red")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()
    
    plt.colorbar(plt.contourf(alpha1, alpha2, theta, 50))
    plt.title("theta hat")
    plt.plot(estimates_QN[:, 0], estimates_QN[:, 1], color = "orange")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()
    

if __name__ == '__main__':
    # 初始化实例
    oilspills = OilSpills(theta, N, b1, b2, epsilon, max_iteration)
    
    # c.
    result_N, iterations_N, estimates_N = oilspills.newtons_method()
    print(result_N)
    print(iterations_N)
    
    result_fisher, iterations_fisher, estimates_fisher = oilspills.fisher_scoring_method()
    print(result_fisher)
    print(iterations_fisher)
    
    trial_and_error_contour("newtons", [5, 1])
    trial_and_error_contour("fisher", [5, 1])
    
    # d.
    print(np.sqrt(np.diag(np.linalg.inv(oilspills.fisher_info(result_N)))))
    print(np.sqrt(np.diag(np.linalg.inv(oilspills.fisher_info(result_fisher)))))
    
    # e.
    result_SA, iterations_SA, estimates_SA = oilspills.steepest_ascent()
    print(result_SA)
    print(iterations_SA)
    
    result_SA_N, iterations_SA_N, estimates_SA_N = oilspills.steepest_ascent(False)
    print(result_SA_N)
    print(iterations_SA_N)
    
    # f.
    result_QN, iterations_QN, estimates_QN = oilspills.quasi_newton()
    print(result_QN)
    print(iterations_QN)
    
    result_QN_N, iterations_QN_N, estimates_QN_N = oilspills.quasi_newton(False)
    print(result_QN_N)
    print(iterations_QN_N)
    
    # g.
    algorithm_contour(estimates_N, estimates_fisher, estimates_SA, estimates_QN)
    