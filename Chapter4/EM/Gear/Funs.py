# -*- coding: utf-8 -*-

import numpy as np


def z_MC(a, b, x):
    """
    Draw z_i from Z|X
    """
    y = np.random.uniform(0, 1, x.shape[0])
    return ((x ** b) - np.log(1 - y) / a) ** (1 / b)

def d_b(a, b, y_array):
    """
    Calculate dQ / db
    """
    return 14 / b + np.sum(np.log(y_array)) / y_array.shape[1] - (a * b * np.sum(np.power(y_array, b - 1))) / y_array.shape[1]
    
def dd_b(a, b, y_array):
    """
    Calculate d^2Q / db^2
    """    
    return -14 / (b ** 2) - (a * np.sum(y_array ** (b - 1) + b * (b - 1) * (y_array ** (b - 2)))) / y_array.shape[1]