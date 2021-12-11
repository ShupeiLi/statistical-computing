# -*- coding: utf-8 -*-

from Tangent import Tangent
from Rejection import Rejection
import numpy as np


if __name__ == '__main__':
    # n = 1
    model = Rejection(1)
    T_k = model.placement()
    solution = np.sort(np.concatenate([-T_k[::-1], T_k]))
    model = Tangent(1)
    model.plot_hull(solution)
    
    # n = 2
    model = Rejection(2)
    T_k = model.placement()
    solution = np.sort(np.concatenate([-T_k[::-1], T_k]))
    model = Tangent(2)
    model.plot_hull(solution)
    
    # n = 3
    model = Rejection(3)
    T_k = model.placement()
    solution = np.sort(np.concatenate([-T_k[::-1], T_k]))
    model = Tangent(3)
    model.plot_hull(solution)
    
    # n = 4
    model = Rejection(4)
    T_k = model.placement()
    solution = np.sort(np.concatenate([-T_k[::-1], T_k]))
    model = Tangent(4)
    model.plot_hull(solution)
    
    # n = 5
    model = Rejection(5)
    T_k = model.placement()
    solution = np.sort(np.concatenate([-T_k[::-1], T_k]))
    model = Tangent(5)
    model.plot_hull(solution)
