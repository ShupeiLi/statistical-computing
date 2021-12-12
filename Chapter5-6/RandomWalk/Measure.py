# -*- coding: utf-8 -*-

import numpy as np


def D_x(x):
    """
    Manhattan distance.
    """
    current = x[x.shape[0] - 1, :]
    return np.sum(np.abs(current))

def R_x(x):
    """
    Counts the number of visits to the current location.
    """
    current = x[x.shape[0] - 1, :]
    r = 0
    for i in range(x.shape[0]):
        if (x[i, 0] == current[0]) and (x[i, 1] == current[1]):
            r += 1
    return r