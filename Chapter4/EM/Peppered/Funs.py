# -*- coding: utf-8 -*-

import numpy as np


def Q_prime_m(n_C, n_I, n_T, n_U, p_C, p_I):
    """
    Calculate Q'(p|p^{(t)}) (manually)
    """
    N = n_C + n_I + n_T + n_U
    d_p_C = n_C / ((2 - p_C) * N)
    d_p_I = (n_I * (1 - p_C)) / (N * (2 - 2 * p_C - p_I)) + (n_U * p_I) / (N * (1 - p_C)) + (n_C * p_I) / (N * (2 - p_C))
    return np.array([d_p_C, d_p_I])

def Q_2prime_m(n_C, n_I, n_T, n_U, p_C, p_I):
    """
    Calculate Q''(p|p^{(t)}) (manually)
    """
    N = n_C + n_I + n_T + n_U
    term11 = 1 / p_C + 1 / (1 - p_C - p_I)
    term12 = 1 / (1 - p_C - p_I)
    term22 = 1 / p_I + 1 / (1 - p_C - p_I)
    return (-2 * N) * np.array([[term11,term12],
                              [term12, term22]])

def Q_m(n_C, n_I, n_T, n_U, p_C, p_I):
    """
    Calculate Q(p|p^{(t)}) (manually)
    """
    p_T, n_CC, n_CI, n_CT, n_II, n_IT, n_TT = relation_m(n_C, n_I, n_T, n_U, p_C, p_I)
    return n_CC * np.log(p_C ** 2) + n_CI * np.log(2 * p_C * p_I) + n_CT * np.log(2 * p_C * p_T) + n_II * np.log(p_I ** 2) + n_IT * np.log(2 * p_I * p_T) + n_TT * np.log(p_T ** 2)

def relation_m(n_C, n_I, n_T, n_U, p_C, p_I):
    """
    Calculate p_T, n_CC, n_CI, n_CT, n_II, n_IT, n_TT
    """
    p_T = 1 - p_C - p_I
    n_CC = (n_C * (p_C ** 2)) / ((p_C ** 2) + 2 * p_C * p_I + 2 * p_C * p_T)
    n_CI = (2 * n_C * p_C * p_I) / ((p_C ** 2) + 2 * p_C * p_I + 2 * p_C * p_T)
    n_CT = (2 * n_C * p_C * p_T) / ((p_C ** 2) + 2 * p_C * p_I + 2 * p_C * p_T)
    n_II = (n_I * (p_I ** 2)) / ((p_I ** 2) + 2 * p_I * p_T) + (n_U * (p_I ** 2)) / ((p_I ** 2) + 2 * p_I * p_T + (p_T ** 2))
    n_IT = (2 * n_I * p_I * p_T) / ((p_I ** 2) + 2 * p_I * p_T) + (2 * n_U * p_I * p_T) / ((p_I ** 2) + 2 * p_I * p_T + (p_T ** 2))
    n_TT = n_T + (n_U * (p_T ** 2)) / ((p_I ** 2) + 2 * p_I * p_T + (p_T ** 2))
    return p_T, n_CC, n_CI, n_CT, n_II, n_IT, n_TT