# -*- coding: utf-8 -*-

from Tangent import Tangent

if __name__ == '__main__':
    # n = 1
    model = Tangent(1)
    T_k = model.tangent_based()
    model.plot_hull(T_k)
    
    # n = 2
    model = Tangent(2)
    T_k = model.tangent_based()
    model.plot_hull(T_k)
    
    # n = 3
    model = Tangent(3)
    T_k = model.tangent_based()
    model.plot_hull(T_k)
    
    # n = 4
    model = Tangent(4)
    T_k = model.tangent_based()
    model.plot_hull(T_k)
    
    # n = 5
    model = Tangent(5)
    T_k = model.tangent_based()
    model.plot_hull(T_k)