# -*- coding: utf-8 -*-

from Case1 import Case1
from Case2 import Case2
from SAW import SAW
from SAWpost import SAWpost
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # a.
    model = Case1(30)
    e_d, e_r = model.estimate_mean()
    sd_d, sd_r = model.estimate_sd()
    
    # b.
    model = Case2(30)
    e_d, e_r = model.estimate_mean()
    sd_d, sd_r = model.estimate_sd()
    
    model = Case2(30, t = 100)
    x_t, w_t = model.sis()
    plt.plot(x_t[:, 0], x_t[:, 1])
    plt.show()
    
    # c.
    model = SAW(30)
    e_d, e_r = model.estimate_mean()
    sd_d, sd_r = model.estimate_sd()
    
    model = SAW(30, t = 100)
    x_t, w_t = model.sis()
    plt.plot(x_t[:, 0], x_t[:, 1])
    plt.show()

    # d.
    model = SAWpost(30, t = 100)
    x_t, iteration = model.path_generator()
    print("Iteration: " + str(iteration))
    plt.plot(x_t[:, 0], x_t[:, 1])
    plt.show()