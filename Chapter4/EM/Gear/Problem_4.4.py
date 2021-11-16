# -*- coding: utf-8 -*-

import sys
sys.path.append(r".\Chapter4\EM\Gear")
from FamilyEM import FamilyEM

a = 0.003
b = 2.5
epsilon = 10 ** (-6)
m = 10000
max_iter = 500


if __name__ == '__main__':
    model = FamilyEM(a, b, epsilon, m, max_iter, False)
    a_history1, b_history1, iteration1 = model.em_gear()
    
    model = FamilyEM(a, b, epsilon, m, max_iter)
    a_history2, b_history2, iteration2 = model.em_gear()
    
    model = FamilyEM(a, b, epsilon, m, max_iter, False, True)
    a_history3, b_history3, iteration3 = model.em_gear()