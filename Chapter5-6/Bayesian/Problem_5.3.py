# -*- coding: utf-8 -*-

from Verify import Verify
from Posterior import Posterior
from Improper import Improper
from Transformation import Transformation

if __name__ == '__main__':
    # a.
    model = Verify()
    k, iterations = model.riemann()
    print("k is: " + str(k))
    print("Iteration: " + str(iterations))
    
    # b.
    # Riemann
    model = Posterior()
    history_total, history_re = model.riemann()
    
    # Trapezoidal
    history_total, history_re = model.trapezoidal()
    
    # Simpson’s
    history_total, history_re = model.simpson()

    # c.
    # Riemann
    model = Improper()
    history_total, history_re = model.riemann()
    
    # Trapezoidal
    history_total, history_re = model.trapezoidal()
    
    # Simpson’s
    history_total, history_re = model.simpson()
    
    # d.
    # Riemann
    model = Transformation()
    history_total, history_re = model.riemann()
    
    # Trapezoidal
    history_total, history_re = model.trapezoidal()
    
    # Simpson’s
    history_total, history_re = model.simpson()