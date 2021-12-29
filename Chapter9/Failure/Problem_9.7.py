# -*- coding: utf-8 -*-

from Failure import Failure


if __name__ == '__main__':
    # Cauchy(0, 1)
    model_cauchy = Failure(n=50, B=10000, seed=1234, distribution="cauchy")
    model_cauchy.bootstrap_summary()
    
    # Unif(0, 1)
    model_unif = Failure(n=50, B=10000, seed=1234, distribution="uniform")
    model_unif.bootstrap_summary()