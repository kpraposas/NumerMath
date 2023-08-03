""" Sample implementaion of the bisection method. """

import numpy as np
import rootscalar

def f(x):
    return 0.25*np.cos(2*x)**2 - x**2

parameter = rootscalar.param()
parameter.maxit = 100
parameter.tol = 1e-15

result = rootscalar.bisection(f, 0., 1., parameter)
print(result)

