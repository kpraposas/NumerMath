""" Sample implementaion of the bisection method. """

import numpy as np
from rootscalar import rootscalar, param

def f(x):
    return 0.25*np.cos(2*x)**2 - x**2

options = {"method" : "bisection"}

parameter = param()
parameter.maxit = 100
parameter.tol = 1e-15

result = rootscalar(f, None, 0., 1., None, options, parameter)
print(result)

