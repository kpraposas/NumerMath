""" Sample implementaion of the modified dekker-brent method. """

import numpy as np
from rootscalar import dekkerbrent, param

def f(x):
    return 0.25*np.cos(2*x)**2 - x**2

parameter = param()
parameter.maxit = 100
parameter.tol = 1e-15

result = dekkerbrent(f, 0., 1., parameter)
print(result)

