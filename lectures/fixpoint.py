""" Sample implementaion of the fix point method. """

import numpy as np
from rootscalar import fixpoint, param

def g(x):
    return 0.5*np.cos(2*x)

parameter = param()
parameter.maxit = 100
parameter.tol = 1e-15

result = fixpoint(g, 0., parameter)
print(result)

