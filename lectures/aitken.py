""" Sample implementaion of the aitken method. """

import numpy as np
from rootscalar import aitken, param

def g(x):
    return 0.5*np.cos(2*x)

parameter = param()
parameter.maxit = 100
parameter.tol = 1e-15

result = aitken(g, 0., parameter)
print(result)

