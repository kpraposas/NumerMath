""" Sample implementaion of the modified dekker-brent method. """

import numpy as np
from rootscalar import rootscalar, param, options

def f(x):
    return 0.25*np.cos(2*x)**2 - x**2

parameter = param()
parameter.maxit = 100
parameter.tol = 1e-15

options = options
options["method"] = "dekker"

result = rootscalar(f, None, 0., 1., None, None, options, parameter)
print(result)

