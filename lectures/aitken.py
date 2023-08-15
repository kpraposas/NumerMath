""" Sample implementaion of the aitken method. """

import numpy as np
from rootscalar import rootscalar, param, options

def g(x):
    return 0.5*np.cos(2*x)

parameter = param()
parameter.maxit = 100
parameter.tol = 1e-15

options = options
options=dict({"method" : "aitken"})

result = rootscalar(g, None, None, None, 0., None, options, parameter)
print(result)

