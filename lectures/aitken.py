""" Sample implementaion of the aitken method. """

import numpy as np
import rootscalar

def g(x):
    return 0.5*np.cos(2*x)

parameter = rootscalar.param()
parameter.maxit = 100
parameter.tol = 1e-15

result = rootscalar.aitken(g, 0., parameter)
print(result)

