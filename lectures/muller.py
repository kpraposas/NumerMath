""" Implementation of Muller methods and its generalizations. """

import numpy as np
from rootscalar import muller, rootpolyinterp, sidisecant, param

def f(x):
    return 0.25*np.cos(2*x)**2 - x**2

parameter = param()
parameter.maxit = 100
parameter.tol = 1e-15

Muller = muller(f, 0., -0.25, -0.5, parameter)
RootPolyInterp = rootpolyinterp(f, [0.1*_ for _ in range(4)], parameter)
SidiSecant = sidisecant(f, [0.1*_ for _ in range(4)], parameter)

method = [Muller, RootPolyInterp, SidiSecant]
    
print('-'*86)    
print("{}{}{}{}{}".format('METHOD', 'APPROXIMATE ROOT',
    'FUNCTION VALUE', 'ERROR', 'NITERS'))
print('-'*86)
for i in range(len(method)):
    data = [method[i].method_name, method[i].x, method[i].funval,
        method[i].error, method[i].numit]
    print("{:<16}{:.17}{:+.12e}{:.12e}{}".format(data[0], data[1],
        data[2], data[3], data[4]))
print('-'*86)