""" Implementation of Muller methods and its generalizations. """

import numpy as np
from rootscalar import rootscalar, param

def f(x):
    return 0.25*np.cos(2*x)**2 - x**2

parameter = param()
parameter.maxit = 100
parameter.tol = 1e-15

Muller = rootscalar(f, None, None, None, [0.25*k for k in range(3)], None, options=dict({"method" : "muller"}), parameter=parameter)
RootPolyInterp = rootscalar(f, None, None, None, [0.1*(k - 1) for k in range(4)], None, options=dict({"method" : "rootpolyinterp"}), parameter=parameter)

print(RootPolyInterp.x)

method = [Muller, RootPolyInterp]
    
print('-'*86)    
print("{:<16}{:<22}{:<22}{:<20}{:<6}".format('METHOD', 'APPROXIMATE ROOT',
    'FUNCTION VALUE', 'ERROR', 'NITERS'))
print('-'*86)
for i in range(len(method)):
    data = [method[i].method_name, method[i].x, method[i].funval,
        method[i].error, method[i].numit]
    print("{:<16}{:.17}   {:+.12e}   {:.12e}   {}".format(data[0], data[1],
        data[2], data[3], data[4]))
print('-'*86)