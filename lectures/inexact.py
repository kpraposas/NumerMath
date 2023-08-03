""" Implementation of first-order inexact newtonraphson methods. """

import numpy as np
import rootscalar

def f(x):
    return 0.25*np.cos(2*x)**2 - x**2

parameter = rootscalar.param()
parameter.maxit = 100
parameter.tol = 1e-15

Inexactcenter = rootscalar.inexactcenter(f, 0.5, parameter)
Inexactforward = rootscalar.inexactforward(f, 0.5, parameter)
Inexactbackward = rootscalar.inexactbackward(f, 0.5, parameter)

method = [Inexactcenter, Inexactforward, Inexactbackward]
for i in range(0, 3):
    name = ['CENTER', 'FORWARD', 'BACKWARD']
    method[i].method_name = name[i]
    
print('-'*86)    
print("{:<16}{:<22}{:<22}{:<20}{:<6}".format('METHOD', 'APPROXIMATE ROOT',
    'FUNCTION VALUE', 'ERROR', 'NITERS'))
print('-'*86)
for i in range(0, 3):
    data = [method[i].method_name, method[i].x, method[i].funval,
        method[i].error, method[i].numit]
    print("{:<16}{:.17}   {:+.12e}   {:.12e}   {}".format(data[0], data[1],
        data[2], data[3], data[4]))
print('-'*86)