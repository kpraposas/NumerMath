""" Implementation of first-order inexact newtonraphson methods. """

import numpy as np
from rootscalar import rootscalar, param

def f(x):
    return 0.25*np.cos(2*x)**2 - x**2

parameter = param()
parameter.maxit = 100
parameter.tol = 1e-15

Inexactcenter = rootscalar(f, None, None, None, 0.5, None, options=dict({"method" : "newton", "inexact" : "center"}), parameter=parameter)
Inexactforward = rootscalar(f, None, None, None, 0.5, None, options=dict({"method" : "newton","inexact" : "forward"}), parameter=parameter)
Inexactbackward = rootscalar(f, None, None, None, 0.5, None, options=dict({"method" : "newton","inexact" : "backward"}), parameter=parameter)

method = [Inexactcenter, Inexactforward, Inexactbackward]
for i in range(3):
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