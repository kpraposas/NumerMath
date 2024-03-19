""" Implementation of first-order inexact newtonraphson methods. """

import numpy as np
from rootscalar import newtonraphson, param

def f(x):
    return 0.25*np.cos(2*x)**2 - x**2

parameter = param()
parameter.maxit = 100
parameter.tol = 1e-15

center = newtonraphson(f, None, 0.5, "center", parameter)
forward = newtonraphson(f, None, 0.5, "forward", parameter)
backward = newtonraphson(f, None, 0.5, "backward", parameter)

method = [center, forward, backward]
for i in range(3):
    name = ['CENTER', 'FORWARD', 'BACKWARD']
    method[i].method_name = name[i]
    
print('-'*94)    
print("{}\t\t{}\t{}\t\t{}\t\t\t{}".format('METHOD', 'APPROXIMATE ROOT',
    'FUNCTION VALUE', 'ERROR', 'NITERS'))
print('-'*94)
for i in range(0, 3):
    data = [method[i].method_name, method[i].x, method[i].funval,
        method[i].error, method[i].numit]
    print("{:<16}{:.17}\t{:+.12e}\t{:.12e}\t{}".format(data[0], data[1],
        data[2], data[3], data[4]))
print('-'*94)