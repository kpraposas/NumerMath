""" Implementation of first-order methods. """

import numpy as np
from rootscalar import rootscalar, param

def f(x):
    return 0.25*np.cos(2*x)**2 - x**2

def g(x):
    return -np.sin(2.0*x)*np.cos(2.0*x) - 2*x

parameter = param()
parameter.maxit = 100
parameter.tol = 1e-15

Chord = rootscalar(f, None, 0.0, 0.5, 1.0, options=dict({"method" : "chord"}), parameter=parameter)
Secant = rootscalar(f, None, None, None, [0.5, 0.0], options=dict({"method" : "secant"}), parameter=parameter)
RegulaFalsi = rootscalar(f, None, None, None, [0.5, 0.0], options=dict({"method" : "regfalsi"}), parameter=parameter)
NewtonRaphson = rootscalar(f, g, None, None, 0.5, options=dict({"method" : "newton"}), parameter=parameter)
Inexactcenter = rootscalar(f, None, None, None, 0.5, options=dict({"method" : "newton", "inexact" : "center"}), parameter=parameter)
Steffensen = rootscalar(f, None, None, None, 0.5, options=dict({"method" : "steffensen"}), parameter=parameter)

def takeNumit(result):
    return result.numit

method = [Chord, Secant, RegulaFalsi, NewtonRaphson, Inexactcenter, Steffensen]
method.sort(reverse=True, key=takeNumit)

print('-'*86)    
print("{:<16}{:<22}{:<22}{:<20}{:<6}".format('METHOD', 'APPROXIMATE ROOT',
    'FUNCTION VALUE', 'ERROR', 'NITERS'))
print('-'*86)
for i in range(0, 6):
    data = [method[i].method_name, method[i].x, method[i].funval,
        method[i].error, method[i].numit]
    print("{:<16}{:.17}   {:+.12e}   {:.12e}   {}".format(data[0], data[1],
        data[2], data[3], data[4]))
print('-'*86)