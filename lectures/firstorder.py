""" Implementation of first-order methods. """

import numpy as np
import rootscalar

def f(x):
    return 0.25*np.cos(2*x)**2 - x**2

def g(x):
    return -np.sin(2.0*x)*np.cos(2.0*x) - 2*x

parameter = rootscalar.param()
parameter.maxit = 100
parameter.tol = 1e-15

Chord = rootscalar.chord(f, 0.0, 0.5, 1.0, parameter)
Secant = rootscalar.secant(f, 0.5, 0.0, parameter)
RegulaFalsi = rootscalar.regfalsi(f, 0.5, 0.0, parameter)
NewtonRaphson = rootscalar.newtonraphson(f, g, 0.5, parameter)
Inexactcenter = rootscalar.inexactcenter(f, 0.5, parameter)
Steffensen = rootscalar.steffensen(f, 0.5, parameter)

method = [Chord, Secant, RegulaFalsi, NewtonRaphson, Inexactcenter, Steffensen]

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