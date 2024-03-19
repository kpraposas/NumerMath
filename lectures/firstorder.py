""" Implementation of first-order methods. """

import numpy as np
from rootscalar import chord, secant, regfalsi, newtonraphson, steffensen, param

def f(x):
    return 0.25*np.cos(2*x)**2 - x**2

def g(x):
    return -np.sin(2.0*x)*np.cos(2.0*x) - 2*x

parameter = param()
parameter.maxit = 100
parameter.tol = 1e-15

Chord = chord(f, 0., 0.5, 1., parameter)
Secant = secant(f, 0., 0.5, parameter)
RegulaFalsi = regfalsi(f, 0., 0.5, parameter)
NewtonRaphson = newtonraphson(f, g, 0.5, None, parameter)
Inexactcenter = newtonraphson(f, None, 0.5, "center",
                              parameter)
Steffensen = steffensen(f, 0.5, parameter)

def takeNumit(result):
    return result.numit

method = [Chord, Secant, RegulaFalsi, NewtonRaphson, Inexactcenter, Steffensen]
method.sort(reverse=True, key=takeNumit)

print('-'*94)    
print("{}\t\t{}\t{}\t\t{}\t\t\t{}".format('METHOD', 'APPROXIMATE ROOT',
    'FUNCTION VALUE', 'ERROR', 'NITERS'))
print('-'*94)
for i in range(0, 6):
    data = [method[i].method_name, method[i].x, method[i].funval,
        method[i].error, method[i].numit]
    print("{:<16}{:.17}\t{:+.12e}\t{:.12e}\t{}".format(data[0], data[1],
        data[2], data[3], data[4]))
print('-'*94)