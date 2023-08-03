""" Implementation of acceleration of newtonraphson methods. """

import numpy as np
import rootscalar

def f(x):
    return (x*x - 1)**7 * np.log(x)

def df(x):
    return (x*x - 1)**6 * (14 * x * np.log(x + np.finfo(float).eps) + (x*x - 1) / x)

parameter = rootscalar.param()
parameter.maxit = 1000
parameter.tol = 1e-10

newton = rootscalar.newtonraphson(f, df, 0.8, parameter)
modnewton = rootscalar.modnewton(f, df, 0.8, 8, parameter)
adaptivenewton = rootscalar.adaptivenewton(f, df, 0.8, parameter, 1, 1e-3, 1e-2)

method = [newton, modnewton, adaptivenewton]
for i in range(0, 3):
    name = ['Newton', 'ModifiedNewton', 'AdaptiveNewton']
    method[i].method_name = name[i]
    
print('-'*90)    
print("{:<16}{:<24}{:<24}{:<20}{:<6}".format('METHOD', 'APPROXIMATE ROOT',
    'FUNCTION VALUE', 'ERROR', 'NITERS'))
print('-'*90)
for i in range(0, 3):
    data = [method[i].method_name, method[i].x, method[i].funval,
        method[i].error, method[i].numit]
    print("{:<16}{:0.17f}\t{:+.12e}\t{:.12e}  {}".format(data[0], data[1],
        data[2], data[3], data[4]))
print('-'*90)
