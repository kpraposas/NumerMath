""" Implementation of acceleration of newtonraphson methods. """

import numpy as np
from rootscalar import rootscalar, param, options

def f(x):
    return (x*x - 1)**7 * np.log(x)

def df(x):
    return (x*x - 1)**6 * (14 * x * np.log(x + np.finfo(float).eps) + (x*x - 1) / x)

parameter = param()
parameter.maxit = 1000
parameter.tol = 1e-10

options = options

newton = rootscalar(f, df, None, None, 0.8, None, options, parameter=parameter)
modnewton = rootscalar(f, df, None, None, 0.8, 8, options=dict({"method" : "modnewton"}), parameter=parameter)
adaptivenewton = rootscalar(f, df, None, None, 0.8, None, options=dict({"method" : "adaptivenewton"}), parameter=parameter)

def takeNumit(result):
    return result.numit

method = [newton, modnewton, adaptivenewton]
method.sort(reverse=True, key=takeNumit)
    
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
