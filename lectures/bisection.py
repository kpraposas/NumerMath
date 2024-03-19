""" Sample implementaion of the bisection method. """

from numpy import cos
from rootscalar import bisection, param

def f(x):
    return 0.25*cos(2*x)**2 - x**2

parameter = param()
parameter.maxit = 100
parameter.tol = 1e-15

result = bisection(f, 0., 1., parameter)
print(result)