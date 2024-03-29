# Sample implementation of Polynomial interpolation methods

import numpy as np
from polyinterp import lagrange, newtonlagrange, hermite, chebyshev, legendre

def f(x):
    return np.e**(3*x)*np.sin(2*np.pi*x)

def df(x):
    return np.e**(3*x)*(3*np.sin(2*np.pi*x) + 2*np.pi*np.cos(2*np.pi*x))

def g(x):
    return 1./(1. + x**2)

x = [1./3.*i for i in range(4)]
print("Interpolation of exp(3x)sin(2pix) in the interval [0, 1] with 4 ",
      "equally-spaced nodes:")
print(f"Via Lagrange:\t\t", lagrange(f, x))
print(f"Via Newton-Lagrange:\t", newtonlagrange(f, x))

x = [1./4.*i for i in range(5)]
print("Interpolation of exp(3x)sin(2pix) in the interval [0, 1] with 5 ",
      "equally-spaced nodes:")
print(f"Via Lagrange:\t\t", lagrange(f, x))
print(f"Via Newton-Lagrange:\t", newtonlagrange(f, x))

print("Interpolation of exp(3x)sin(2pix) in the interval [0, 1] with 2 ",
      "equally-spaced nodes:")
print(f"Via Hermite with 2 nodes:\t\t", hermite(f, df, [0., 1.]))
print(f"Via Hermite with 3 nodes:\t\t", hermite(f, df, [0., 0.5, 1.]))

print("Interpolation of 1/(1 + x^2) in the interval [-5, 5]\n",
      "with nodes:")
print("Via Chebyshev:\t", chebyshev(g, 15, -5, 5))
print("Via Legendre:\t", legendre(g, 15, -5, 5))

