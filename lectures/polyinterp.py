"Module for polynomial interpolations"

from linalg import vector, matrix, solveLU
from rootspoly import param, newtonhorner
import numpy as np

class polynomial(list):
    """
    Class of polynomials where polynomial of the form
        a0 + a1x + a2x^2 + ... + anx^2
        is stored in a list 
        [a0, a1, a2, ..., an]
    """
    def __init__(self, iterable=list()):
        if not iterable:
            super().__init__()
        else:
            super().__init__(iterable)
    
    def __setitem__(self, index, item):
        super().__setitem__(index, item)
        
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __str__(self):
        os = ""
        n = len(self)
        for k in range(n- 1, -1, -1):
            coeff = self[k]
            if coeff == 0.:
                continue
            elif coeff < 0.:
                coeff *= -1.
                if k == n - 1:
                    os += "-"
                else:
                    os += " - "
            else:
                if k < n - 1:
                    os += " + "
            if coeff != 1. or k == 0:
                os += "{:.30f}".format(coeff)
            if k == 1:
                os += "x"
            if k > 1:
                os += f"x^{k}"
        return os

    def __add__(self, other):
        n, m = len(self), len(other)
        deg = max(n, m)
        C = polynomial()
        for i in range(deg):
            if i < n:
                a = self[i]
            else:
                a = 0.
            if i < m:
                b = other[i]
            else:
                b = 0
            C.append(a + b)
        return C
    
    __radd__ = __add__
    
    def __neg__(self):
        n = len(self)
        other = polynomial()
        for i in range(n):
            other.append(-self[i])
        return other

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other - self
    
    def __mul__(self, other):
        if type(other) == int or type(other) == float:
            n = len(self)
            C = polynomial()
            for i in range(n):
                C.append(self[i]*other)
            return C
        elif type(other) == polynomial:
            C = polynomial()
            n = len(self) + len(other) - 2
            for k in range(n + 1):
                c = 0
                for j in range(k + 1):
                    if j < len(self) and k - j < len(other):
                        c += self[j]*other[k - j]
                C.append(c)
            return C
        else:
            return NotImplemented

    def __rmul__(self, other):
        if type(other) == int or type(other) == float:
            return self*other
        else:
            return NotImplemented
    
    def __truediv__(self, other):
        if type(other) == int or type(other) == float:
            return self*(1./other)
        else:
            return NotImplemented

def undeterminedcoeff(f: callable, x: vector) -> polynomial:
    """
    Finds the coefficients of the interpolating polynomial on f
    given the interpolation nodes x by the Method of
    Undetermined Coefficientes.

    Parameters
    ----------
    f : callable
        objective real valued function
    x : vector
        nodes to interpolate function at

    Returns
    -------
    list
        coefficients of interpolating polynomial
    """
    n = len(x)
    V = matrix([[0]*n for _ in range(n)])
    for i in range(n):
        v = 1.
        for j in range(n):
            V[i][j] = v
            v = v*x[i]
    b = vector()
    for i in range(n):
        b.append(f(x[i]))
    return polynomial(solveLU(V, b).x)

def lagrange(f: callable, x: vector) -> polynomial:
    """
    Lagrange polynomial interpolation of f at nodes x

    Parameters
    ----------
    f : callable
        objective real valued function
    x : vector
        nodes to interpolate function at

    Returns
    -------
    polynomial
        interpolating polynomial of f at nodes x
    """
    n = len(x)
    p = polynomial([0])
    for k in range(n):
        l = polynomial([1])
        for i in range(n):
            if i == k:
                continue
            temp = polynomial([-x[i], 1])
            temp = temp / (x[k] - x[i])
            l = l * temp
        p = p + f(x[k])*l 
    return p

def newtonlagrange(f: callable, x: vector) -> polynomial:
    """
    Newton-Lagrange polynomial interpolation of f at nodes x

    Parameters
    ----------
    f : callable
        objective real valued function
    x : vector
        nodes to interpolate function at

    Returns
    -------
    polynomial
        
    """
    n = len(x)
    D = matrix([[0]*n for _ in range(n)])
    for k in range(n):
        D[k][0] = f(x[k])
    for k in range(1 , n):
        for l in range(k, n):
            D[l][k] = (D[l][k - 1] - D[l - 1][k - 1]) / (x[l] - x[l - k])
    p = polynomial([D[n - 1][n - 1]])
    for k in range(1, n):
        temp = polynomial([-x[n - k - 1], 1])
        p = p*temp + polynomial([D[n - k - 1][n - k - 1]])
    return p

def hermite(f: callable, df: callable, x: vector) -> polynomial:
    """
    Hermite polynomial interpolation of f at nodes x

    Parameters
    ----------
    f : callable
        objective real valued function
    df : callable
        derivative of f
    x : vector
        nodes to interpolate function at

    Returns
    -------
    polynomial
        interpolating polynomial of f at nodes x
    """
    n = len(x)
    p = polynomial([0])
    for k in range(n):
        delta = 1
        nu = polynomial([1])
        ell1 = 0
        for i in range(n):
            if i == k:
                continue
            nu = nu * polynomial([-x[i], 1])
            delta = delta * (x[k] - x[i])
            ell1 = ell1 + 1. / (x[k] - x[i])
        ell2 = (nu/delta)*(nu/delta)
        eta = polynomial([-x[k], 1]) * ell2 
        h = (polynomial([1]) - 2.*ell1*polynomial([-x[k], 1])) * ell2
        p = p + f(x[k]) * h + df(x[k]) * eta
    return p

def chebyshev(f: callable, n: int, a: float=0., b: float=1.) -> polynomial:
    """
    Lagrange intepolating polynomial at the n affine-transformed Chebyshev points

    Parameters
    ----------
    f : callable
        objective real valued function
    n : int
        one less the number of chebyshev points
    a : float, optional
        left end point of the polynomial, by default 0.
    b : float, optional
        right end point of the polynomial, by default 1.

    Returns
    -------
    polynomial
        interpolating polynomial of f at n Chebyshev
    """
    x = vector()
    for k in range(n+1):
        temp = np.cos((k+0.5)*np.pi/(n+1.))
        temp = 0.5*((b - a)*temp + a + b)
        x.append(temp)
    return newtonlagrange(f, x)

def legendre_poly(n: int) -> polynomial:
    """
    Construct the nth Legendre Polynomial

    Parameters
    ----------
    n : int
        order of the Polynomial

    Returns
    -------
    polynomial
        nth order Legendre Polynomial
    """
    if n == 0:
        return polynomial([1.])
    if n == 1:
        return polynomial([0., 1.])
    return polynomial([0., (2. - 1./n)]) * legendre_poly(n - 1)\
        + polynomial([-1. + 1./n]) * legendre_poly(n - 2)

def legendre(f: callable, n: int, a: float=0., b: float=1.) -> polynomial:
    """
    Lagrange Interpolating polynomial at the n affine-transformed Legendre points

    Parameters
    ----------
    f : callable
        objective real valued function
    n : int
        one less the number of chebyshev points
    a : float, optional
        left end point of the polynomial, by default 0.
    b : float, optional
        right end point of the polynomial, by default 1.

    Returns
    -------
    polynomial
        interpolating polynomial of f at n legendre points
    """
    p = legendre_poly(n + 1)
    z = newtonhorner(p, a+b/2., param()).z
    x = [0.5*(b - a)*y.real + a + b for y in z]
    print(x)
    return newtonlagrange(f, x)