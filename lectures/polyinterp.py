"Module for polynomial interpolations"

from linalg import vector, matrix, solveLU
import numpy as np

def undeterminedcoeff(f: callable, x: vector) -> list:
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
    return solveLU(V, b).x


def lagrange(f, x, z):
    """
    Lagrange polynomial interpolation of f at nodes x evaluated at points z
    
    Attributes
    ----------
        f : callable
            objective real valued function
        x : list
            nodes to interpolate function at
        z : list
            values of Lagrange interpolation polynomial
    """
    n = len(x)
    m = len(z)
    v = []
    for j in range(m):
        val = 0
        for k in range(n):
            l = 1
            for i in range(n):
                if i == k:
                    continue
                l = l * (z[j] - x[i]) / (x[k] - x[i])
            val = val + f(x[k]) * l
        v.append(val)
    return v


def newtonlagrange(f, x, z):
    """
    Newton-Lagrange polynomial interpolation of f at nodes x evaluated at points z
    
    Attributes
    ----------
        f : callable
            objective real valued function
        x : list
            nodes to interpolate function at
        z : list
            values of Newton-Lagrange interpolation polynomial
    """
    n = len(x)
    m = len(z)
    v = [0] * m
    D = [[0] * n for _ in range(n)]
    for k in range(n):
        D[k][0] = f(x[k])
    for k in range(1 , n):
        for l in range(k, n):
            D[l][k] = (D[l][k - 1] - D[l - 1][k - 1]) / (x[l] - x[l - k])
    for j in range(m):
        v[j] = D[n - 1][n - 1]
        for k in range(1, n):
            v[j] = v[j] * (z[j] - x[n - k - 1]) + D[n - k - 1][n - k - 1]
    return v


def hermite(f, df, x, z):
    """
    Hermite polynomial interpolation of f at nodes x evaluated at points z
    
    Attributes
    ----------
        f : callable
            objective real valued function
        df : callable
            derivative of f
        x : list
            nodes to interpolate function at
        z : list
            values of Hermite interpolation polynomial
    """
    n = len(x)
    m = len(z)
    v = [0] * m
    for j in range(m):
        for k in range(n):
            delta = 1
            nu = 1
            l1 = 0
            for i in range(n):
                if i == k:
                    continue
                nu = nu * (z[j] - x[i])
                delta = delta * (x[k] - x[i])
                l1 = l1 + 1 / (x[k] - x[i])
            l2 = (nu / delta)**2
            eta = (z[j] - x[k]) * l2 
            h = (1 - 2 * l1*(z[j] - x[k])) * l2
            v[j] = v[j] + f(x[k]) * h + df(x[k]) * eta
    return v


def piecewise(f, x, z, p):
    """
    Piecewise Lagrange polynomial interpolation of f at nodes x evaluated at points z
        where p is the degree of each interpolating polynomial at each interval
    
    Attributes
    ----------
        f : callable
            objective real valued function
        x : list
            nodes to interpolate function at
        z : list
            values of Lagrange interpolation polynomial
        p : int
            degree of interpolating polynomial
    """
    n = len(x)
    m = len(z)
    v = []
    j = 0
    for i in range(n - p):
        xi = [x[i + _] for _ in range(p + 1)]
        zi = []
        if z[j] < xi[0]:
            j += 1
        while z[j] <= xi[-1]:
            zi.append(z[j])
            j += 1
            if j == m:
                break
        print(len(zi))
        v.extend(lagrange(f, xi, zi))
    return v
    