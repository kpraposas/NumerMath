"Module for polynomial interpolations"

import numpy as np

def LagrangeInterp(f, x, z):
    """
    
    """
    n = len(x)
    m = len(z)
    v = []
    for j in range(m):
        v.append(0)
        for k in range(n):
            l = 1
            for i in range(n):
                if i == k: continue
                l = l * (z[j] - x[i]) / (x[k] - x[i])
            v[j] = v[j] + f(x[k]) * l
    return v


def NewtonLagrangeInterp(f, x, z):
    """
    
    """
    n = len(x)
    m = len(z)
    v = [0] * m
    D = [[0] * n for i in range(n)]
    for k in range(n):
        D[k][0] = f(x[k])
    for k in range(1 , n):
        for l in range(k, n):
            D[l][k] = (D[l][k - 1] - D[l - 1][k - 1]) / (x[l] - x[l - k])
    for j in range(m):
        v[j] = D[n - 1][n - 1]
        for k in range(1, n):
            v[j] = v[j] * (z[j] - x[n - k]) + D[n - k][n - k]
    return v


def HermiteInterp(f, df, x, z):
    """
    
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
                if i == k: continue
                nu = nu * (z[j] - x[i])
                delta = delta * (x[k] - x[i])
                l1 = (l1 + 1) / (x[k] - x[i])
            l2 = (nu / delta)**2
            eta = (z[j] - x[k]) * l2 
            h = (1 - 2 * l1(z[j] - x[k])) * l2
            v[j] = v[j] + f(x[k]) * h + df(x[k]) * eta
    return v