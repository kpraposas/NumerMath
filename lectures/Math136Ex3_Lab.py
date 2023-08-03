"""
Math 136 Exercise 2
Name: Krizelda Claire V. Cayabyab
      Kenneth P. Raposas
      Hannah Denise C. Sarilla
Date: 10 June 2022
"""
import numpy as np

# An, bn definition
def An(n):
    An = [[0]*n for i in range(n)]
    for i in range(n):
        An[i][i] = 2
    for i in range(n):
        for j in range(n):
            if i == j + 1 or j == i + 1:
                An[i][j] = -1
    for i in range(n):
        for j in range(n):
            An[i][j] *= (n + 1)**2
    return An

def bn(n):
    bn = []
    for i in range(1, n+1):
        bn.append(np.sin(i*np.pi/(n + 1)) * np.pi**2)
    return bn

# Vector and matrix operations
def vectordiff(A, B):
    n = len(A)
    C = [0]*n
    for i in range(n):
        C[i] = A[i] - B[i]
    return C

def matrixvectorSAXPY(A, x):
    m = len(A)
    n = len(A[0])
    y = [0] * m
    for j in range(0, n):
        for i in range(0, m):
            y[i] += A[i][j] * x[j]
    return y

def scalarvectorprod(A, x):
    n = len(A)
    for i in range(n):
        A[i] *= x
    return A

# Norm definitions
def Euclidnorm(X):
    n = len(X)
    sum = 0
    for i in range(n):
        sum += X[i]**2
    return np.sqrt(sum)

def Maxnorm(X):
    n = len(X)
    for i in range(n):
        X[i] = abs(X[i])
    return max(X)

# LU factorization
def forwardsubcol(L, b):
    n = len(b)
    for j in range(0, n - 1):
        b[j] = b[j] / L[j][j]
        for i in range(j + 1, n):
            b[i] = b[i] - L[i][j] * b[j]
    b[n - 1] = b[n - 1] / L[n - 1][n - 1]
    return b

def backwardsubcol(U, b):
    n = len(b)
    for j in range(n - 1, 0, -1):
        b[j] = b[j] / U[j][j]
        for i in range(0, j):
            b[i] = b[i] - U[i][j] * b[j]
    b[0] = b[0] / U[0][0]
    return b

def LUijk(A):
    n = len(A)
    for j in range(1, n):
        A[j][0] = A[j][0] / A[0][0]
    for i in range(1, n):
        for j in range(i, n):
            s = 0
            for k in range(0, i):
                s += A[i][k] * A[k][j]
            A[i][j] = A[i][j] - s
        for j in range(i + 1, n):
            s = 0
            for k in range(0, i):
                s += A[j][k] * A[k][i]
            A[j][i] = (A[j][i] - s) / A[i][i]
    return A

def getLUijk(A):
    n = len(A)
    A = LUijk(A)
    L = [[0] * n for i in range (0, n)]
    U = [[0] * n for i in range (0, n)]
    for i in range(0, n):
        L[i][i] = 1
        for j in range(0, i):
            L[i][j] = A[i][j]
        for j in range(i, n):
            U[i][j] = A[i][j]
    return L, U

def LUijk_Solve(A, b):
    L, U = getLUijk(A)
    y = forwardsubcol(L, b)
    x = backwardsubcol(U, y)
    return x

# Over-relaxation
def JOR(A, b, x, tol, maxit, omega):
    n = len(A)
    b_norm = Euclidnorm(b)
    Ax = matrixvectorSAXPY(A, x)
    err = Euclidnorm(vectordiff(b, Ax)) / b_norm
    k = 0
    while err > tol and k < maxit:
        for i in range(0, n):
            x_old = x
            s = 0
            for j in range(0, n):
                if j != i:
                    s = s + A[i][j] * x_old[j]
            x[i] = omega * (b[i] - s) / A[i][i] + (1 - omega) * x_old[i]
        Ax = matrixvectorSAXPY(A, x)
        err = Euclidnorm(vectordiff(b, Ax)) / b_norm
        k += 1
    if err > tol and k == maxit:
        pass
    return x, k

def SOR(A, b, x, tol, maxit, omega):
    n = len(b)
    b_norm = Euclidnorm(b)
    Ax = matrixvectorSAXPY(A, x)
    err = Euclidnorm(vectordiff(b, Ax)) / b_norm
    k = 0
    while err > tol and k < maxit:
        for i in range(0, n):
            x_old = x
            s = 0
            for j in range(0, i):
                s = s + A[i][j] * x[j]
            for j in range(i + 1, n):
                s = s + A[i][j] * x_old[j]
            x[i] = omega * (b[i] - s) / A[i][i] + (1 - omega) * x_old[i]
        Ax = matrixvectorSAXPY(A, x)
        err = Euclidnorm(vectordiff(b, Ax)) / b_norm
        k += 1
    if err > tol and k == maxit:
        pass
    return x, k

# Result

print("> Direct Methods")
print('-'*69)
print("{:<15}{:<6}{:<27}{:28}".format('METHOD', 'n', 'RESIDUAL MAX NORM', 'ERROR NORM'))
print('-'*69)
for n in [10, 50, 250]:
    result = LUijk_Solve(An(n), bn(n))
    max_norm = Maxnorm(vectordiff(bn(n), matrixvectorSAXPY(An(n), result)))
    error_norm = (n + 1)**-1 * Maxnorm(vectordiff(result, scalarvectorprod(bn(n), np.pi**-2)))
    print("{:<15}{:<6}{:.15e}\t{:.15e}".format('LUSolve - IJK', n, max_norm, error_norm))
print('-'*69, '\n')

print("> Over-Relaxation Methods")
print('-'*80)
print("{:<8}{:<6}{:<8}{:<10}{:<27}{:<28}".format('METHOD', 'n', 'OMEGA', 'NUMIT', 'RESIDUAL MAX NORM', 'ERROR NORM'))
print('-'*80)
for n in [10, 50, 250]:
    for omega in [0.5, 1.0, 1.5]:
        x = [0] * n
        tol = 1e-12
        maxit = 10000
        result, numit = JOR(An(n), bn(n), x, tol, maxit, omega)
        max_norm = Maxnorm(vectordiff(bn(n), matrixvectorSAXPY(An(n), result)))
        error_norm = (n + 1)**-1 * Maxnorm(vectordiff(result, scalarvectorprod(bn(n), np.pi**-2)))
        print("{:<8}{:<6}{:<8}{:<10}{:.15e}\t   {:.15e}".format('JOR', n, omega, numit, max_norm, error_norm))
for n in [10, 50, 250]:
    for omega in [0.5, 1.0, 1.5]:
        x = [0] * n
        tol = 1e-12
        maxit = 10000
        result, numit = SOR(An(n), bn(n), x, tol, maxit, omega)
        max_norm = Maxnorm(vectordiff(bn(n), matrixvectorSAXPY(An(n), result)))
        error_norm = (n + 1)**-1 * Maxnorm(vectordiff(result, scalarvectorprod(bn(n), np.pi**-2)))
        print("{:<8}{:<6}{:<8}{:<10}{:.15e}\t   {:.15e}".format('SOR', n, omega, numit, max_norm, error_norm))
print('-'*80)