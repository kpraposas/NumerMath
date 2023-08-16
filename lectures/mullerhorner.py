import numpy as np
import matplotlib.pyplot as plt
from rootspoly import mullerhorner, horner, param

def pn(n):
    pn = []
    for i in range(0, n):
        pn.append(-1)
    pn.append(1)
    return pn

def modulus(z):
    modulus = np.sqrt(z.real**2 + z.imag**2)
    return modulus

parameter = param()
parameter.maxit = 100
parameter.refmax = 100
parameter.tol = 10**3 * np.finfo(float).eps
parameter.reftol = 1e-3
parameter.ref = False

for j in [1, 2, 4, 8, 10]:
    n = 5 * j
    print("\n> Degree", n)
    print('-'*97)
    print("{:<28}\t{:<28}\t{:<12}\t{:<32}".format('REAL PART', 'IMAG PART', '|FUNVAL|', 'MODULUS'))
    print('-'*97)
    z = [-0.5*(k-1) for k in range(3)]
    degn_result = mullerhorner(pn(n), z, parameter).x
    degn_result_real = []
    degn_result_imag = []
    for i in range(0, n):
        degn_result_real.append(degn_result[i].real)
        degn_result_imag.append(degn_result[i].imag)
        funval = abs(horner(pn(n), degn_result[i])[0]) / np.finfo(float).eps
        print("{:+.15e}\t\t{:+.15e}\t\t{:.1f}eps\t\t{:.15f}".format(degn_result[i].real,
            degn_result[i].imag, funval, modulus(degn_result[i])))
    print('-'*97)


parameter.ref = True

for j in [1, 2, 4, 8, 10]:
    n = 5 * j
    print("\n> Degree", n)
    print('-'*97)
    print("{:<28}\t{:<28}\t{:<12}\t{:<32}".format('REAL PART', 'IMAG PART', '|FUNVAL|', 'MODULUS'))
    print('-'*97)
    z = [-0.5*(k-1) for k in range(3)]
    degn_result = mullerhorner(pn(n), z, parameter).x
    degn_result_real = []
    degn_result_imag = []
    for i in range(0, n):
        degn_result_real.append(degn_result[i].real)
        degn_result_imag.append(degn_result[i].imag)
        funval = abs(horner(pn(n), degn_result[i])[0]) / np.finfo(float).eps
        print("{:+.15e}\t\t{:+.15e}\t\t{:.1f}eps\t\t{:.15f}".format(degn_result[i].real,
            degn_result[i].imag, funval, modulus(degn_result[i])))
    print('-'*97)