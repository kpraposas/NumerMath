import numpy as np
import matplotlib.pyplot as plt
from rootspoly import mullerhorner, param, eps

def pn(n):
    pn = []
    for _ in range(0, n):
        pn.append(-1)
    pn.append(1)
    return pn

parameter = param()
parameter.maxit = 100
parameter.refmax = 100
parameter.tol = 10**3 * eps
parameter.reftol = 1e-3
parameter.ref = False

print("\n> Degree", 5)
print('-'*100)
print("{}\t\t\t{}\t\t\t{}\t\t\t{}".format('REAL PART', 'IMAG PART', '|FUNVAL|',
                                          'ITER'))
print('-'*100)
z = [0.5*(k-1) for k in range(3)]
degn_result = mullerhorner(pn(5), z, parameter)
degn_result_real = []
degn_result_imag = []
for i in range(0, 5):
    degn_result_real.append(degn_result.z[i].real)
    degn_result_imag.append(degn_result.z[i].imag)
    funval = np.abs(degn_result.funval[i])
    print("{:+.15e}\t\t{:+.15e}\t\t{:.15e}\t\t{}".format(
        degn_result.z[i].real, degn_result.z[i].imag, funval,
        degn_result.numit[i]))
print('-'*100)

parameter.ref = True

print("\n> Degree", 5)
print('-'*107)
print("{}\t\t\t{}\t\t\t{}\t\t\t{}\t{}".format('REAL PART', 'IMAG PART',
                                              '|FUNVAL|', 'ITER', 'REF'))
print('-'*107)
z = [0.5*(k-1) for k in range(3)]
degn_result = mullerhorner(pn(5), z, parameter)
degn_result_real = []
degn_result_imag = []
for i in range(0, 5):
    degn_result_real.append(degn_result.z[i].real)
    degn_result_imag.append(degn_result.z[i].imag)
    funval = np.abs(degn_result.funval[i])
    print("{:+.15e}\t\t{:+.15e}\t\t{:.15e}\t\t{}\t{}".format(
        degn_result.z[i].real, degn_result.z[i].imag, funval,
        degn_result.numit[i], degn_result.refnum[i]))
print('-'*107)