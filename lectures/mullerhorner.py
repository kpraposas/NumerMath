import numpy as np
import rootscalar

def pn(n):
    pn = []
    for i in range(0, n):
        pn.append(-1)
    pn.append(1)
    return pn

parameter = rootscalar.param()
parameter.maxit = 100
parameter.refmax = 100
parameter.tol = 10**3 * np.finfo(float).eps
parameter.reftol = 1e-3

parameter.ref = 0
print('-'*100)
print("{:<32}{:<32}{:<11}{:<32}".format('REAL PART', 'IMAG PART', '|FUNVAL|', 'ITER'))
print('-'*85)
deg5_result = rootscalar.mullerhorner(pn(5), -0.5, 0., 0.5, parameter).x
deg5_it = rootscalar.mullerhorner(pn(5), -0.5, 0., 0.5, parameter).numit
for i in range(0, 5):
    funval = abs(rootscalar.horner(pn(5), deg5_result[i])[0])
    print("{:+.15e}\t\t{:+.15e}\t\t{:.15e}\t\t{}".format(deg5_result[i].real,
        deg5_result[i].imag, funval, deg5_it[i]))
print('-'*85)

parameter.ref = 1 
print('-'*85)
print("{:<32}{:<32}{:<11}".format('REAL PART', 'IMAG PART', '|FUNVAL|'))
print('-'*85)
deg5_result = rootscalar.mullerhorner(pn(5), -0.5, 0., 0.5, parameter).x
for i in range(0, 5):
    funval = abs(rootscalar.usualpolyeval(pn(5), deg5_result[i]))
    print("{:+.15e}\t\t{:+.15e}\t\t{:.15e}".format(deg5_result[i].real, deg5_result[i].imag, funval))
print('-'*85)