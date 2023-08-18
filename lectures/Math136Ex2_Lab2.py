import numpy as np
import matplotlib.pyplot as plt
from rootspoly import *

def pn(n):
    pn = []
    for i in range(0, n):
        pn.append(-1)
    pn.append(1)
    return pn

def modulus(z):
    modulus = np.sqrt(z.real**2 + z.imag**2)
    return modulus

z = 1 + 1j
parameter = param()
parameter.maxit = 100
parameter.refmax = 100
parameter.tol = 10**3 * np.finfo(float).eps
parameter.reftol = 1e-3
parameter.ref = True

result_reals = []
result_imags = []
for j in [1, 2, 4, 8, 10]:
    n = 5 * j
    print("\n> Degree", n)
    print('-'*97)
    print("{:<28}\t{:<28}\t{:<12}\t{:<32}".format('REAL PART', 'IMAG PART', '|FUNVAL|', 'MODULUS'))
    print('-'*97)
    degn_result = newtonhorner(pn(n), z, parameter).x
    degn_result_real = []
    degn_result_imag = []
    for i in range(0, n):
        degn_result_real.append(degn_result[i].real)
        degn_result_imag.append(degn_result[i].imag)
        funval = abs(horner(pn(n), degn_result[i])[0]) / np.finfo(float).eps
        print("{:+.15e}\t\t{:+.15e}\t\t{:.1f}eps\t\t{:.15f}".format(degn_result[i].real,
            degn_result[i].imag, funval, modulus(degn_result[i])))
    print('-'*97)
    result_reals.append(degn_result_real)
    result_imags.append(degn_result_imag)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(result_reals[0], result_imags[0], s=15, marker="o", label=r"$n=5$")
ax.scatter(result_reals[1], result_imags[1], s=15, marker="o", label=r"$n=10$")
ax.scatter(result_reals[2], result_imags[2], s=15, marker="o", label=r"$n=20$")
ax.scatter(result_reals[3], result_imags[3], s=15, marker="o", label=r"$n=40$")
ax.scatter(result_reals[4], result_imags[4], s=15, marker="o", label=r"$n=50$")
plt.legend(loc='upper right')
ax.grid(color="gray", linestyle="--", linewidth=0.5)
ax.set_xlabel("Real Axis", fontsize=12)
ax.set_ylabel("Imaginary Axis", fontsize=12)
#plt.savefig("plots/newtonhorner.png", format='png', dpi=300)
plt.show()