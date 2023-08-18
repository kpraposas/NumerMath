# Runge Phenomenon in Lagrange interpolation

import numpy as np
from matplotlib import pyplot as plt
from polyinterp import lagrange

def fun(x):
    return 1 / (1 + x**2)

x = np.linspace(-5., 5., 200)
y = fun(x)

lagrange5 = lagrange(fun, [-5. + 10./5.*_ for _ in range(6)], x)
lagrange9 = lagrange(fun, [-5. + 10./9.*_ for _ in range(10)], x)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.grid(color="gray", linestyle="--", linewidth=0.5)
ax1.plot(x, y, 'r', label=r"$f$")
ax1.plot(x, lagrange5, label=r"$L_{f, 5}$")
ax1.scatter(np.linspace(0., 1., 4), fun(np.linspace(0., 1., 4)), s=15, marker="o", c="k")
ax1.legend(loc='upper right')
ax1.set_ylim(-0.5, 1.5)

ax2.grid(color="gray", linestyle="--", linewidth=0.5)
ax2.plot(x, y, 'r', label=r"$f$")
ax2.plot(x, lagrange9, label=r"$L_{f, 9}$")
ax2.scatter(np.linspace(0., 1., 5), fun(np.linspace(0., 1., 5)), s=15, marker="o", c="k")
ax2.legend(loc='upper right')
ax2.set_ylim(-0.5, 1.5)

plt.savefig('plots/runge.png', format='png', dpi=300)
plt.show()
plt.gca().set_aspect('equal')
