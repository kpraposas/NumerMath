# Sample implementation of Newton Lagrange Interpolation

import numpy as np
from matplotlib import pyplot as plt
from polyinterp import newtonlagrange

def fun(x):
    return np.e**(3*x)*np.sin(2*np.pi*x)

x = np.linspace(0., 1., 100)
y = fun(x)

cubic = newtonlagrange(fun, [1./3.*_ for _ in range(4)], x)
quartic = newtonlagrange(fun, [1./4.*_ for _ in range(5)], x)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.grid(color="gray", linestyle="--", linewidth=0.5)
ax1.plot(x, y, 'r', label=r"$f$")
ax1.plot(x, cubic, label=r"$L_{f, 3}$")
ax1.scatter(np.linspace(0., 1., 4), fun(np.linspace(0., 1., 4)), s=15, marker="o", c="k")
ax1.legend(loc='upper right')

ax2.grid(color="gray", linestyle="--", linewidth=0.5)
ax2.plot(x, y, 'r', label=r"$f$")
ax2.plot(x, quartic, label=r"$L_{f, 4}$")
ax2.scatter(np.linspace(0., 1., 5), fun(np.linspace(0., 1., 5)), s=15, marker="o", c="k")
ax2.legend(loc='upper right')

# plt.savefig('plots/lagrange.png', format='png', dpi=300)
plt.show()
plt.gca().set_aspect('equal')
