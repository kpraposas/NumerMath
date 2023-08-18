# Sample implementation of Hermite Interpolation

import numpy as np
from matplotlib import pyplot as plt
from polyinterp import hermite

def fun(x):
    return np.e**(3*x)*np.sin(2*np.pi*x)

def dfun(x):
    return np.e**(3*x)*(3*np.sin(2*np.pi*x) + 2*np.pi*np.cos(2*np.pi*x))

x = np.linspace(0., 1., 100)
y = fun(x)

hermite5 = hermite(fun, dfun, [1.*_ for _ in range(2)], x)
hermite9 = hermite(fun, dfun, [1./2.*_ for _ in range(3)], x)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.grid(color="gray", linestyle="--", linewidth=0.5)
ax1.plot(x, y, 'r', label=r"$f$")
ax1.plot(x, hermite5, label=r"$H_{f, 5}$")
ax1.scatter(np.linspace(0., 1., 2), fun(np.linspace(0., 1., 2)), s=15, marker="o", c="k")
ax1.legend(loc='upper right')

ax2.grid(color="gray", linestyle="--", linewidth=0.5)
ax2.plot(x, y, 'r', label=r"$f$")
ax2.plot(x, hermite9, label=r"$H_{f, 9}$")
ax2.scatter(np.linspace(0., 1., 3), fun(np.linspace(0., 1., 3)), s=15, marker="o", c="k")
ax2.legend(loc='upper right')

# plt.savefig('plots/lagrange.png', format='png', dpi=300)
plt.show()
plt.gca().set_aspect('equal')
