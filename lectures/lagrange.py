# Sample implementation of Lagrange Interpolation

import numpy as np
from matplotlib import pyplot as plt
from polyinterp import LagrangeInterp

def fun(x):
    return np.e**(3*x)*np.sin(2*np.pi*x)

x = np.linspace(0., 1., 100)
fig = plt.figure()
ax = fig.add_subplot()
y = fun(x)
ax.plot(x, y, 'r')
plt.show()