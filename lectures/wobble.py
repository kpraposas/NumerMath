from matplotlib import pyplot as plt
import numpy as np

# number of subdivisions and machine epsilon
subdivision = 1000
eps = np.finfo(float).eps
# initialize arrays
x = np.array([])
y = np.array([])
# relative errors
for k in range(5):
    x_ = np.linspace(2**k, 2**(k + 1), subdivision)
    y_ = eps / (x_ / 2**k)
    x = np.append(x, x_)
    y = np.append(y, y_)
#plot figure
ax = plt.figure(1, figsize=(10,4)).add_subplot(111)
ax.plot(x, y, label=r"$\frac{\tt eps}{m(x)}$", color="darkblue")
ax.legend(loc="best", fontsize=18)
ax.set_xticks([2**k for k in range(6)])
ax.grid(color="gray", linestyle="--", linewidth=0.5)
plt.show()
