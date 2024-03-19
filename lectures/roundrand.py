from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib import rcParams
import numpy as np
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'STIXGeneral'

# Kahan rational function
def r(x):
    n = 622.0 - x * (751.0 - x *(324.0 - x * (59.0 - 4.0 * x)))
    d = 112.0 - x * (151.0 - x * (72.0 - x * (14.0 - x)))
    return n / d

# point and machine epsilon
a = 1.606
# array of values for x and y = r(x)
k = range(-200, 201)
eps = np.finfo(float).eps
x = np.array([a + j * eps for j in k])
y = r(x)
z = [r(a)] * len(k)
y_max = y.max()
y_min = y.min()

# plot figure
ax = plt.figure(1, figsize=(10, 4)).add_subplot(111)
ax.plot(k, z, linestyle="--", linewidth=1, color="firebrick")
ax.semilogy(k, y, marker="o", linestyle="-", linewidth=0.5, color="darkblue",
            markersize=4, markeredgecolor="k", alpha=0.5)
ax.set_xlabel("$k$", fontsize=12)
ax.set_ylabel(r"$fl(r(x_k))$", fontsize=12)
ax.set_yticks([y_min, (y_min + z[0]) / 2, z[0], (y_max + z[0]) / 2, y_max])
ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
ax.get_yaxis().set_minor_formatter(ticker.NullFormatter())
ax.grid(color="gray", linestyle="--", linewidth=0.5, which="major")
plt.tight_layout()
plt.axis("tight")
plt.show()