import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# Here are a few tips and tricks to simplify coding in matplotlib.

# Let's make some generic data
t = np.linspace(0, 120, 25)  # min
c_a = 30 * np.exp(-0.021 * t)  # mM
c_b = 6 * np.exp(-0.072 * t)  # mM
c_c = 20 * (1 - np.exp(-0.045 * t))  # mM
c_d = 15 * (1 - np.exp(-0.015 * t))  # mM

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.5, 4), dpi=200, layout="constrained")

# Let's say you want to have some common formatting across multiple subplots.
# You can define a dict and pass them as keyword arguments (i.e., kwargs):

kwargs_scatter = dict(s=20, edgecolors="black", linewidth=1)

axes[0, 0].scatter(t, c_a, label="A", color="tab:blue", marker="o", **kwargs_scatter)
axes[0, 1].scatter(t, c_b, label="B", color="tab:orange", marker="D", **kwargs_scatter)
axes[1, 0].scatter(t, c_c, label="C", color="tab:green", marker="s", **kwargs_scatter)
axes[1, 1].scatter(t, c_d, label="D", color="tab:red", marker="^", **kwargs_scatter)

# This will apply the settings in kwargs_scatter to each subplot.

# Sometimes there may be other things you want to set for each ax in your axes.
# You can use np.ravel to temporarily make you axes 1D so it becomes easier to iterate over it:

for ax in np.ravel(axes):
    # Time is typically on a basis of 60 or 24.
    # We can change the tickmarks to appear at more convenient factors of these numbers.
    ax.xaxis.set_major_locator(MultipleLocator(15))  # Tick mark every 15 minutes.

    # We can also set up our xlabel, ylabel, and legend for each ax:
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Concentration [mM]")
    ax.legend()

fig.savefig(os.path.join("figures", "figure_04.png"))
plt.close(fig)
