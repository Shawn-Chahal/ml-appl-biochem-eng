import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from units.unit_08.src.classes import MonodMechanisticModel

# By default, matplotlib makes math in italics, but we want our units in regular.
plt.rcParams['mathtext.default'] = 'regular'

df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_hybrid_monod.csv"))

ph = df.loc[:, "pH"].to_numpy()
temp = df.loc[:, "Temperature [C]"].to_numpy()

x = df.loc[:, "Glucose [mM]"].to_numpy()
y = df.loc[:, "Specific Growth Rate [d^-1]"].to_numpy()

tts_output = train_test_split(x, y, ph, temp, test_size=0.05, random_state=0)

x_train, x_test, y_train, y_test, ph_train, ph_test, temp_train, temp_test = tts_output

model = MonodMechanisticModel()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
rmse_test = root_mean_squared_error(y_test, y_pred)

x_plot = np.linspace(0, np.max(x), 1000)
y_plot = model.predict(x_plot)

# ph_alpha = (ph_test - np.min(ph_test)) / (np.max(ph_test) - np.min(ph_test))
# s_scatter_min = 3 ** 2
# s_scatter_max = 6 ** 2
# s_scatter = (s_scatter_max - s_scatter_min) * ph_alpha + s_scatter_min

c_scatter = temp_test

message = f"RMSE: {rmse_test:.3f} " + r"$d^{-1}$"

c_iter = [
    (temp_test, "Temperature [C]"),
    (ph_test, "pH")
]

for c_scatter, c_name in c_iter:
    # Note how we are only plotting below here, all other calculations or definitions were done above
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14.39 / 2.54, 12.09 / 2.54), dpi=300, layout="constrained")

    splot = ax.scatter(x_test, y_test, label="Test samples", s=25, c=c_scatter, cmap="plasma", marker="D",
                       edgecolors="none", alpha=0.9, zorder=2)
    cbar = fig.colorbar(splot, label=c_name, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    ax.plot(x_plot, y_plot, color="black", label="Model (Mechanistic)", linestyle="-", linewidth=3, zorder=3)

    ax.set_xlabel("Glucose [mM]", fontsize=12)
    ax.set_ylabel(r"Specific Growth Rate [$d^{-1}$]", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=9)
    ax.text(x=0.02, y=0.87, s=message, va='top', ha='left', transform=ax.transAxes, fontsize=9)
    ax.xaxis.set_major_locator(MultipleLocator(5))

    fig.savefig(os.path.join("figures", f"figure_01_MonodMechanistic_{c_name.split(' ')[0]}.png"))
    plt.close(fig)
