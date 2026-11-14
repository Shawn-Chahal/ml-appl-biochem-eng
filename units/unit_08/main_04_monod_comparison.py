import os
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from units.unit_08.src.classes import MonodMechanisticModel, MonodDataDrivenModel, MonodHybridModel

# By default, matplotlib makes math in italics, but we want our units in regular.
plt.rcParams["mathtext.default"] = "regular"

# Set random seeds for reproducibility
random_state = 0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_hybrid_monod.csv"))

ph = df.loc[:, "pH"].to_numpy()
temp = df.loc[:, "Temperature [C]"].to_numpy()
s = df.loc[:, "Glucose [mM]"].to_numpy()
mu = df.loc[:, "Specific Growth Rate [d^-1]"].to_numpy()

x = np.stack((s, ph, temp), axis=1)
y = np.expand_dims(mu, axis=1)

tts_output = train_test_split(x, y, s, ph, temp, test_size=0.2, random_state=random_state)

x_train, x_test, y_train, y_test, s_train, s_test, ph_train, ph_test, temp_train, temp_test = tts_output

models = {
    "Mechanistic": MonodMechanisticModel(),
    "Data-Driven": MonodDataDrivenModel(random_state=random_state),
    "Hybrid": MonodHybridModel(random_state=random_state)
}

for model_name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    rmse_test_overall = root_mean_squared_error(y_test, y_pred)
    print(f"RMSE {model_name:12s}: {rmse_test_overall:.3f} d^-1")

u_ph_test = np.unique(ph_test)[2:-1:3]
u_temp_test = np.unique(temp_test)[3:-1:2]

nrows = u_ph_test.shape[0]
ncols = u_temp_test.shape[0]
figsize = (29.21 / 2.54, 12.09 / 2.54)
fontsize_xy_labels = 12
fontsize_ticks = 12
fontsize_legend = 6.5
fontsize_text = 7.5
colors = ["tab:blue", "tab:red", "black"]
linestyles = ["--", ":", "-"]

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=300, layout="constrained")

for i in range(nrows):
    for j in range(ncols):
        ph_val = u_ph_test[i]
        temp_val = u_temp_test[j]

        s_array = np.linspace(0, np.max(s), 1000)
        ph_array = ph_val * np.ones_like(s_array)
        temp_array = temp_val * np.ones_like(s_array)
        x_array = np.stack((s_array, ph_array, temp_array), axis=1)

        x_plot = s_array

        mask_test_iter = (ph_test == ph_val) * (temp_test == temp_val)

        if np.sum(mask_test_iter) == 0:
            print(f"Skipping pH: {ph_val}, T: {temp_val} C")
            continue

        s_test_iter = s_test[mask_test_iter]
        x_test_iter = x_test[mask_test_iter]
        y_test_iter = y_test[mask_test_iter]

        message = f"pH: {ph_val} | T: {temp_val:.0f} C"

        axes[i, j].scatter(s_test_iter, y_test_iter, label="Test samples", s=16, c="tab:orange", marker="D",
                           edgecolors="none", alpha=0.5, zorder=2)

        for k, (model_name, model) in enumerate(models.items()):
            y_plot = model.predict(x_array)[:, 0]
            axes[i, j].plot(x_plot, y_plot, color=colors[k], label=f"Model ({model_name})", linestyle=linestyles[k],
                            linewidth=1.5, zorder=3)

            y_pred_iter = model.predict(x_test_iter)
            rmse_test_iter = root_mean_squared_error(y_test_iter, y_pred_iter)
            message += "\n" + f"RMSE({model_name[0]}): {rmse_test_iter:.3f} " + r"$d^{-1}$"

        axes[i, j].tick_params(axis="both", which="major", labelsize=fontsize_ticks)

        if np.max(y_test_iter) < 0.6 * np.max(y_test):
            kwargs_text = dict(x=0.02, y=0.97, va="top", ha="left")
        else:
            kwargs_text = dict(x=0.98, y=0.03, va="bottom", ha="right")

        axes[i, j].text(s=message, transform=axes[i, j].transAxes, family="monospace", fontsize=fontsize_text,
                        **kwargs_text)

        axes[i, j].set_xlim(0, np.max(s_test))
        axes[i, j].set_ylim(0, 1.01 * np.max(y_test))

        if i == axes.shape[0] - 1:
            axes[i, j].set_xlabel("Glucose [mM]", fontsize=fontsize_xy_labels)
            axes[i, j].xaxis.set_major_locator(MultipleLocator(5))
        else:
            axes[i, j].set_xticks([])

        if j == 0:
            axes[i, j].set_ylabel(r"$\mu$ [$d^{-1}$]", fontsize=fontsize_xy_labels)
            axes[i, j].yaxis.set_major_locator(MultipleLocator(0.2))
        else:
            axes[i, j].set_yticks([])

        # Add the legend to only one subplot to avoid repetition
        if i == 0 and j == 0:
            axes[i, j].legend(loc="upper right", ncols=1, fontsize=fontsize_legend)

fig.savefig(os.path.join("figures", "figure_04_MonodModelComparison.png"))
plt.close(fig)
