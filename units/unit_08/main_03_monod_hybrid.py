import os
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from units.unit_08.src.classes import MonodHybridModel

# By default, matplotlib makes math in italics, but we want our units in regular.
plt.rcParams["mathtext.default"] = "regular"

# Set random seeds for reproducibility
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

tts_output = train_test_split(x, y, s, ph, temp, test_size=0.2, random_state=0)

x_train, x_test, y_train, y_test, s_train, s_test, ph_train, ph_test, temp_train, temp_test = tts_output

model = MonodHybridModel(epochs=200, random_state=0, verbose=True)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
rmse_test_overall = root_mean_squared_error(y_test, y_pred)

u_ph_test = np.unique(ph_test)
u_temp_test = np.unique(temp_test)

for ph_val in u_ph_test:
    for temp_val in u_temp_test:
        s_array = np.linspace(0, np.max(s), 1000)
        ph_array = ph_val * np.ones_like(s_array)
        temp_array = temp_val * np.ones_like(s_array)
        x_input = np.stack((s_array, ph_array, temp_array), axis=1)

        x_plot = s_array
        y_plot = model.predict(x_input)[:, 0]

        mask_test_iter = (ph_test == ph_val) * (temp_test == temp_val)

        if np.sum(mask_test_iter) == 0:
            print(f"Skipping pH: {ph_val}, T: {temp_val} C")
            continue

        s_test_iter = s_test[mask_test_iter]
        x_test_iter = x_test[mask_test_iter]
        y_test_iter = y_test[mask_test_iter]
        y_pred_iter = model.predict(x_test_iter)
        rmse_test_iter = root_mean_squared_error(y_test_iter, y_pred_iter)

        ph_string = f"pH_{str(ph_val).replace('.', '_')}"
        temp_string = f"T_{str(temp_val).replace('.', '_')}"
        filename = f"figure_03_MonodHybrid_{ph_string}_{temp_string}.png"
        filepath = os.path.join("figures", filename)

        message = f"RMSE (overall):   {rmse_test_overall:.3f} " + r"$d^{-1}$" + "\n"
        message += f"RMSE (iteration): {rmse_test_iter:.3f} " + r"$d^{-1}$" + "\n"
        message += f"pH: {ph_val} | T: {temp_val} C"

        # Note how we are only plotting below here, all other calculations or definitions were done above
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(14.39 / 2.54, 12.09 / 2.54),
            dpi=300,
            layout="constrained",
        )

        ax.scatter(
            s_test_iter,
            y_test_iter,
            label="Test samples",
            s=25,
            c="tab:orange",
            marker="D",
            edgecolors="none",
            alpha=0.9,
            zorder=2,
        )
        ax.plot(
            x_plot,
            y_plot,
            color="black",
            label="Model (Hybrid)",
            linestyle="-",
            linewidth=3,
            zorder=3,
        )

        ax.set_xlabel("Glucose [mM]", fontsize=12)
        ax.set_ylabel(r"Specific Growth Rate [$d^{-1}$]", fontsize=12)
        ax.set_xlim(0, np.max(x))
        ax.set_ylim(0, 1.01 * np.max(y))
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.legend(loc="upper left", fontsize=9)
        ax.text(
            x=0.02,
            y=0.87,
            s=message,
            va="top",
            ha="left",
            transform=ax.transAxes,
            family="monospace",
            fontsize=9,
        )
        ax.xaxis.set_major_locator(MultipleLocator(5))

        fig.savefig(filepath)
        plt.close(fig)
