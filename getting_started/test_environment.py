import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
import scipy
import sklearn
import torch

print("matplotlib   ", matplotlib.__version__)
print("numpy        ", numpy.__version__)
print("pandas       ", pandas.__version__)
print("scikit-learn ", sklearn.__version__)
print("scipy        ", scipy.__version__)
print("torch        ", torch.__version__)

# Example derived from: https://matplotlib.org/stable/gallery/statistics/time_series_histogram.html

fig, axes = plt.subplots(nrows=3, figsize=(6, 8), dpi=96, layout='constrained')
np.random.seed(19680801)
num_series = 1000
num_points = 100
SNR = 0.10
x = np.linspace(0, 4 * np.pi, num_points)
Y = np.cumsum(np.random.randn(num_series, num_points), axis=-1)
num_signal = round(SNR * num_series)
phi = (np.pi / 8) * np.random.randn(num_signal, 1)
Y[-num_signal:] = (np.sqrt(np.arange(num_points)) * (np.sin(x - phi) + 0.05 * np.random.randn(num_signal, num_points)))
axes[0].plot(x, Y.T, color="C0", alpha=0.1)
axes[0].set_title("Line plot with alpha")
num_fine = 800
x_fine = np.linspace(x.min(), x.max(), num_fine)
y_fine = np.concatenate([np.interp(x_fine, x, y_row) for y_row in Y])
x_fine = np.broadcast_to(x_fine, (num_series, num_fine)).ravel()
cmap = plt.colormaps["plasma"]
cmap = cmap.with_extremes(bad=cmap(0))
h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[400, 100])
pcm = axes[1].pcolormesh(xedges, yedges, h.T, cmap=cmap, norm="log", vmax=1.5e2, rasterized=True)
fig.colorbar(pcm, ax=axes[1], label="# points", pad=0)
axes[1].set_title("2d histogram and log color scale")
pcm = axes[2].pcolormesh(xedges, yedges, h.T, cmap=cmap, vmax=1.5e2, rasterized=True)
fig.colorbar(pcm, ax=axes[2], label="# points", pad=0)
axes[2].set_title("2d histogram and linear color scale")
fig.savefig("test_figure.png")
plt.close(fig)
