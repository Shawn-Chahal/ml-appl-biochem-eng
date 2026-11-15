from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot_monod(filepath, x_scatter, y_scatter, x_plot, y_plot, x_max, y_max, model_name, message):
    plt.rcParams["mathtext.default"] = "regular"

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14.39 / 2.54, 12.09 / 2.54), dpi=300, layout="constrained")

    ax.scatter(
        x_scatter, y_scatter, label="Test samples", s=25, c="tab:orange", marker="D", edgecolors="none", alpha=0.9,
        zorder=2,
    )

    ax.plot(x_plot, y_plot, color="black", label=f"Model ({model_name})", linestyle="-", linewidth=3, zorder=3)
    ax.set_xlabel("Glucose [mM]", fontsize=12)
    ax.set_ylabel(r"Specific Growth Rate [$d^{-1}$]", fontsize=12)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(x=0.02, y=0.87, s=message, va="top", ha="left", transform=ax.transAxes, family="monospace", fontsize=9)
    ax.xaxis.set_major_locator(MultipleLocator(5))

    fig.savefig(filepath)
    plt.close(fig)
