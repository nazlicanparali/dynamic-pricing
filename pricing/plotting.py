import matplotlib

matplotlib.use("Agg")  # no display needed, figures are only saved
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREY = "#718096"
GREEN = "#2f855a"
RED = "#c53030"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
