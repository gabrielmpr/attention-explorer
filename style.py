"""House style for every build. Import this; don't restyle per build."""
import matplotlib as mpl

BG = "#0e1116"
FG = "#d7dae0"
MUTED = "#6b7280"
GRID = "#1f2630"
# one accent per series, muted so nothing screams
ACCENTS = ["#e07a5f", "#81b29a", "#8ab4f8", "#f2cc8f"]
WARN = "#e0605f"

FIGSIZE_16x9 = (12.8, 7.2)
FIGSIZE_4x5 = (8.64, 10.8)
DPI = 150


def apply():
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": FG,
        "axes.titlecolor": FG,
        "text.color": FG,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "font.size": 11,
    })
