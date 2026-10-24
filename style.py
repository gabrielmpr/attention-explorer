"""Figure style, matched to the explorer page: ink on paper, colored-pencil accents. Two themes, one per page theme."""
import matplotlib as mpl

THEMES = {
    'light': dict(BG="#ffffff", FG="#17191c", MUTED="#5c616a", GRID="#e3e5e8",
                  ACCENTS=["#c62f28", "#2e7a4c", "#2457c9", "#b07a1f"], SEQ="magma_r"),
    'dark': dict(BG="#1a1b1e", FG="#ebe9e4", MUTED="#a0a4ab", GRID="#2c2e33",
                 ACCENTS=["#ff6a5c", "#5fc28a", "#7aa2ff", "#e0b45a"], SEQ="magma"),
}
BG = FG = MUTED = GRID = SEQ = None
ACCENTS = []
DPI = 150


def use(theme):
    """ set the module's colors for `theme` and apply them to matplotlib """
    globals().update(THEMES[theme])
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
