"""TimeSage plotting theme -- beautiful, consistent visuals."""

import matplotlib.pyplot as plt
import matplotlib as mpl

# TimeSage color palette
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#2BA84A",
    "danger": "#E63946",
    "neutral": "#6C757D",
    "background": "#FAFBFC",
    "grid": "#E8ECEF",
    "text": "#2D3436",
}

PALETTE = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["accent"],
    COLORS["success"],
    COLORS["danger"],
    "#6A4C93",
    "#1982C4",
    "#FF595E",
]


def sage_theme():
    """Apply the TimeSage matplotlib theme."""
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "figure.facecolor": COLORS["background"],
        "axes.facecolor": COLORS["background"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.7,
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "text.color": COLORS["text"],
        "font.family": "sans-serif",
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def set_theme(theme: str = "sage"):
    """Set the plotting theme.

    Parameters
    ----------
    theme : str
        Theme name. Currently supports: sage, dark, minimal.
    """
    if theme == "sage":
        sage_theme()
    elif theme == "dark":
        plt.style.use("dark_background")
    elif theme == "minimal":
        plt.style.use("seaborn-v0_8-whitegrid")
    else:
        sage_theme()
