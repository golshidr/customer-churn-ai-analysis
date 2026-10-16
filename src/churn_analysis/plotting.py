"""Shared figure style: one palette, readable fonts, light hairline grid.

Colours follow a colour-vision-deficiency-checked categorical order (blue, orange, aqua,
yellow, magenta, green, violet, red). Each model keeps the same colour in every figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = PALETTE
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"

#: Fixed colour per model ("colour follows the entity").
MODEL_COLORS = {
    "logreg": BLUE,
    "random_forest": ORANGE,
    "xgboost": AQUA,
    "lightgbm": YELLOW,
    "catboost": MAGENTA,
    "rf_original": MUTED,
    "dummy": AXIS,
}

#: Diverging map for signed quantities (blue = lowers churn risk, red = raises it).
DIVERGING = LinearSegmentedColormap.from_list("churn_div", [BLUE, NEUTRAL, RED])
#: Low-to-high feature values in SHAP beeswarm plots.
FEATURE_VALUE = LinearSegmentedColormap.from_list("churn_val", [BLUE, "#b9a3c9", RED])
#: Sequential single-hue ramp for magnitudes.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "churn_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 11,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "text.color": INK,
        }
    )


def savefig(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name, MUTED)
