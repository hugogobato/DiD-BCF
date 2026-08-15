"""Shared figure design system for the revision report.

One fixed categorical slot per method (never cycled), with a marker/linestyle as
secondary encoding so identity survives grayscale print.  Palette validated with
the dataviz skill's checker (all checks pass).

Imported by ``make_analysis.py`` (Workstreams B/C/D) and by the Workstream-D
ramp and Workstream-F4 pre-trend analyses, so every figure in the report shares
one legend.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METHOD_ORDER = ["plain", "corrected", "twfe", "did_dr", "did2s",
                "doubleml", "synthdid", "wang"]
COLOR = {"plain": "#2a78d6", "corrected": "#eb6834", "twfe": "#e34948",
         "did_dr": "#1baf7a", "did2s": "#eda100", "doubleml": "#e87ba4",
         "synthdid": "#008300", "wang": "#4a3aa7"}
MARKER = {"plain": "o", "corrected": "s", "twfe": "X", "did_dr": "^",
          "did2s": "v", "doubleml": "D", "synthdid": "P", "wang": "*"}
DASH = {"plain": "-", "corrected": "-", "twfe": "--", "did_dr": "-.",
        "did2s": ":", "doubleml": "--", "synthdid": "-.", "wang": "-"}
LABEL = {"plain": "DiD-BCF (plain)", "corrected": "DiD-BCF (corrected)",
         "twfe": "TWFE", "did_dr": "Callaway--Sant'Anna",
         "did2s": "Gardner (did2s)", "doubleml": "DoubleML DR-DiD",
         "synthdid": "Synthetic DiD", "wang": "grf-DiD (Wang)"}

# The pre-trend diagnostic and its standard-practice comparator get their own
# two slots, reusing the DiD-BCF and TWFE hues so the reader carries the
# association across sections.
COLOR.update({"pretrend": COLOR["plain"], "twfe_es": COLOR["twfe"]})
MARKER.update({"pretrend": "o", "twfe_es": "X"})
DASH.update({"pretrend": "-", "twfe_es": "--"})
LABEL.update({"pretrend": "DiD-BCF pre-trend diagnostic",
              "twfe_es": "TWFE event-study placebo"})

TEXLABEL = dict(LABEL)
PLAINLABEL = {k: v.replace("--", "-") for k, v in LABEL.items()}

SCEN_ORDER = ["B1_baseline", "B1_strong_confounder", "B1_serial_corr",
              "B1_selection_obs", "B1_null", "B2_sweep", "B2_sweep_serial",
              "D_staggered", "D_contamination"]
SCEN_TEX = {s: s.replace("_", r"\_") for s in SCEN_ORDER}

RC = {
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False,
    "axes.labelcolor": "#3c3c3c", "text.color": "#1a1a1a",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.edgecolor": "#9a9a95", "figure.facecolor": "white",
}
plt.rcParams.update(RC)


def order_methods(df: pd.DataFrame, col: str = "method", order=None) -> pd.DataFrame:
    """Sort ``df`` into the canonical method order (stable across figures)."""
    order = order or METHOD_ORDER
    out = df.copy()
    out[col] = pd.Categorical(out[col], categories=order, ordered=True)
    return out.sort_values(col)


def style_of(method: str) -> dict:
    """Keyword arguments pinning one method to its fixed visual slot."""
    return {"color": COLOR.get(method, "#666666"),
            "marker": MARKER.get(method, "o"),
            "linestyle": DASH.get(method, "-"),
            "label": PLAINLABEL.get(method, method)}
