"""Convert a revision panel to the column layout the R benchmark scripts expect.

The R estimators in ``R_code/`` (Callaway-Sant'Anna ``did``, ``did2s``,
``DoubleML``, ``synthdid``) read CSVs with the *original* study's column names::

    Y, time, unit_id, first_treat_period, X_1..X_5, CATE,
    post_treatment, eventually_treated, treatment_group, D, event_time

This module maps the revision's tidy frame (``cohort`` -> ``first_treat_period``,
``X1..X5`` -> ``X_1..X_5``, ``CATT`` -> ``CATE``, ``post`` -> ``post_treatment``)
onto that layout and drops the unobserved ``alpha`` (never exposed to estimators).
``first_treat_period`` keeps ``np.inf`` for never-treated; the R scripts convert
it to 0, the convention ``did::att_gt`` uses for the never-treated group.
"""

from __future__ import annotations

import pandas as pd

R_COLUMNS = ["unit_id", "time", "first_treat_period", "treatment_group",
             "eventually_treated", "D", "post_treatment", "event_time",
             "X_1", "X_2", "X_3", "X_4", "X_5", "CATE", "Y"]


def to_r_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with R-benchmark column names (alpha dropped)."""
    out = pd.DataFrame({
        "unit_id": df["unit_id"].to_numpy(),
        "time": df["time"].to_numpy(),
        "first_treat_period": df["cohort"].to_numpy(),     # np.inf = never-treated
        "treatment_group": df["treatment_group"].to_numpy(),
        "eventually_treated": df["eventually_treated"].to_numpy(),
        "D": df["D"].to_numpy(),
        "post_treatment": df["post"].to_numpy(),
        "event_time": df["event_time"].to_numpy(),
        "X_1": df["X1"].to_numpy(),
        "X_2": df["X2"].to_numpy(),
        "X_3": df["X3"].to_numpy(),
        "X_4": df["X4"].to_numpy(),
        "X_5": df["X5"].to_numpy(),
        "CATE": df["CATT"].to_numpy(),                      # true conditional effect
        "Y": df["Y"].to_numpy(),
    })
    return out[R_COLUMNS]
