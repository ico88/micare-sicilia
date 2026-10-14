"""Clinical breakpoints used to interpret MICs reported without an S/I/R category.

Only MICs that arrive without a laboratory interpretation are interpreted;
results reported by the laboratory are never overwritten. Interpreted rows are
marked ``result_source = "eucast_calcolato"`` in the stored dataset.

Values: EUCAST Clinical Breakpoint Tables v14.0 (2024), mg/L.
S if MIC <= s_le; R if MIC > r_gt; I (susceptible, increased exposure) in between.
Verify against the official table before changing BREAKPOINTS_VERSION;
pathogen/antibiotic pairs not listed (or EUCAST "IE" / "-") stay uninterpreted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BREAKPOINTS_VERSION = "EUCAST v14.0 (2024)"

ENTEROBACTERALES = ["ESCCOL", "KLEPNE"]

# (antibiotic, pathogen code) -> (S <=, R >)
BREAKPOINTS: dict[tuple[str, str], tuple[float, float]] = {
    # Colistin
    **{("COL", p): (2.0, 2.0) for p in ENTEROBACTERALES},
    ("COL", "ACISPP"): (2.0, 2.0),
    ("COL", "PSEAER"): (4.0, 4.0),
    # Tigecycline: among Enterobacterales only E. coli (and C. koseri); IE for Acinetobacter
    ("TGC", "ESCCOL"): (0.5, 0.5),
    ("TGC", "STAAUR"): (0.5, 0.5),
    ("TGC", "ENTFAE"): (0.25, 0.25),
    ("TGC", "ENTFCM"): (0.25, 0.25),
    # Daptomycin: Staphylococcus only (IE for enterococci)
    ("DAP", "STAAUR"): (1.0, 1.0),
}


def interpret_mic(antibiotic: pd.Series, pathogen: pd.Series,
                  operator: pd.Series, value: pd.Series) -> pd.Series:
    """S/I/R from MIC and breakpoints; NaN when not determinable.

    Censored MICs are interpreted only when unambiguous: '<=0.5' is S when
    0.5 <= S breakpoint, '>4' is R when 4 >= R breakpoint.
    """
    keys = list(zip(antibiotic, pathogen))
    s_le = pd.Series([BREAKPOINTS.get(k, (np.nan, np.nan))[0] for k in keys], index=value.index)
    r_gt = pd.Series([BREAKPOINTS.get(k, (np.nan, np.nan))[1] for k in keys], index=value.index)
    op = operator.fillna("=")
    upper_bounded = op.isin(["=", "<=", "<"])   # true MIC <= value
    lower_bounded = op.isin(["=", ">=", ">"])   # true MIC >= value (strictly for ">")
    exact = op.eq("=")

    out = pd.Series(np.nan, index=value.index, dtype=object)
    out[upper_bounded & (value <= s_le)] = "S"
    out[(op.eq(">") & (value >= r_gt)) | (lower_bounded & ~op.eq(">") & (value > r_gt))] = "R"
    out[exact & (value > s_le) & (value <= r_gt)] = "I"
    return out.where(s_le.notna() & value.notna())
