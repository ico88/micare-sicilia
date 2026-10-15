"""Harmonised interpretation of MICs with a single EUCAST breakpoint version.

The laboratory S/I/R ("refertato") reflects the breakpoints in force when the
isolate was tested. For trend forecasting every MIC is re-read with one
version, so a breakpoint revision does not look like a change in resistance.

Breakpoints come from the official EUCAST workbook converted by
scripts/import_eucast_breakpoints.py into app/resources/breakpoints/.
S if MIC <= S breakpoint; R if MIC > R breakpoint; I in between.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BREAKPOINTS_VERSION = "16.1"
RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources" / "breakpoints"

# Pathogen codes of the Rete MIC export -> EUCAST organism sheet and the words
# that identify the species in species-specific rows.
PATHOGEN_GROUPS: dict[str, dict] = {
    "ESCCOL": {"sheet": "Enterobacterales", "species": ["E. coli", "Escherichia coli"], "group_words": ["enterobacterales"]},
    "KLEPNE": {"sheet": "Enterobacterales", "species": ["K. pneumoniae", "Klebsiella"], "group_words": ["enterobacterales"]},
    "PSEAER": {"sheet": "Pseudomonas", "species": ["P. aeruginosa", "Pseudomonas aeruginosa"], "group_words": ["pseudomonas"]},
    "ACISPP": {"sheet": "Acinetobacter", "species": ["Acinetobacter"], "group_words": ["acinetobacter"]},
    "STAAUR": {"sheet": "Staphylococcus", "species": ["S. aureus", "Staphylococcus aureus"], "group_words": ["staphylococci"]},
    "ENTFAE": {"sheet": "Enterococcus", "species": ["E. faecalis", "Enterococcus faecalis"], "group_words": ["enterococci"]},
    "ENTFCM": {"sheet": "Enterococcus", "species": ["E. faecium", "Enterococcus faecium"], "group_words": ["enterococci"]},
    "STRPNE": {"sheet": "S.pneumoniae", "species": ["S. pneumoniae", "Streptococcus pneumoniae"], "group_words": ["pneumoniae"]},
}

# Antibiotic codes of the export -> agent name as written in the EUCAST tables
ANTIBIOTIC_NAMES: dict[str, str] = {
    "AMK": "Amikacin", "AMC": "Amoxicillin-clavulanic acid", "AMP": "Ampicillin",
    "FEP": "Cefepime", "FDC": "Cefiderocol", "CTX": "Cefotaxime", "FOX": "Cefoxitin",
    "CPT": "Ceftaroline", "CAZ": "Ceftazidime", "CZA": "Ceftazidime-avibactam",
    "CZT": "Ceftolozane-tazobactam", "CRO": "Ceftriaxone", "CIP": "Ciprofloxacin",
    "CLI": "Clindamycin", "COL": "Colistin", "SXT": "Trimethoprim-sulfamethoxazole",
    "DAP": "Daptomycin", "DOR": "Doripenem", "ERY": "Erythromycin", "ETP": "Ertapenem",
    "FOS": "Fosfomycin", "GEN": "Gentamicin", "GEH": "Gentamicin (test for acquired",
    "IPM": "Imipenem", "IMR": "Imipenem-relebactam", "LVX": "Levofloxacin",
    "LNZ": "Linezolid", "MEM": "Meropenem", "MEV": "Meropenem-vaborbactam",
    "NIT": "Nitrofurantoin", "ORI": "Oritavancin", "OXA": "Oxacillin",
    "PEN": "Benzylpenicillin", "TZP": "Piperacillin-tazobactam", "RIF": "Rifampicin",
    "STH": "Streptomycin", "TEC": "Teicoplanin", "TCY": "Tetracycline",
    "TGC": "Tigecycline", "TOB": "Tobramycin", "VAN": "Vancomycin",
}


@lru_cache(maxsize=4)
def load_breakpoints(version: str = BREAKPOINTS_VERSION) -> dict[tuple[str, str], tuple[float, float]]:
    """(antibiotic, pathogen) -> (S <=, R >) for pairs with numeric breakpoints."""
    path = RESOURCE_DIR / f"eucast_v{version}.csv"
    if not path.exists():
        return {}
    table = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["status"] == "ok" and row["s_le"] and row["r_gt"]:
                table[(row["antibiotic"], row["pathogen"])] = (float(row["s_le"]), float(row["r_gt"]))
    return table


def breakpoint_table(version: str = BREAKPOINTS_VERSION) -> pd.DataFrame:
    """The chosen EUCAST row for every pathogen x antibiotic pair (for review)."""
    path = RESOURCE_DIR / f"eucast_v{version}.csv"
    return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame()


def interpret_mic(antibiotic: pd.Series, pathogen: pd.Series,
                  operator: pd.Series, value: pd.Series,
                  version: str = BREAKPOINTS_VERSION) -> pd.Series:
    """S/I/R from MIC; 'NR' (not R, S or I) when only that is certain; NaN when undeterminable.

    Censored MICs are read conservatively: '<=0.5' is S when 0.5 <= S breakpoint,
    '>4' is R when 4 >= R breakpoint. For %R forecasting 'NR' is as good as S/I.
    """
    table = load_breakpoints(version)
    keys = list(zip(antibiotic, pathogen))
    s_le = pd.Series([table.get(k, (np.nan, np.nan))[0] for k in keys], index=value.index)
    r_gt = pd.Series([table.get(k, (np.nan, np.nan))[1] for k in keys], index=value.index)
    op = operator.fillna("=")
    at_most = op.isin(["=", "<=", "<"])  # true MIC <= value
    exact = op.eq("=")

    out = pd.Series(np.nan, index=value.index, dtype=object)
    out[at_most & (value <= r_gt)] = "NR"
    out[at_most & (value <= s_le)] = "S"
    out[exact & (value > s_le) & (value <= r_gt)] = "I"
    out[(op.eq(">") & (value >= r_gt)) | (op.isin(["=", ">="]) & (value > r_gt))] = "R"
    return out.where(s_le.notna() & value.notna())
