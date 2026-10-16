"""Resistance series per laboratory x pathogen x antibiotic, built from the
validated isolate archive (dataset_store), the way AMR surveillance requires:

- separate populations by specimen (MATERIALE, flusso AMR-RETEMIC): invasive
  (blood, CSF), urine (in the network from 2023) and other; mixing them shifts
  %R whenever the specimen mix changes
- first isolate per patient x pathogen in each period and population (CLSI M39)
- intrinsic resistances and EUCAST "IE" results left out
- wards pooled: one value per laboratory and period, weighted by isolates
- two readings: "reported" (laboratory S/I/R, breakpoints of that year) and
  "harmonized" (MIC re-read with one EUCAST version, see breakpoints.py)

Every period between the first and the last month of data is present, also
when nothing was tested (n = 0), so gaps are explicit for the models.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from micare_sicilia.config import START_DATE

from .dataset_store import first_isolate_mask, load_ast_results, load_isolates

MIN_ISOLATES_PER_PERIOD = 30  # CLSI M39 minimum for a reportable percentage
KEYS = ["popolazione", "laboratorio", "microorganismo", "antibiotic"]

# MATERIALE codes of the regional flow (Allegato 1): 1 blood, 2 CSF, 3 urine,
# 4 urethral, 5 cervical, 6 rectal, 7 pharyngeal, 8 other
POPULATIONS = {"1": "invasivi", "2": "invasivi", "3": "urine"}
OTHER_POPULATION = "altri materiali"
# Invasive isolates of non-hospitalised patients ("5 - altro", PAZIENTE_RICOVERATO) form a
# population of their own, outside the regional forecasts. In 2025 one laboratory started
# sending them by the thousand (2,057 tested in 2024, 12,073 in 2025), with far lower
# resistance: pooled with the inpatients they turned a change of who is tested into an
# apparent fall of resistance. Without them the 2025 out-of-sample error went from 4.1 to
# 3.2 points, the bias from -2.1 to -0.3 and the 80% interval coverage from 71% to 84%.
NON_HOSPITALISED_INVASIVE = "invasivi non ricoverati"
NON_HOSPITALISED_CODES = {"5"}


def _series_isolates(folder: str | Path) -> pd.DataFrame:
    """Archive isolates from the network start (START_DATE) on, with their population.

    Earlier isolates stay in the archive but are not part of any series.
    """
    isolates = load_isolates(folder)
    if isolates.empty:
        return isolates
    return add_population(isolates[isolates["data_prelievo"] >= START_DATE])


def add_population(isolates: pd.DataFrame) -> pd.DataFrame:
    material = isolates["materiale"].astype(str).str.strip() if "materiale" in isolates else ""
    population = pd.Series(material, index=isolates.index).map(POPULATIONS).fillna(OTHER_POPULATION)
    if "paziente_ricoverato" in isolates:
        setting = isolates["paziente_ricoverato"].astype(str).str.strip()
        population = population.mask(population.eq("invasivi") & setting.isin(NON_HOSPITALISED_CODES),
                                     NON_HOSPITALISED_INVASIVE)
    return isolates.assign(popolazione=population)
READINGS = {"reported": "result", "harmonized": "result_harmonized"}


def wilson_interval(k: pd.Series, n: pd.Series, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    """95% Wilson interval for k/n, in percent; NaN where n = 0."""
    n_safe = n.where(n > 0)
    p = k / n_safe
    centre = (p + z * z / (2 * n_safe)) / (1 + z * z / n_safe)
    half = z * np.sqrt(p * (1 - p) / n_safe + z * z / (4 * n_safe * n_safe)) / (1 + z * z / n_safe)
    return 100 * (centre - half), 100 * (centre + half)


# Patient type (PAZIENTE_RICOVERATO) as a short stratum label for split_by
SETTING_LABELS = {"1": "ricoverati", "2": "day hospital", "3": "pronto soccorso", "4": "RSA",
                  "5": "non ricoverati, altro"}


def build_series(folder: str | Path, freq: str = "Q", split_by: str | None = None) -> pd.DataFrame:
    """One row per laboratory x pathogen x antibiotic x period ("Q" or "M").

    split_by="paziente_ricoverato" splits each laboratory by patient type
    ("IT168 · ricoverati"): the strata add up to the same regional figures.

    Columns per reading r in (reported, harmonized):
      n_<r>         first isolates with a usable result
      r_<r>         of which resistant
      pct_r_<r>     % resistant; low_<r> / high_<r> its 95% interval
    plus isolates (first isolates of the pathogen in the laboratory), tested_share
    (n_harmonized / isolates) and reportable (n_harmonized >= 30).
    """
    isolates = _series_isolates(folder)
    if isolates.empty:
        return pd.DataFrame()
    isolates = isolates[first_isolate_mask(isolates, freq, by=["popolazione"])]
    if split_by == "paziente_ricoverato" and split_by in isolates:
        # after the first-isolate rule, so the strata add up to the regional series
        setting = isolates[split_by].astype(str).str.strip().map(SETTING_LABELS).fillna("tipo non indicato")
        isolates = isolates.assign(laboratorio=isolates["laboratorio"] + " · " + setting)
    ast = load_ast_results(folder, with_isolate_columns=False)
    ast = ast.merge(isolates[["upload_id", "isolate_id", "popolazione", "laboratorio", "microorganismo", "data_prelievo"]],
                    on=["upload_id", "isolate_id"], how="inner")
    ast = ast[~ast["intrinsic"].astype(bool)]
    ast["period"] = ast["data_prelievo"].dt.to_period(freq)

    parts = []
    for reading, col in READINGS.items():
        usable = ast.loc[ast[col].notna(), KEYS + ["period"]].assign(r=ast[col].eq("R").astype(int))
        grouped = usable.groupby(KEYS + ["period"]).agg(n=("r", "size"), r=("r", "sum"))
        parts.append(grouped.add_suffix(f"_{reading}"))
    series = pd.concat(parts, axis=1).fillna(0).astype(int).reset_index()

    # Explicit zero rows for periods without results, inside the data window of each
    # laboratory and population (urine only exists from 2023: no empty quarters before)
    windows = isolates.groupby(["laboratorio", "popolazione"])["data_prelievo"].agg(["min", "max"])
    lab_periods = pd.concat([
        pd.DataFrame({"laboratorio": lab, "popolazione": pop,
                      "period": pd.period_range(w["min"], w["max"], freq=freq)})
        for (lab, pop), w in windows.iterrows()
    ])
    grid = series[KEYS].drop_duplicates().merge(lab_periods, on=["laboratorio", "popolazione"])
    series = grid.merge(series, on=KEYS + ["period"], how="left").fillna(0)

    den_keys = ["popolazione", "laboratorio", "microorganismo", "period"]
    denominators = (
        isolates.assign(period=isolates["data_prelievo"].dt.to_period(freq))
        .groupby(den_keys).size().rename("isolates").reset_index()
    )
    series = series.merge(denominators, on=den_keys, how="left")
    series["isolates"] = series["isolates"].fillna(0).astype(int)

    for reading in READINGS:
        n, r = series[f"n_{reading}"].astype(int), series[f"r_{reading}"].astype(int)
        series[f"n_{reading}"], series[f"r_{reading}"] = n, r
        series[f"pct_r_{reading}"] = 100 * r / n.where(n > 0)
        series[f"low_{reading}"], series[f"high_{reading}"] = wilson_interval(r, n)
    series["tested_share"] = series["n_harmonized"] / series["isolates"].where(series["isolates"] > 0)
    series["reportable"] = series["n_harmonized"] >= MIN_ISOLATES_PER_PERIOD
    series["period_start"] = series["period"].dt.start_time
    series["period"] = series["period"].astype(str)
    return series.sort_values(KEYS + ["period_start"]).reset_index(drop=True)


def cumulative_antibiogram(folder: str | Path) -> pd.DataFrame:
    """Annual cumulative antibiogram per population and laboratory (CLSI M39), reported reading.

    First isolate per patient x pathogen x year; %S and %R with 95% interval;
    reportable only with at least 30 isolates tested.
    """
    isolates = _series_isolates(folder)
    if isolates.empty:
        return pd.DataFrame()
    isolates = isolates[first_isolate_mask(isolates, "Y", by=["popolazione"])]
    ast = load_ast_results(folder, with_isolate_columns=False)
    ast = ast.merge(isolates[["upload_id", "isolate_id", "popolazione", "laboratorio", "microorganismo", "data_prelievo"]],
                    on=["upload_id", "isolate_id"], how="inner")
    ast = ast[~ast["intrinsic"].astype(bool) & ast["result"].notna()]
    ast["anno"] = ast["data_prelievo"].dt.year
    ast = ast.assign(s=ast["result"].eq("S").astype(int), i=ast["result"].eq("I").astype(int),
                     r=ast["result"].eq("R").astype(int))
    table = ast.groupby(KEYS + ["anno"]).agg(
        testati=("result", "size"), s=("s", "sum"), i=("i", "sum"), r=("r", "sum")
    ).reset_index()
    table["pct_s"] = 100 * table["s"] / table["testati"]
    table["pct_r"] = 100 * table["r"] / table["testati"]
    table["pct_r_low"], table["pct_r_high"] = wilson_interval(table["r"], table["testati"])
    table["riportabile"] = table["testati"] >= MIN_ISOLATES_PER_PERIOD
    return table.sort_values(KEYS + ["anno"]).reset_index(drop=True)


def regional_series(series: pd.DataFrame, population: str = "invasivi") -> pd.DataFrame:
    """Laboratories pooled: one row per pathogen x antibiotic x period of one population."""
    keys = ["microorganismo", "antibiotic", "period", "period_start"]
    counts = ["n_reported", "r_reported", "n_harmonized", "r_harmonized", "isolates"]
    data = series[series["popolazione"] == population]
    # isolates is per laboratory x pathogen: count it once per laboratory before pooling
    pooled = data.groupby(keys)[counts[:-1]].sum()
    isolates = (data.drop_duplicates(["laboratorio", "microorganismo", "period"])
                .groupby(["microorganismo", "period"])["isolates"].sum())
    out = pooled.reset_index().merge(isolates.reset_index(), on=["microorganismo", "period"], how="left")
    out["laboratori"] = data[data["n_harmonized"] > 0].groupby(keys)["laboratorio"].nunique().reindex(
        pd.MultiIndex.from_frame(out[keys])).fillna(0).astype(int).to_numpy()
    for reading in READINGS:
        n, r = out[f"n_{reading}"], out[f"r_{reading}"]
        out[f"pct_r_{reading}"] = 100 * r / n.where(n > 0)
        out[f"low_{reading}"], out[f"high_{reading}"] = wilson_interval(r, n)
    out["reportable"] = out["n_harmonized"] >= MIN_ISOLATES_PER_PERIOD
    out.insert(0, "popolazione", population)
    return out.sort_values(["microorganismo", "antibiotic", "period_start"]).reset_index(drop=True)
