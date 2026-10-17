"""Data for the consultation dashboard: aggregated counts the operator can slice like
the regional Qlik dashboard, with the same rules as the forecasts.

The cube is built once from the archive and kept as Parquet next to it:
- first isolate per laboratory x patient x pathogen x quarter x population (CLSI M39),
  as in the series behind the forecasts
- isolates outside the period the data really cover are left out (covered_period)
- intrinsic resistances left out; both readings kept (reported, harmonised EUCAST)
It holds counts only, never a patient: one row per population x laboratory x patient
type x ward x sex x age band x pathogen x antibiotic x quarter.
A percentage is reportable with at least 30 isolates tested (MIN_ISOLATES_PER_PERIOD).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset_store import covered_period, first_isolate_mask, load_ast_results
from .reference import ANTIBIOTIC_LABELS, ICU_WARD_CODES, PATHOGEN_LABELS, laboratory_names
from .series import (
    MIN_ISOLATES_PER_PERIOD, NON_HOSPITALISED_INVASIVE, OTHER_POPULATION, READINGS, SETTING_LABELS,
    _series_isolates, wilson_interval,
)

CUBE_VERSION = 1
CUBE_FILE = "cubo.parquet"
ISOLATES_FILE = "isolati.parquet"
SIGNATURE_FILE = "firma.json"

# Filter dimensions, in the order of the filter row; "trimestre" is the period filter
DIMENSIONS = ["popolazione", "laboratorio", "tipo_paziente", "reparto", "sesso", "eta", "microorganismo"]
POPULATION_ORDER = ["invasivi", NON_HOSPITALISED_INVASIVE, "urine", OTHER_POPULATION]
AGE_BANDS = [(0, 18, "0-17 anni"), (18, 65, "18-64 anni"), (65, 80, "65-79 anni"), (80, 200, "80 anni e oltre")]
NOT_GIVEN = "non indicato"
SEX_LABELS = {"M": "maschi", "F": "femmine"}
ICU, OTHER_WARDS = "terapia intensiva", "altri reparti"


def pathogen_name(code: str) -> str:
    return PATHOGEN_LABELS.get(code, code)


def antibiotic_name(code: str) -> str:
    return ANTIBIOTIC_LABELS.get(code, code)


# ---------------------------------------------------------------------------
# Cube
# ---------------------------------------------------------------------------

def archive_signature(dataset_folder: str | Path) -> str:
    """Changes whenever an upload is added, replaced or removed."""
    files = sorted(Path(dataset_folder).glob("*/upload_*.parquet"))
    parts = [f"{f.parent.name}/{f.name}:{f.stat().st_size}:{int(f.stat().st_mtime)}" for f in files]
    return f"v{CUBE_VERSION}|" + "|".join(parts)


def _dimensions(isolates: pd.DataFrame) -> pd.DataFrame:
    def column(name: str) -> pd.Series:
        if name not in isolates:
            return pd.Series("", index=isolates.index)
        return isolates[name].fillna("").astype(str).str.strip()

    setting = column("paziente_ricoverato").map(SETTING_LABELS).fillna(NOT_GIVEN)
    ward = column("reparto_di_ricovero").str.lstrip("0")
    age = pd.to_numeric(isolates.get("eta"), errors="coerce") if "eta" in isolates else pd.Series(np.nan, index=isolates.index)
    band = pd.Series(NOT_GIVEN, index=isolates.index)
    for low, high, name in AGE_BANDS:
        band = band.mask(age.ge(low) & age.lt(high), name)
    return pd.DataFrame({
        "popolazione": isolates["popolazione"],
        "laboratorio": isolates["laboratorio"],
        "tipo_paziente": setting,
        "reparto": np.where(ward.isin(ICU_WARD_CODES), ICU, OTHER_WARDS),
        "sesso": column("sesso").map(SEX_LABELS).fillna(NOT_GIVEN),
        "eta": band,
        "microorganismo": isolates["microorganismo"],
        "trimestre": isolates["data_prelievo"].dt.to_period("Q").astype(str),
    }, index=isolates.index)


def build_cube(dataset_folder: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(results, isolates): counts of tested / resistant isolates per antibiotic, and of
    first isolates per pathogen, on every dimension of the dashboard."""
    isolates = _series_isolates(dataset_folder)
    if isolates.empty:
        return pd.DataFrame(), pd.DataFrame()
    start, end = covered_period(isolates["data_prelievo"])
    isolates = isolates[isolates["data_prelievo"].between(start, end)]
    isolates = isolates[first_isolate_mask(isolates, "Q", by=["popolazione"])]
    dims = _dimensions(isolates)
    keys = DIMENSIONS + ["trimestre"]
    counted = dims.groupby(keys, observed=True).size().rename("isolati").reset_index()

    ast = load_ast_results(dataset_folder, with_isolate_columns=False)
    ast = ast[~ast["intrinsic"].astype(bool)]
    ids = isolates[["upload_id", "isolate_id"]].join(dims)
    ast = ast.merge(ids, on=["upload_id", "isolate_id"], how="inner")
    parts = []
    for reading, col in READINGS.items():
        usable = ast.loc[ast[col].notna(), keys + ["antibiotic"]].assign(r=ast[col].eq("R").astype(int))
        grouped = usable.groupby(keys + ["antibiotic"], observed=True).agg(n=("r", "size"), r=("r", "sum"))
        parts.append(grouped.add_suffix(f"_{reading}"))
    results = pd.concat(parts, axis=1).fillna(0).astype("int32").reset_index()
    return _compact(results), _compact(counted)


def _compact(frame: pd.DataFrame) -> pd.DataFrame:
    for col in frame.columns:
        if frame[col].dtype == object and col != "trimestre":  # quarters are compared as text
            frame[col] = frame[col].astype("category")
    return frame


_cache: dict[str, tuple[str, pd.DataFrame, pd.DataFrame]] = {}


def load_cube(dataset_folder: str | Path, cube_folder: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The cube for the current archive: from memory, from disk, or rebuilt."""
    signature = archive_signature(dataset_folder)
    key = str(cube_folder)
    if key in _cache and _cache[key][0] == signature:
        return _cache[key][1], _cache[key][2]
    folder = Path(cube_folder)
    stored = folder / SIGNATURE_FILE
    if stored.exists() and json.loads(stored.read_text()).get("firma") == signature:
        results, counted = pd.read_parquet(folder / CUBE_FILE), pd.read_parquet(folder / ISOLATES_FILE)
    else:
        results, counted = build_cube(dataset_folder)
        folder.mkdir(parents=True, exist_ok=True)
        results.to_parquet(folder / CUBE_FILE, index=False)
        counted.to_parquet(folder / ISOLATES_FILE, index=False)
        stored.write_text(json.dumps({"firma": signature}))
    _cache[key] = (signature, results, counted)
    return results, counted


def cube_ready(dataset_folder: str | Path, cube_folder: str | Path) -> bool:
    signature = archive_signature(dataset_folder)
    if _cache.get(str(cube_folder), ("",))[0] == signature:
        return True
    stored = Path(cube_folder) / SIGNATURE_FILE
    return stored.exists() and json.loads(stored.read_text()).get("firma") == signature


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def _select(frame: pd.DataFrame, filters: dict, skip: tuple[str, ...] = ()) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for dim in DIMENSIONS:
        value = filters.get(dim)
        if value and dim not in skip:
            mask &= frame[dim] == value
    if filters.get("da"):
        mask &= frame["trimestre"].astype(str) >= filters["da"]
    if filters.get("a"):
        mask &= frame["trimestre"].astype(str) <= filters["a"]
    return frame[mask]


def _rates(frame: pd.DataFrame, reading: str) -> pd.DataFrame:
    n, r = frame[f"n_{reading}"].astype(int), frame[f"r_{reading}"].astype(int)
    low, high = wilson_interval(r, n)
    return pd.DataFrame({"testati": n, "resistenti": r, "pct": (100 * r / n.where(n > 0)).round(1),
                         "basso": low.round(1), "alto": high.round(1),
                         "riportabile": n >= MIN_ISOLATES_PER_PERIOD}, index=frame.index)


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def _options(values: pd.Series, order: list[str] | None = None) -> list[str]:
    present = [str(v) for v in values.dropna().unique() if str(v)]
    if order:
        return [v for v in order if v in present] + sorted(v for v in present if v not in order)
    return sorted(present)


def query(results: pd.DataFrame, counted: pd.DataFrame, filters: dict) -> dict:
    """Everything the dashboard shows for one selection.

    filters: the DIMENSIONS values ("" = all), "antibiotico", "da"/"a" (quarters,
    e.g. "2023Q1") and "lettura" ("harmonized" or "reported").
    """
    if results.empty:
        return {"vuoto": True}
    reading = filters.get("lettura") if filters.get("lettura") in READINGS else "harmonized"
    population = filters.get("popolazione") or "invasivi"
    filters = {**filters, "popolazione": population}
    # the quarters of the population (urine only from 2023: no empty quarters before)
    quarters = sorted(counted.loc[counted["popolazione"] == population, "trimestre"].astype(str).unique())
    lab_names = laboratory_names()

    in_population = results[results["popolazione"] == population]
    pathogens = (_select(counted, filters, skip=("microorganismo",))
                 .groupby("microorganismo", observed=True)["isolati"].sum().sort_values(ascending=False))
    pathogens = [p for p, n in pathogens.items() if n > 0]
    pathogen = filters.get("microorganismo") or (pathogens[0] if pathogens else "")
    filters["microorganismo"] = pathogen

    chosen = _select(results, filters)
    tested = chosen.groupby("antibiotic", observed=True)[f"n_{reading}"].sum().sort_values(ascending=False)
    antibiotics = [a for a, n in tested.items() if n > 0]
    antibiotic = filters.get("antibiotico") if filters.get("antibiotico") in antibiotics else (antibiotics[0] if antibiotics else "")
    combo = chosen[chosen["antibiotic"] == antibiotic]
    counts = [f"n_{reading}", f"r_{reading}"]

    # Key figures
    isolates = _select(counted, filters)
    total = combo[counts].sum()
    headline = _rates(pd.DataFrame([total]), reading).iloc[0].to_dict() if len(combo) else None
    kpi = {
        "isolati": int(isolates["isolati"].sum()),
        "laboratori": int(isolates.loc[isolates["isolati"] > 0, "laboratorio"].nunique()),
        "trimestri": int(isolates.loc[isolates["isolati"] > 0, "trimestre"].nunique()),
        "combinazione": headline,
    }

    # Trend per quarter, every quarter of the selected period (gaps explicit)
    period = [q for q in quarters if (not filters.get("da") or q >= filters["da"]) and (not filters.get("a") or q <= filters["a"])]
    by_quarter = combo.groupby("trimestre", observed=True)[counts].sum().reindex(period, fill_value=0)
    trend = _rates(by_quarter, reading).reset_index(names="trimestre")

    # Laboratories, over the whole selected period
    by_lab = combo.groupby("laboratorio", observed=True)[counts].sum()
    by_lab = by_lab[by_lab[counts[0]] > 0]
    labs = _rates(by_lab, reading).reset_index(names="laboratorio").sort_values("pct", ascending=False)
    labs["nome"] = labs["laboratorio"].map(lambda c: lab_names.get(str(c), str(c)))

    # Antibiogram: every pathogen x antibiotic of the selection (pathogen and antibiotic not filtered)
    grid = _select(results, filters, skip=("microorganismo",))
    cells = grid.groupby(["antibiotic", "microorganismo"], observed=True)[counts].sum()
    cells = cells[cells[counts[0]] > 0]
    antibiogram = _rates(cells, reading).reset_index()
    columns = [p for p in pathogens if p in set(antibiogram["microorganismo"].astype(str))]
    order = antibiogram.groupby("antibiotic", observed=True)["testati"].sum().sort_values(ascending=False)

    options = {
        "popolazione": _options(results["popolazione"], POPULATION_ORDER),
        "laboratorio": [{"valore": lab, "nome": lab_names.get(lab, lab)}
                        for lab in _options(in_population["laboratorio"])],
        "tipo_paziente": _options(in_population["tipo_paziente"], list(SETTING_LABELS.values()) + [NOT_GIVEN]),
        "reparto": _options(in_population["reparto"], [ICU, OTHER_WARDS]),
        "sesso": _options(in_population["sesso"], list(SEX_LABELS.values()) + [NOT_GIVEN]),
        "eta": _options(in_population["eta"], [b[2] for b in AGE_BANDS] + [NOT_GIVEN]),
        "microorganismo": [{"valore": p, "nome": pathogen_name(p)} for p in pathogens],
        "antibiotico": [{"valore": a, "nome": antibiotic_name(a)} for a in antibiotics],
        "trimestri": quarters,
    }
    return {
        "vuoto": False,
        "selezione": {**{d: filters.get(d) or "" for d in DIMENSIONS}, "microorganismo": pathogen,
                      "antibiotico": antibiotic, "da": filters.get("da") or "", "a": filters.get("a") or "",
                      "lettura": reading},
        "nomi": {"microorganismo": pathogen_name(pathogen), "antibiotico": antibiotic_name(antibiotic)},
        "opzioni": options,
        "numeri": kpi,
        "andamento": _records(trend),
        "laboratori": _records(labs),
        "antibiogramma": {
            "colonne": [{"valore": p, "nome": pathogen_name(p)} for p in columns],
            "righe": [{"valore": a, "nome": antibiotic_name(a)} for a in order.index.astype(str)],
            "celle": _records(antibiogram),
        },
        "soglia": MIN_ISOLATES_PER_PERIOD,
    }


FILTER_LABELS = {"popolazione": "Popolazione", "laboratorio": "Laboratorio", "tipo_paziente": "Tipo di paziente",
                 "reparto": "Reparto", "sesso": "Sesso", "eta": "Età", "microorganismo": "Microrganismo",
                 "antibiotico": "Antibiotico", "da": "Dal trimestre", "a": "Al trimestre", "lettura": "Lettura"}
READING_LABELS = {"harmonized": "armonizzata EUCAST", "reported": "come refertata dal laboratorio"}


def export_frames(answer: dict) -> dict[str, pd.DataFrame]:
    """The aggregated tables behind one view (a query answer), sheet name -> table."""
    chosen = answer["selezione"]
    selection = pd.DataFrame([
        {"filtro": FILTER_LABELS[k],
         "valore": READING_LABELS.get(v, v) if k == "lettura" else (v or "tutti")}
        for k, v in chosen.items()
    ] + [{"filtro": "Regole", "valore": f"primo isolato per trimestre; percentuale riportabile con almeno "
                                       f"{answer['soglia']} isolati testati"}])
    rename = {"testati": "testati", "resistenti": "resistenti", "pct": "% R", "basso": "IC95 basso",
              "alto": "IC95 alto", "riportabile": "riportabile"}
    trend = pd.DataFrame(answer["andamento"]).rename(columns=rename)
    labs = pd.DataFrame(answer["laboratori"]).rename(columns=rename)
    cells = pd.DataFrame(answer["antibiogramma"]["celle"])
    if not cells.empty:
        cells = cells.assign(microorganismo_nome=cells["microorganismo"].map(pathogen_name),
                             antibiotico_nome=cells["antibiotic"].map(antibiotic_name)).rename(columns=rename)
    return {"Selezione": selection, "Andamento": trend, "Laboratori": labs, "Antibiogramma": cells}
