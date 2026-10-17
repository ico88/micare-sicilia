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
from .forecast_run import POPULATION as FORECAST_POPULATION, READING as FORECAST_READING
from .forecast_run import RESULT_FILE as FORECAST_FILE, archive_signature as _archive_files
from .forecast_run import load_forecasts, stale_reason
from .reference import ANTIBIOTIC_LABELS, ICU_WARD_CODES, PATHOGEN_LABELS, laboratory_names
from .series import (
    MIN_ISOLATES_PER_PERIOD, NON_HOSPITALISED_INVASIVE, OTHER_POPULATION, READINGS, SETTING_LABELS,
    _series_isolates, wilson_interval,
)

CUBE_VERSION = 2
CUBE_FILE = "cubo.parquet"
ISOLATES_FILE = "isolati.parquet"
SIGNATURE_FILE = "firma.json"

# List boxes, in the order of the filter pane. Single: always one value selected (a %R
# pools neither blood with urine nor two antibiotics); multi: any number, none = all.
SINGLE = ["popolazione", "microorganismo", "antibiotico"]
MULTI = ["anno", "trimestre_anno", "laboratorio", "tipo_paziente", "reparto", "sesso", "eta"]
DIMENSIONS = ["popolazione", "microorganismo"] + MULTI  # columns of both tables of the cube
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
    """Changes whenever an upload is added, replaced or removed, or the cube changes form."""
    return f"v{CUBE_VERSION}|" + _archive_files(dataset_folder)


def has_data(dataset_folder: str | Path) -> bool:
    return bool(_archive_files(dataset_folder))


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
    quarter = isolates["data_prelievo"].dt.to_period("Q")
    return pd.DataFrame({
        "popolazione": isolates["popolazione"],
        "laboratorio": isolates["laboratorio"],
        "tipo_paziente": setting,
        "reparto": np.where(ward.isin(ICU_WARD_CODES), ICU, OTHER_WARDS),
        "sesso": column("sesso").map(SEX_LABELS).fillna(NOT_GIVEN),
        "eta": band,
        "microorganismo": isolates["microorganismo"],
        "anno": quarter.dt.year.astype(str),
        "trimestre_anno": quarter.dt.quarter.astype(str),
        "trimestre": quarter.astype(str),
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
        if pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_string_dtype(frame[col]):
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

QUARTER_OF_YEAR = {"1": "1° trimestre", "2": "2° trimestre", "3": "3° trimestre", "4": "4° trimestre"}
FIXED_ORDER = {
    "popolazione": POPULATION_ORDER,
    "trimestre_anno": list(QUARTER_OF_YEAR),
    "tipo_paziente": list(SETTING_LABELS.values()) + [NOT_GIVEN],
    "reparto": [ICU, OTHER_WARDS],
    "sesso": list(SEX_LABELS.values()) + [NOT_GIVEN],
    "eta": [b[2] for b in AGE_BANDS] + [NOT_GIVEN],
}
# Forecasts exist for invasive isolates, harmonised reading, for the region or one laboratory
FORECAST_FREE = ("tipo_paziente", "reparto", "sesso", "eta")


def _select(frame: pd.DataFrame, filters: dict, skip: tuple[str, ...] = ()) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for dim in DIMENSIONS:
        value = filters.get(dim)
        if not value or dim in skip:
            continue
        mask &= frame[dim].isin(value).to_numpy() if isinstance(value, list) else (frame[dim] == value).to_numpy()
    return frame[mask]


def _rates(frame: pd.DataFrame, reading: str) -> pd.DataFrame:
    n, r = frame[f"n_{reading}"].astype(int), frame[f"r_{reading}"].astype(int)
    low, high = wilson_interval(r, n)
    return pd.DataFrame({"testati": n, "resistenti": r, "pct": (100 * r / n.where(n > 0)).round(1),
                         "basso": low.round(1), "alto": high.round(1),
                         "riportabile": n >= MIN_ISOLATES_PER_PERIOD}, index=frame.index)


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def _value_name(dim: str, value: str, lab_names: dict) -> str:
    if dim == "laboratorio":
        return lab_names.get(value, value)
    if dim == "microorganismo":
        return pathogen_name(value)
    if dim == "antibiotico":
        return antibiotic_name(value)
    if dim == "trimestre_anno":
        return QUARTER_OF_YEAR.get(value, value)
    return value


def _list_box(dim: str, present, counts: pd.Series, selected, lab_names: dict) -> list[dict]:
    """Values of one list box with their state, as in Qlik: selezionato, possibile, escluso."""
    chosen = set(selected if isinstance(selected, list) else [selected] if selected else [])
    present = [str(v) for v in present if str(v)]
    counts = counts.groupby(counts.index.astype(str)).sum()
    if dim in FIXED_ORDER:
        order = FIXED_ORDER[dim]
        present = [v for v in order if v in present] + sorted(v for v in present if v not in order)
    elif dim in ("microorganismo", "antibiotico"):
        present = sorted(present, key=lambda v: (-int(counts.get(v, 0)), v))
    else:
        present = sorted(present)
    box = []
    for value in present:
        n = int(counts.get(value, 0))
        state = "selezionato" if value in chosen else "possibile" if n > 0 else "escluso"
        box.append({"valore": value, "nome": _value_name(dim, value, lab_names), "conteggio": n, "stato": state})
    return box


def query(results: pd.DataFrame, counted: pd.DataFrame, filters: dict, forecasts: dict | None = None,
          stale: str | None = None) -> dict:
    """Everything the dashboard shows for one selection.

    filters: SINGLE fields as strings ("" = default), MULTI fields as lists ([] = all),
    "lettura" ("harmonized" or "reported"). forecasts: the stored forecasts (forecast_run),
    shown where the selection matches what they forecast; stale: why they are out of date.
    """
    if results.empty:
        return {"vuoto": True}
    reading = filters.get("lettura") if filters.get("lettura") in READINGS else "harmonized"
    filters = {**{d: [] for d in MULTI}, **{k: v for k, v in filters.items() if v}}
    filters["popolazione"] = filters.get("popolazione") or "invasivi"
    lab_names = laboratory_names()
    counts = [f"n_{reading}", f"r_{reading}"]
    in_population = counted[counted["popolazione"] == filters["popolazione"]]

    # Pathogen and antibiotic: always one; the most frequent of the selection when none
    by_pathogen = _select(counted, filters, skip=("microorganismo",)).groupby("microorganismo", observed=True)["isolati"].sum()
    possible = by_pathogen[by_pathogen > 0].sort_values(ascending=False)
    if filters.get("microorganismo") not in set(in_population["microorganismo"].astype(str)):
        filters["microorganismo"] = str(possible.index[0]) if len(possible) else ""
    chosen = _select(results, filters)
    by_antibiotic = chosen.groupby("antibiotic", observed=True)[counts[0]].sum()
    tested = by_antibiotic[by_antibiotic > 0].sort_values(ascending=False)
    pathogen_antibiotics = set(results.loc[(results["popolazione"] == filters["popolazione"])
                                           & (results["microorganismo"] == filters["microorganismo"]), "antibiotic"].astype(str))
    if filters.get("antibiotico") not in pathogen_antibiotics:
        filters["antibiotico"] = str(tested.index[0]) if len(tested) else ""
    combo = chosen[chosen["antibiotic"] == filters["antibiotico"]]

    # List boxes: every value of the population, with what the other selections leave possible
    boxes = {"popolazione": _list_box("popolazione", counted["popolazione"].unique(),
                                      _select(counted, filters, skip=("popolazione", "microorganismo"))
                                      .groupby("popolazione", observed=True)["isolati"].sum(),
                                      filters["popolazione"], lab_names),
             "microorganismo": _list_box("microorganismo", in_population["microorganismo"].unique(), by_pathogen,
                                         filters["microorganismo"], lab_names),
             "antibiotico": _list_box("antibiotico", sorted(pathogen_antibiotics), by_antibiotic,
                                      filters["antibiotico"], lab_names)}
    for dim in MULTI:
        others = _select(counted, filters, skip=(dim,)).groupby(dim, observed=True)["isolati"].sum()
        boxes[dim] = _list_box(dim, in_population[dim].unique(), others, filters[dim], lab_names)

    # Key figures
    isolates = _select(counted, filters)
    headline = _rates(pd.DataFrame([combo[counts].sum()]), reading).iloc[0].to_dict() if len(combo) else None
    kpi = {
        "isolati": int(isolates["isolati"].sum()),
        "laboratori": int(isolates.loc[isolates["isolati"] > 0, "laboratorio"].nunique()),
        "trimestri": int(isolates.loc[isolates["isolati"] > 0, "trimestre"].nunique()),
        "combinazione": headline,
    }

    # Trend: every quarter of the population inside the selected years and quarters (gaps explicit)
    quarters = sorted(_select(in_population, {d: filters[d] for d in ("anno", "trimestre_anno")})["trimestre"].astype(str).unique())
    by_quarter = combo.groupby("trimestre", observed=True)[counts].sum().reindex(quarters, fill_value=0)
    trend = _rates(by_quarter, reading).reset_index(names="trimestre")

    # Laboratories over the selected period (the laboratory selection itself is not applied)
    lab_rows = _select(results, filters, skip=("laboratorio",))
    lab_rows = lab_rows[lab_rows["antibiotic"] == filters["antibiotico"]]
    by_lab = lab_rows.groupby("laboratorio", observed=True)[counts].sum()
    by_lab = by_lab[by_lab[counts[0]] > 0]
    labs = _rates(by_lab, reading).reset_index(names="laboratorio").sort_values("pct", ascending=False)
    labs["nome"] = labs["laboratorio"].map(lambda c: lab_names.get(str(c), str(c)))
    region = _rates(pd.DataFrame([by_lab.sum()]), reading).iloc[0].to_dict() if len(by_lab) else None

    # Antibiogram: every pathogen x antibiotic of the selection
    grid = _select(results, filters, skip=("microorganismo",))
    cells = grid.groupby(["antibiotic", "microorganismo"], observed=True)[counts].sum()
    cells = cells[cells[counts[0]] > 0]
    antibiogram = _rates(cells, reading).reset_index()
    columns = [str(p) for p in possible.index if p in set(antibiogram["microorganismo"].astype(str))]
    order = antibiogram.groupby("antibiotic", observed=True)["testati"].sum().sort_values(ascending=False)

    selection = {d: filters.get(d) or ([] if d in MULTI else "") for d in SINGLE + MULTI}
    return {
        "vuoto": False,
        "selezione": {**selection, "lettura": reading},
        "nomi": {"microorganismo": pathogen_name(filters["microorganismo"]),
                 "antibiotico": antibiotic_name(filters["antibiotico"])},
        "liste": boxes,
        "numeri": kpi,
        "andamento": _records(trend),
        "laboratori": _records(labs),
        "regione": region,
        "antibiogramma": {
            "colonne": [{"valore": p, "nome": pathogen_name(p)} for p in columns],
            "righe": [{"valore": a, "nome": antibiotic_name(a)} for a in order.index.astype(str)],
            "celle": _records(antibiogram),
        },
        "previsione": forecast_view(forecasts, stale, filters, reading),
        "soglia": MIN_ISOLATES_PER_PERIOD,
    }


def forecast_view(forecasts: dict | None, stale: str | None, filters: dict, reading: str) -> dict:
    """The stored forecasts that answer this selection: regional (quarters and next 12
    months) or of the one selected laboratory (next 12 months); with the reasons when none do."""
    if not forecasts or not forecasts.get("meta", {}).get("available"):
        return {"disponibile": False, "motivi": ["Le previsioni non sono ancora state calcolate (Percorso guidato, passo 3)."]}
    reasons = []
    if filters["popolazione"] != FORECAST_POPULATION:
        reasons.append("Le previsioni riguardano gli isolati invasivi.")
    if reading != FORECAST_READING:
        reasons.append("Le previsioni usano la lettura armonizzata EUCAST.")
    if any(filters.get(d) for d in FORECAST_FREE):
        reasons.append("Le previsioni non sono calcolate per tipo di paziente, reparto, sesso o età.")
    labs = filters.get("laboratorio") or []
    if len(labs) > 1:
        reasons.append("Le previsioni sono per la regione o per un laboratorio alla volta.")
    meta = forecasts["meta"]
    view = {"disponibile": not reasons, "motivi": reasons, "ambito": "laboratorio" if labs else "regione",
            "laboratorio": labs[0] if len(labs) == 1 else "", "calcolate_il": meta.get("calcolato_il"),
            "dati_fino_al": meta.get("dati_fino_al"), "ultimo_trimestre": meta.get("ultimo_trimestre"),
            "periodi": _next_quarters(meta.get("ultimo_trimestre", "")), "da_ricalcolare": stale}
    if reasons:
        return view
    lab = view["laboratorio"]
    by_key = {(c["microorganismo"], c["antibiotico"]): c for c in forecasts["combinations"]}
    combo = by_key.get((filters["microorganismo"], filters["antibiotico"]))

    def year(entry: dict | None) -> dict | None:
        if not entry:
            return None
        if lab:
            rows = entry.get("previsioni_laboratori") or {}
            row = next((r for r in rows.get("laboratori", []) if r["codice"] == lab), None)
            return _forecast_value(row, rows.get("verifica"))
        row = entry.get("prossimi_12_mesi")
        return _forecast_value(row, (row or {}).get("verifica"))

    view["dodici_mesi"] = year(combo)
    view["trimestrali"] = [] if lab or not combo else [
        {"periodo": f["periodo"], "trimestre": f["trimestre"], "pct": f["pct"], "basso": f["low"], "alto": f["high"],
         "semaforo": f["semaforo"], "errore": f["errore_tipico"]}
        for f in combo.get("previsioni", [])]
    view["motivo_assenza"] = (combo or {}).get("motivo_assenza") or (
        "" if combo else "Nessuna previsione per questa combinazione: troppo pochi isolati testati.")
    lab_rows = ((combo or {}).get("previsioni_laboratori") or {}).get("laboratori", [])
    view["laboratori"] = [dict(_forecast_value(r, None), laboratorio=r["codice"], nome=r["nome"]) for r in lab_rows]
    view["celle"] = [dict(value, microorganismo=c["microorganismo"], antibiotic=c["antibiotico"])
                     for c in forecasts["combinations"] if (value := year(c))]
    return view


def _next_quarters(label: str) -> list[str]:
    """The four quarters after the last one forecast from ("4° trimestre 2025" -> 2026Q1..2026Q4)."""
    try:
        last = pd.Period(f"{label.split()[-1]}Q{label[0]}", freq="Q")
    except (ValueError, IndexError):
        return []
    return [str(last + h) for h in range(1, 5)]


def _forecast_value(row: dict | None, check: dict | None) -> dict | None:
    if not row:
        return None
    return {"pct": round(row["pct"], 1), "basso": round(row["low"], 1), "alto": round(row["high"], 1),
            "semaforo": row["semaforo"], "testati_ultimi_12_mesi": row.get("n_ultimi_12_mesi"),
            "errore": (check or {}).get("errore")}


_forecast_cache: dict[str, tuple[float, dict | None]] = {}


def stored_forecasts(forecast_folder: str | Path, dataset_folder: str | Path) -> tuple[dict | None, str | None]:
    """The stored forecasts (read again only when the file changes) and why they are stale, if they are."""
    path = Path(forecast_folder) / FORECAST_FILE
    mtime = path.stat().st_mtime if path.exists() else 0.0
    if _forecast_cache.get(str(path), (None,))[0] != mtime:
        _forecast_cache[str(path)] = (mtime, load_forecasts(forecast_folder))
    data = _forecast_cache[str(path)][1]
    return data, stale_reason((data or {}).get("meta"), dataset_folder) if data else None


FILTER_LABELS = {"popolazione": "Popolazione", "microorganismo": "Microrganismo", "antibiotico": "Antibiotico",
                 "anno": "Anno", "trimestre_anno": "Trimestre", "laboratorio": "Laboratorio",
                 "tipo_paziente": "Tipo di paziente", "reparto": "Reparto", "sesso": "Sesso", "eta": "Età",
                 "lettura": "Lettura"}
READING_LABELS = {"harmonized": "armonizzata EUCAST", "reported": "come refertata dal laboratorio"}
TRAFFIC_WORDS = {"verde": "affidabile", "giallo": "indicativa", "grigio": "non affidabile"}


def export_frames(answer: dict) -> dict[str, pd.DataFrame]:
    """The aggregated tables behind one view (a query answer), sheet name -> table."""
    chosen = answer["selezione"]

    def shown(key, value):
        if key == "lettura":
            return READING_LABELS.get(value, value)
        if isinstance(value, list):
            return ", ".join(value) if value else "tutti"
        return value or "tutti"

    selection = pd.DataFrame([{"filtro": FILTER_LABELS[k], "valore": shown(k, v)} for k, v in chosen.items()] + [
        {"filtro": "Regole", "valore": f"primo isolato per trimestre; percentuale riportabile con almeno "
                                       f"{answer['soglia']} isolati testati"}])
    rename = {"pct": "% R", "basso": "IC95 basso", "alto": "IC95 alto"}
    trend = pd.DataFrame(answer["andamento"]).rename(columns=rename)
    labs = pd.DataFrame(answer["laboratori"]).rename(columns=rename)
    cells = pd.DataFrame(answer["antibiogramma"]["celle"])
    if not cells.empty:
        cells = cells.assign(microorganismo_nome=cells["microorganismo"].map(pathogen_name),
                             antibiotico_nome=cells["antibiotic"].map(antibiotic_name)).rename(columns=rename)
    frames = {"Selezione": selection, "Andamento": trend, "Laboratori": labs, "Antibiogramma": cells}
    forecast = answer["previsione"]
    if forecast.get("disponibile"):
        rows = [dict(f, periodo=f["trimestre"]) for f in forecast["trimestrali"]]
        if forecast.get("dodici_mesi"):
            rows.append(dict(forecast["dodici_mesi"], periodo="prossimi 12 mesi"))
        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.assign(affidabilita=table["semaforo"].map(TRAFFIC_WORDS)).rename(
                columns={"pct": "% R prevista", "basso": "intervallo 80% basso", "alto": "intervallo 80% alto"})
        frames["Previsione"] = table
    return frames
