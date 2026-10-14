"""Canonical isolate-level dataset built from the MIC network Excel export.

The Excel (one row per isolate, ``<ABX>_QUALITATIVO`` / ``<ABX>_QUANTITATIVO``
column pairs) is read once, validated, and stored as Parquet so every model
can reuse the full record (patient, age, sex, ward, material, MIC values),
not only the monthly S/I/R aggregates.

Files written under the dataset folder, one partition per upload:
  isolates/upload_<id>.parquet    one row per isolate, typed columns
  ast/upload_<id>.parquet         one row per isolate x antibiotic tested
  reports/upload_<id>.json        validation report
  reports/upload_<id>_issues.csv  row-level issues (capped)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from micare_sicilia.config import ANTIBIOTIC_CLASS_CODES, INTRINSIC_RESISTANCE_EXCLUSIONS

QUAL_SUFFIX = "_QUALITATIVO"
QUANT_SUFFIX = "_QUANTITATIVO"

REQUIRED_COLUMNS = ["DATA_PRELIEVO", "MICROORGANISMO", "LABORATORIO"]
RECOMMENDED_COLUMNS = [
    "REGIONE", "OSPEDALE", "CODICE_PAZIENTE", "SESSO", "ETA", "NAZIONALITA",
    "PROVINCIA_RESIDENZA", "PAZIENTE_RICOVERATO", "DATA_RICOVERO",
    "REPARTO_DI_RICOVERO", "CODICE_CAMPIONE", "MATERIALE",
]
# Codes kept as text so leading zeros survive (e.g. 00006267, 088009)
CODE_COLUMNS = [
    "REGIONE", "LABORATORIO", "OSPEDALE", "CODICE_PAZIENTE", "SESSO", "NAZIONALITA",
    "PROVINCIA_RESIDENZA", "PAZIENTE_RICOVERATO", "REPARTO_DI_RICOVERO",
    "CODICE_CAMPIONE", "MATERIALE", "MICROORGANISMO",
]

MAX_ISSUE_ROWS = 100_000
MIN_DATE = pd.Timestamp("2000-01-01")

_MIC_RE = re.compile(r"^\s*(<=|>=|<|>|=)?\s*([0-9]+(?:[.,][0-9]+)?)\s*$")
_RESULT_MAP = {
    "S": "S", "I": "I", "R": "R",
    "SENSIBILE": "S", "SENSITIVE": "S", "SUSCEPTIBLE": "S",
    "INTERMEDIO": "I", "INTERMEDIATE": "I",
    "RESISTENTE": "R", "RESISTANT": "R",
}


class DatasetValidationError(ValueError):
    """Raised when the file cannot be imported at all (blocking errors)."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("; ".join(report["errors"]) or "Validazione fallita.")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def is_mic_export(columns) -> bool:
    return any(str(col).strip().upper().endswith(QUAL_SUFFIX) for col in columns)


def read_raw_file(path: str | Path) -> pd.DataFrame:
    """Read every cell as text (codes keep leading zeros, MIC keep operators).

    Uses the Rust-based calamine engine when installed (much faster than
    openpyxl on large workbooks), falling back to openpyxl.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine="python")
    elif suffix in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(path, dtype=str, keep_default_na=False, engine="calamine")
        except ImportError:
            df = pd.read_excel(path, dtype=str, keep_default_na=False)
    else:
        raise ValueError("Formato non supportato. Usa CSV, XLS o XLSX.")
    df.columns = [str(col).strip().upper() for col in df.columns]
    return df.apply(lambda col: col.astype(object).where(col.notna(), "").map(_clean_cell))


def _clean_cell(value) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"nan", "nat", "none"} else text


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_dates(series: pd.Series) -> pd.Series:
    """Parse dd/mm/yyyy and ISO dates without day/month swaps."""
    text = series.astype(str).str.strip()
    out = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%d-%m-%Y", "%d/%m/%y"):
        missing = out.isna() & text.ne("")
        if not missing.any():
            break
        out[missing] = pd.to_datetime(text[missing], format=fmt, errors="coerce")
    # Excel serial numbers (e.g. "44634") when the cell was not formatted as a date
    missing = out.isna() & text.str.fullmatch(r"\d{5}(\.0+)?")
    if missing.any():
        serial = pd.to_numeric(text[missing], errors="coerce")
        out[missing] = pd.to_datetime("1899-12-30") + pd.to_timedelta(serial, unit="D")
    return out


def parse_mic(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split '<=0.5' into ('<=', 0.5). Unparseable values give (None, NaN)."""
    parts = series.astype(str).str.extract(_MIC_RE)
    operator = parts[0].fillna("=").where(parts[1].notna())
    value = pd.to_numeric(parts[1].str.replace(",", ".", regex=False), errors="coerce")
    return operator, value


def normalize_result(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().map(_RESULT_MAP)


def _code(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.upper()
    # Integer-like values exported as floats ("82053.0") lose the ".0"
    return text.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)


# ---------------------------------------------------------------------------
# Build + validate
# ---------------------------------------------------------------------------

def build_and_validate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Return (isolates, ast, report, issues). Raises DatasetValidationError."""
    columns = list(raw.columns)
    antibiotics = sorted(
        col.removesuffix(QUAL_SUFFIX) for col in columns if col.endswith(QUAL_SUFFIX)
    )
    report: dict = {"errors": [], "warnings": [], "info": [], "stats": {}}
    missing_required = [col for col in REQUIRED_COLUMNS if col not in columns]
    if missing_required:
        report["errors"].append(f"Colonne obbligatorie mancanti: {', '.join(missing_required)}")
    if not antibiotics:
        report["errors"].append("Nessuna colonna <ANTIBIOTICO>_QUALITATIVO trovata.")
    if report["errors"]:
        raise DatasetValidationError(report)

    missing_recommended = [col for col in RECOMMENDED_COLUMNS if col not in columns]
    if missing_recommended:
        report["warnings"].append(
            f"Colonne consigliate assenti (alcune analisi non saranno disponibili): {', '.join(missing_recommended)}"
        )
    known = set(REQUIRED_COLUMNS + RECOMMENDED_COLUMNS)
    known |= {f"{a}{QUAL_SUFFIX}" for a in antibiotics} | {f"{a}{QUANT_SUFFIX}" for a in antibiotics}
    unexpected = [col for col in columns if col not in known]
    if unexpected:
        report["info"].append(f"Colonne non riconosciute (conservate così come sono): {', '.join(unexpected)}")
    orphan_quant = [
        col.removesuffix(QUANT_SUFFIX) for col in columns
        if col.endswith(QUANT_SUFFIX) and col.removesuffix(QUANT_SUFFIX) not in antibiotics
    ]
    if orphan_quant:
        report["warnings"].append(f"Colonne _QUANTITATIVO senza _QUALITATIVO: {', '.join(orphan_quant)}")

    issues = _IssueCollector()
    n_rows = len(raw)
    excel_row = pd.Series(np.arange(n_rows) + 2, index=raw.index)  # header is row 1

    # --- isolate table ----------------------------------------------------
    isolates = pd.DataFrame(index=raw.index)
    isolates["riga_excel"] = excel_row
    for col in CODE_COLUMNS:
        isolates[col.lower()] = _code(raw[col]) if col in columns else ""
    for col in unexpected:
        isolates[col.lower()] = raw[col]

    isolates["data_prelievo"] = parse_dates(raw["DATA_PRELIEVO"])
    isolates["data_ricovero"] = parse_dates(raw["DATA_RICOVERO"]) if "DATA_RICOVERO" in columns else pd.NaT
    isolates["eta"] = pd.to_numeric(raw["ETA"].str.replace(",", ".", regex=False), errors="coerce") if "ETA" in columns else np.nan

    empty_row = raw.eq("").all(axis=1)
    issues.add(empty_row, excel_row, "-", raw.iloc[:, 0], "Riga vuota", "info")

    date_text = raw["DATA_PRELIEVO"]
    bad_date = isolates["data_prelievo"].isna() & ~empty_row
    issues.add(bad_date, excel_row, "DATA_PRELIEVO", date_text, "Data prelievo mancante o non valida: riga esclusa", "errore")
    today = pd.Timestamp.today().normalize()
    out_of_range = isolates["data_prelievo"].notna() & (
        (isolates["data_prelievo"] < MIN_DATE) | (isolates["data_prelievo"] > today)
    )
    issues.add(out_of_range, excel_row, "DATA_PRELIEVO", date_text, "Data prelievo fuori intervallo (prima del 2000 o futura)", "avviso")

    no_pathogen = isolates["microorganismo"].eq("") & ~empty_row
    issues.add(no_pathogen, excel_row, "MICROORGANISMO", raw["MICROORGANISMO"], "Microrganismo mancante: riga esclusa", "errore")
    no_lab = isolates["laboratorio"].eq("") & ~empty_row
    issues.add(no_lab, excel_row, "LABORATORIO", raw["LABORATORIO"], "Laboratorio mancante: riga esclusa", "errore")

    if "DATA_RICOVERO" in columns:
        bad_adm = raw["DATA_RICOVERO"].ne("") & isolates["data_ricovero"].isna()
        issues.add(bad_adm, excel_row, "DATA_RICOVERO", raw["DATA_RICOVERO"], "Data ricovero non valida", "avviso")
        adm_after = isolates["data_ricovero"] > isolates["data_prelievo"]
        issues.add(adm_after, excel_row, "DATA_RICOVERO", raw["DATA_RICOVERO"], "Data ricovero successiva al prelievo", "avviso")
    if "SESSO" in columns:
        bad_sex = ~isolates["sesso"].isin(["M", "F", ""])
        issues.add(bad_sex, excel_row, "SESSO", raw["SESSO"], "Sesso diverso da M/F", "avviso")
    if "ETA" in columns:
        bad_age = raw["ETA"].ne("") & (isolates["eta"].isna() | (isolates["eta"] < 0) | (isolates["eta"] > 120))
        issues.add(bad_age, excel_row, "ETA", raw["ETA"], "Età non numerica o fuori 0-120", "avviso")
        isolates.loc[bad_age, "eta"] = np.nan

    valid_row = ~(empty_row | bad_date | no_pathogen | no_lab)

    # --- duplicates --------------------------------------------------------
    test_cols = [c for c in columns if c.endswith(QUAL_SUFFIX) or c.endswith(QUANT_SUFFIX)]
    exact_dup = raw.duplicated(keep="first") & valid_row
    issues.add(exact_dup, excel_row, "-", isolates["codice_campione"], "Riga duplicata identica: esclusa", "avviso")
    valid_row &= ~exact_dup
    if "CODICE_CAMPIONE" in columns:
        key = isolates[["codice_campione", "microorganismo"]]
        dup_sample = valid_row & isolates["codice_campione"].ne("") & key.duplicated(keep=False) & ~raw[test_cols].duplicated(keep=False)
        issues.add(dup_sample, excel_row, "CODICE_CAMPIONE", isolates["codice_campione"],
                   "Stesso campione e microrganismo con antibiogramma diverso", "avviso")

    # --- antibiogram long table -------------------------------------------
    ast_parts = []
    for abx in antibiotics:
        qual_col, quant_col = f"{abx}{QUAL_SUFFIX}", f"{abx}{QUANT_SUFFIX}"
        qual_raw = raw[qual_col]
        quant_raw = raw[quant_col] if quant_col in columns else pd.Series("", index=raw.index)
        result = normalize_result(qual_raw)
        mic_op, mic_val = parse_mic(quant_raw)

        bad_qual = qual_raw.ne("") & result.isna() & valid_row
        issues.add(bad_qual, excel_row, qual_col, qual_raw, "Valore qualitativo diverso da S/I/R", "avviso")
        bad_quant = quant_raw.ne("") & mic_val.isna() & valid_row
        issues.add(bad_quant, excel_row, quant_col, quant_raw, "MIC non interpretabile", "avviso")
        quant_only = quant_raw.ne("") & qual_raw.eq("") & valid_row
        issues.add(quant_only, excel_row, quant_col, quant_raw, "MIC presente senza interpretazione S/I/R", "info")

        tested = valid_row & (result.notna() | mic_val.notna())
        if not tested.any():
            continue
        ast_parts.append(pd.DataFrame({
            "isolate_id": raw.index[tested],
            "antibiotic": abx,
            "result": result[tested],
            "mic_raw": quant_raw[tested],
            "mic_operator": mic_op[tested],
            "mic_value": mic_val[tested],
        }))

    isolates = isolates[valid_row].copy()
    isolates.index.name = "isolate_id"
    isolates = isolates.reset_index()
    ast = (
        pd.concat(ast_parts, ignore_index=True)
        if ast_parts
        else pd.DataFrame(columns=["isolate_id", "antibiotic", "result", "mic_raw", "mic_operator", "mic_value"])
    )

    if isolates.empty or ast["result"].notna().sum() == 0:
        report["errors"].append("Nessun isolato valido con risultati S/I/R dopo la validazione.")
        report["stats"] = {"rows_read": int(n_rows)}
        raise DatasetValidationError(report | {"issue_summary": issues.summary()})

    # --- intrinsic resistance tests (informative) -------------------------
    intrinsic_pairs = {
        f"{pathogen}|{code}"
        for pathogen, classes in INTRINSIC_RESISTANCE_EXCLUSIONS.items()
        for cls in classes
        for code in ANTIBIOTIC_CLASS_CODES.get(cls, [])
    }
    joined = ast.merge(isolates[["isolate_id", "microorganismo"]], on="isolate_id")
    intrinsic = (joined["microorganismo"] + "|" + joined["antibiotic"]).isin(intrinsic_pairs)
    if intrinsic.any():
        combos = joined[intrinsic].groupby(["microorganismo", "antibiotic"]).size()
        report["info"].append(
            "Test su antibiotici con resistenza intrinseca (esclusi dalle previsioni): "
            + ", ".join(f"{p}/{a} ({n})" for (p, a), n in combos.items())
        )

    report["stats"] = _stats(n_rows, isolates, ast, antibiotics)
    pts = report["stats"]
    if pts.get("repeat_isolates", 0):
        report["info"].append(
            f"{pts['repeat_isolates']} isolati ripetuti sullo stesso paziente/microrganismo: "
            "considerare la deduplicazione 'primo isolato' (CLSI M39) prima dell'analisi."
        )
    report["issue_summary"] = issues.summary()
    for item in report["issue_summary"]:
        if item["gravita"] == "errore":
            report["warnings"].append(f"{item['righe']} righe escluse: {item['problema']}")
    return isolates, ast, report, issues.frame()


def _stats(n_rows: int, isolates: pd.DataFrame, ast: pd.DataFrame, antibiotics: list[str]) -> dict:
    stats = {
        "rows_read": int(n_rows),
        "isolates_valid": int(len(isolates)),
        "rows_excluded": int(n_rows - len(isolates)),
        "ast_results": int(ast["result"].notna().sum()),
        "mic_values": int(ast["mic_value"].notna().sum()),
        "antibiotics": antibiotics,
        "date_from": str(isolates["data_prelievo"].min().date()),
        "date_to": str(isolates["data_prelievo"].max().date()),
        "years": sorted(int(y) for y in isolates["data_prelievo"].dt.year.unique()),
        "laboratories": int(isolates["laboratorio"].nunique()),
        "hospitals": int(isolates["ospedale"].replace("", np.nan).nunique()),
        "pathogens": isolates["microorganismo"].value_counts().head(30).to_dict(),
        "isolates_per_year": {int(k): int(v) for k, v in isolates["data_prelievo"].dt.year.value_counts().sort_index().items()},
        "sir_distribution": ast["result"].value_counts().to_dict(),
    }
    has_patient = isolates["codice_paziente"].ne("")
    if has_patient.any():
        stats["patients"] = int(isolates.loc[has_patient, "codice_paziente"].nunique())
        first = first_isolate_mask(isolates)
        stats["first_isolates"] = int(first.sum())
        stats["repeat_isolates"] = int((~first & has_patient).sum())
    return stats


def first_isolate_mask(isolates: pd.DataFrame) -> pd.Series:
    """True for the first isolate per patient x pathogen x calendar year (CLSI M39).

    Rows without a patient code are always kept.
    """
    order = isolates.sort_values(["data_prelievo", "codice_campione"])
    key = [order["codice_paziente"], order["microorganismo"], order["data_prelievo"].dt.year]
    first = ~order.groupby(key, sort=False).cumcount().astype(bool)
    first |= order["codice_paziente"].eq("")
    return first.reindex(isolates.index)


class _IssueCollector:
    def __init__(self):
        self._parts: list[pd.DataFrame] = []
        self._counts: list[dict] = []

    def add(self, mask: pd.Series, excel_row: pd.Series, column: str, values, problem: str, severity: str) -> None:
        mask = mask.fillna(False).astype(bool)
        n = int(mask.sum())
        if n == 0:
            return
        self._counts.append({"problema": problem, "colonna": column, "gravita": severity, "righe": n})
        if sum(len(p) for p in self._parts) >= MAX_ISSUE_ROWS:
            return
        values = values if isinstance(values, pd.Series) else pd.Series(values, index=mask.index)
        self._parts.append(pd.DataFrame({
            "riga_excel": excel_row[mask],
            "colonna": column,
            "valore": values[mask].astype(str),
            "problema": problem,
            "gravita": severity,
        }))

    def summary(self) -> list[dict]:
        order = {"errore": 0, "avviso": 1, "info": 2}
        return sorted(self._counts, key=lambda c: (order.get(c["gravita"], 3), -c["righe"]))

    def frame(self) -> pd.DataFrame:
        if not self._parts:
            return pd.DataFrame(columns=["riga_excel", "colonna", "valore", "problema", "gravita"])
        return pd.concat(self._parts, ignore_index=True).head(MAX_ISSUE_ROWS).sort_values("riga_excel")


# ---------------------------------------------------------------------------
# Long observations for the existing aggregation pipeline
# ---------------------------------------------------------------------------

def to_observations(isolates: pd.DataFrame, ast: pd.DataFrame) -> pd.DataFrame:
    """Same shape as preprocessing.normalize_uploaded_dataframe output."""
    tested = ast[ast["result"].notna()]
    obs = tested.merge(
        isolates[["isolate_id", "data_prelievo", "microorganismo", "laboratorio", "reparto_di_ricovero"]],
        on="isolate_id",
    )
    out = pd.DataFrame({
        "date": obs["data_prelievo"],
        "pathogen": obs["microorganismo"],
        "antibiotic": obs["antibiotic"],
        "laboratory": obs["laboratorio"],
        "ward": obs["reparto_di_ricovero"].fillna(""),
        "result": obs["result"],
        "samples": 1,
    })
    for name, code in [("sensitive_count", "S"), ("intermediate_count", "I"), ("resistant_count", "R")]:
        out[name] = (out["result"] == code).astype(float)
    return out


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def save_dataset(folder: str | Path, upload_id: int, isolates: pd.DataFrame, ast: pd.DataFrame,
                 report: dict, issues: pd.DataFrame) -> None:
    root = Path(folder)
    for sub in ("isolates", "ast", "reports"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    isolates = isolates.assign(upload_id=upload_id)
    ast = ast.assign(upload_id=upload_id)
    isolates.to_parquet(root / "isolates" / f"upload_{upload_id}.parquet", index=False)
    ast.to_parquet(root / "ast" / f"upload_{upload_id}.parquet", index=False)
    save_report(folder, upload_id, report, issues)


def delete_upload_data(folder: str | Path, upload_id: int) -> None:
    """Drop the stored partitions of an upload (keeps its validation report)."""
    for sub in ("isolates", "ast"):
        (Path(folder) / sub / f"upload_{upload_id}.parquet").unlink(missing_ok=True)


def save_report(folder: str | Path, upload_id: int, report: dict, issues: pd.DataFrame | None = None) -> None:
    reports = Path(folder) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"upload_{upload_id}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if issues is not None:
        issues.to_csv(reports / f"upload_{upload_id}_issues.csv", index=False)


def load_report(folder: str | Path, upload_id: int) -> dict | None:
    path = Path(folder) / "reports" / f"upload_{upload_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def issues_path(folder: str | Path, upload_id: int) -> Path:
    return Path(folder) / "reports" / f"upload_{upload_id}_issues.csv"


def load_isolates(folder: str | Path, first_isolate_only: bool = False) -> pd.DataFrame:
    """All stored isolates (every upload). Entry point for any model."""
    isolates = _load_partitions(Path(folder) / "isolates")
    if first_isolate_only and not isolates.empty:
        isolates = isolates[first_isolate_mask(isolates)]
    return isolates


def load_ast_results(folder: str | Path, with_isolate_columns: bool = True,
                     first_isolate_only: bool = False) -> pd.DataFrame:
    """One row per isolate x antibiotic, optionally joined with isolate fields."""
    ast = _load_partitions(Path(folder) / "ast")
    if not with_isolate_columns or ast.empty:
        return ast
    isolates = load_isolates(folder, first_isolate_only=first_isolate_only)
    return ast.merge(isolates, on=["upload_id", "isolate_id"], how="inner")


def _load_partitions(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("upload_*.parquet")) if path.exists() else []
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
