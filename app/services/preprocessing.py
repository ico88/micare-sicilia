from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATE_ALIASES = ["data", "date", "DATA_PRELIEVO", "DATA PRELIEVO", "data_prelievo", "mese", "month"]
PATHOGEN_ALIASES = ["patogeno", "microorganismo", "MICROORGANISMO", "specie", "batterio"]
ANTIBIOTIC_ALIASES = ["antibiotico", "antibiotic", "farmaco"]
LAB_ALIASES = ["laboratorio", "LABORATORIO", "lab", "codice_laboratorio"]
WARD_ALIASES = ["reparto", "REPARTO_DI_RICOVERO", "ward", "unit", "unita_operativa"]
RESULT_ALIASES = ["risultato", "sir", "interpretazione", "esito"]
SAMPLES_ALIASES = ["campioni", "numero_campioni", "n_campioni", "samples", "Totale_Campioni"]


def load_tabular_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Formato non supportato. Usa CSV, XLS o XLSX.")


def find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for alias in aliases:
        found = lower_map.get(alias.lower())
        if found is not None:
            return found
    return None


def normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def normalize_result(value: object) -> str | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip().upper()
    mapping = {
        "SENSIBILE": "S",
        "SENSITIVE": "S",
        "SUSCEPTIBLE": "S",
        "INTERMEDIO": "I",
        "INTERMEDIATE": "I",
        "RESISTENTE": "R",
        "RESISTANT": "R",
    }
    normalized = mapping.get(normalized, normalized)
    return normalized if normalized in {"S", "I", "R"} else None


def _parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")


def normalize_uploaded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-form observations with normalized S/I/R fields."""
    df = df.copy()
    if "MICROORGANISMO" in df.columns and "patogeno" not in df.columns:
        df = df.rename(columns={"MICROORGANISMO": "patogeno"})

    qualitative_cols = [col for col in df.columns if str(col).endswith("_QUALITATIVO")]
    if qualitative_cols:
        return _normalize_wide_mic_dataframe(df, qualitative_cols)

    return _normalize_long_or_aggregated_dataframe(df)


def _normalize_wide_mic_dataframe(df: pd.DataFrame, qualitative_cols: list[str]) -> pd.DataFrame:
    date_col = find_column(df, DATE_ALIASES)
    pathogen_col = find_column(df, PATHOGEN_ALIASES)
    lab_col = find_column(df, LAB_ALIASES)
    ward_col = find_column(df, WARD_ALIASES)

    missing = [
        name
        for name, col in [("data", date_col), ("patogeno", pathogen_col), ("laboratorio", lab_col)]
        if col is None
    ]
    if missing:
        raise ValueError(f"Colonne obbligatorie mancanti: {', '.join(missing)}")

    rows = []
    base = df.copy()
    base["_date"] = _parse_dates(base[date_col])
    base["_pathogen"] = base[pathogen_col].map(normalize_code)
    base["_laboratory"] = base[lab_col].map(normalize_code)
    base["_ward"] = base[ward_col].map(normalize_code) if ward_col else ""

    for qualitative_col in qualitative_cols:
        antibiotic = str(qualitative_col).removesuffix("_QUALITATIVO").strip().upper()
        temp = base[["_date", "_pathogen", "_laboratory", "_ward", qualitative_col]].copy()
        temp = temp.rename(columns={qualitative_col: "result"})
        temp["result"] = temp["result"].map(normalize_result)
        temp = temp.dropna(subset=["_date", "result"])
        if temp.empty:
            continue
        temp["antibiotic"] = antibiotic
        temp["samples"] = 1
        temp["sensitive_count"] = np.where(temp["result"] == "S", 1.0, 0.0)
        temp["intermediate_count"] = np.where(temp["result"] == "I", 1.0, 0.0)
        temp["resistant_count"] = np.where(temp["result"] == "R", 1.0, 0.0)
        rows.append(temp)

    if not rows:
        raise ValueError("Nessun risultato S/I/R valido trovato nelle colonne *_QUALITATIVO.")

    out = pd.concat(rows, ignore_index=True)
    return out.rename(
        columns={
            "_date": "date",
            "_pathogen": "pathogen",
            "_laboratory": "laboratory",
            "_ward": "ward",
        }
    )[
        [
            "date",
            "pathogen",
            "antibiotic",
            "laboratory",
            "ward",
            "result",
            "samples",
            "sensitive_count",
            "intermediate_count",
            "resistant_count",
        ]
    ]


def _numeric_column(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    col = find_column(df, names)
    if col is None:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def _normalize_long_or_aggregated_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    date_col = find_column(df, DATE_ALIASES)
    pathogen_col = find_column(df, PATHOGEN_ALIASES)
    antibiotic_col = find_column(df, ANTIBIOTIC_ALIASES)
    lab_col = find_column(df, LAB_ALIASES)
    ward_col = find_column(df, WARD_ALIASES)
    result_col = find_column(df, RESULT_ALIASES)

    missing = [
        name
        for name, col in [
            ("data", date_col),
            ("patogeno", pathogen_col),
            ("antibiotico", antibiotic_col),
            ("laboratorio", lab_col),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(f"Colonne obbligatorie mancanti: {', '.join(missing)}")

    out = pd.DataFrame(
        {
            "date": _parse_dates(df[date_col]),
            "pathogen": df[pathogen_col].map(normalize_code),
            "antibiotic": df[antibiotic_col].map(normalize_code),
            "laboratory": df[lab_col].map(normalize_code),
            "ward": df[ward_col].map(normalize_code) if ward_col else "",
        }
    )
    out["result"] = df[result_col].map(normalize_result) if result_col else None
    samples = _numeric_column(df, SAMPLES_ALIASES)
    out["samples"] = samples.fillna(1).astype(int) if samples is not None else 1

    s_count = _numeric_column(df, ["sensibili_count", "Conteggio_S", "sensitive_count", "S_count"])
    i_count = _numeric_column(df, ["intermedi_count", "Conteggio_I", "intermediate_count", "I_count"])
    r_count = _numeric_column(df, ["resistenti_count", "Conteggio_R", "resistant_count", "R_count"])

    if s_count is not None or i_count is not None or r_count is not None:
        out["sensitive_count"] = s_count.fillna(0) if s_count is not None else 0.0
        out["intermediate_count"] = i_count.fillna(0) if i_count is not None else 0.0
        out["resistant_count"] = r_count.fillna(0) if r_count is not None else 0.0
    else:
        out["sensitive_count"] = np.where(out["result"] == "S", out["samples"], 0.0)
        out["intermediate_count"] = np.where(out["result"] == "I", out["samples"], 0.0)
        out["resistant_count"] = np.where(out["result"] == "R", out["samples"], 0.0)

    return out.dropna(subset=["date"]).reset_index(drop=True)


def aggregate_observations(observations: pd.DataFrame) -> pd.DataFrame:
    df = observations.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M")
    grouped = (
        df.groupby(["month", "pathogen", "antibiotic", "laboratory", "ward"], dropna=False)
        .agg(
            samples=("samples", "sum"),
            sensitive_count=("sensitive_count", "sum"),
            intermediate_count=("intermediate_count", "sum"),
            resistant_count=("resistant_count", "sum"),
        )
        .reset_index()
    )
    total = grouped[["sensitive_count", "intermediate_count", "resistant_count"]].sum(axis=1)
    total = total.replace(0, np.nan)
    grouped["sensitive_pct"] = (grouped["sensitive_count"] / total * 100).fillna(0)
    grouped["intermediate_pct"] = (grouped["intermediate_count"] / total * 100).fillna(0)
    grouped["resistant_pct"] = (grouped["resistant_count"] / total * 100).fillna(0)
    return grouped.sort_values(["pathogen", "antibiotic", "laboratory", "month"]).reset_index(drop=True)
