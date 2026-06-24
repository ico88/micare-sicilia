from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ANTIBIOTIC_CLASS_CODES,
    INTRINSIC_RESISTANCE_EXCLUSIONS,
    MIN_SAMPLES_PER_MONTH,
    PAZIENTE_RICOVERATO_CODE,
    REPARTO_UTI_CODE,
    START_DATE,
)


@dataclass(frozen=True)
class DataQualityReport:
    total_rows: int
    clean_rows: int
    duplicate_samples: int
    consolidated_samples: int
    true_conflict_samples: int


def read_excel_files(paths: list[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        xl = pd.ExcelFile(path)
        # Usa il primo foglio dati (salta fogli di conteggio come 'totali')
        data_sheet = xl.sheet_names[0]
        frame = pd.read_excel(xl, sheet_name=data_sheet)
        frames.append(frame)

    if not frames:
        raise ValueError("Nessun file Excel indicato.")

    # Normalizza e converte la data per ogni file prima del concat per evitare conflitti di tipo
    normalized = []
    for frame in frames:
        frame = frame.rename(columns={"DATA PRELIEVO": "DATA_PRELIEVO"})
        if "MICROORGANISMO" in frame.columns and "patogeno" not in frame.columns:
            frame = frame.rename(columns={"MICROORGANISMO": "patogeno"})
        if "DATA_PRELIEVO" in frame.columns:
            frame["DATA_PRELIEVO"] = pd.to_datetime(
                frame["DATA_PRELIEVO"], dayfirst=True, errors="coerce"
            )
        normalized.append(frame)

    df = pd.concat(normalized, ignore_index=True)

    required = {"DATA_PRELIEVO", "patogeno", "LABORATORIO"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Colonne obbligatorie mancanti: {', '.join(missing)}")

    return df


def antibiotic_qualitative_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.endswith("_QUALITATIVO")]


def _antibiotic_codes_for_classes(class_names: list[str], available_columns: set[str]) -> list[str]:
    codes: set[str] = set()
    for class_name in class_names:
        codes.update(ANTIBIOTIC_CLASS_CODES.get(class_name, []))
    return [f"{code}_QUALITATIVO" for code in codes if f"{code}_QUALITATIVO" in available_columns]


def apply_intrinsic_resistance_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    available_columns = set(df.columns)

    for species, class_names in INTRINSIC_RESISTANCE_EXCLUSIONS.items():
        condition = df["patogeno"] == species
        for qualitative_col in _antibiotic_codes_for_classes(class_names, available_columns):
            quantitative_col = qualitative_col.replace("_QUALITATIVO", "_QUANTITATIVO")
            df.loc[condition, qualitative_col] = np.nan
            if quantitative_col in df.columns:
                df.loc[condition, quantitative_col] = np.nan

    return df


def consolidate_duplicates(
    df: pd.DataFrame,
    conflict_output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    key_columns = ["DATA_PRELIEVO", "CODICE_CAMPIONE", "LABORATORIO", "patogeno"]
    qualitative_columns = antibiotic_qualitative_columns(df)

    missing_keys = [col for col in key_columns if col not in df.columns]
    if missing_keys:
        report = DataQualityReport(
            total_rows=len(df),
            clean_rows=len(df),
            duplicate_samples=0,
            consolidated_samples=0,
            true_conflict_samples=0,
        )
        return df.copy(), report

    duplicated = df[df.duplicated(subset=key_columns, keep=False)].copy()
    unique = df.drop_duplicates(subset=key_columns, keep=False).copy()

    if duplicated.empty:
        report = DataQualityReport(
            total_rows=len(df),
            clean_rows=len(df),
            duplicate_samples=0,
            consolidated_samples=0,
            true_conflict_samples=0,
        )
        return df.copy(), report

    hierarchy = {"R": 3, "I": 2, "S": 1}
    consolidated_rows = []
    conflict_groups = []

    for _, group in duplicated.groupby(key_columns):
        merged_row = group.iloc[0].copy()
        has_true_conflict = False

        for qualitative_col in qualitative_columns:
            values = group[qualitative_col].dropna()
            if values.empty:
                merged_row[qualitative_col] = np.nan
                continue

            final_value = np.nan
            max_priority = -1

            for value in values.unique():
                priority = hierarchy.get(value, 0)
                if priority == 0 and max_priority > 0:
                    has_true_conflict = True
                    break
                if priority > max_priority:
                    max_priority = priority
                    final_value = value

            if has_true_conflict:
                break

            merged_row[qualitative_col] = final_value
            quantitative_col = qualitative_col.replace("_QUALITATIVO", "_QUANTITATIVO")
            if quantitative_col in group.columns:
                matching_rows = group[group[qualitative_col] == final_value]
                mic_values = matching_rows[quantitative_col].dropna()
                merged_row[quantitative_col] = mic_values.iloc[0] if not mic_values.empty else np.nan

        if has_true_conflict:
            conflict_groups.append(group)
        else:
            consolidated_rows.append(merged_row)

    consolidated = pd.DataFrame(consolidated_rows)
    clean = pd.concat([unique, consolidated], ignore_index=True)

    true_conflict_samples = 0
    if conflict_groups:
        conflicts = pd.concat(conflict_groups, ignore_index=True)
        true_conflict_samples = len(conflicts.drop_duplicates(subset=key_columns))
        if conflict_output_path is not None:
            Path(conflict_output_path).parent.mkdir(parents=True, exist_ok=True)
            conflicts.to_excel(conflict_output_path, index=False)

    report = DataQualityReport(
        total_rows=len(df),
        clean_rows=len(clean),
        duplicate_samples=len(duplicated.drop_duplicates(subset=key_columns)),
        consolidated_samples=len(consolidated),
        true_conflict_samples=true_conflict_samples,
    )
    return clean, report


def prepare_clean_dataframe(
    df: pd.DataFrame,
    conflict_output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    df = df[df["DATA_PRELIEVO"] >= START_DATE].copy()
    df["patogeno"] = df["patogeno"].astype(str).str.strip()

    qualitative_columns = antibiotic_qualitative_columns(df)
    if not qualitative_columns:
        raise ValueError("Nessuna colonna *_QUALITATIVO trovata nel dataset.")

    df = df.dropna(subset=qualitative_columns, how="all")
    df = apply_intrinsic_resistance_exclusions(df)
    return consolidate_duplicates(df, conflict_output_path=conflict_output_path)


def compute_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola covariati mensili per laboratorio dal dataset grezzo.

    Restituisce un DataFrame con colonne:
    - data: fine mese
    - LABORATORIO
    - pct_icu: % campioni da reparto UTI (codice REPARTO_UTI_CODE)
    - pct_inpatient: % campioni da pazienti ricoverati ordinari
    """
    temp = df[["DATA_PRELIEVO", "LABORATORIO", "REPARTO_DI_RICOVERO", "PAZIENTE_RICOVERATO"]].copy()
    temp = temp.dropna(subset=["DATA_PRELIEVO", "LABORATORIO"])

    temp["mese"] = temp["DATA_PRELIEVO"].dt.to_period("M").dt.to_timestamp("M")
    temp["is_icu"] = (temp["REPARTO_DI_RICOVERO"] == REPARTO_UTI_CODE).astype(float)
    temp["is_inpatient"] = (temp["PAZIENTE_RICOVERATO"] == PAZIENTE_RICOVERATO_CODE).astype(float)

    cov = (
        temp.groupby(["mese", "LABORATORIO"])
        .agg(
            n_tot=("is_icu", "count"),
            n_icu=("is_icu", "sum"),
            n_inpatient=("is_inpatient", "sum"),
        )
        .reset_index()
    )
    cov["pct_icu"] = (cov["n_icu"] / cov["n_tot"] * 100).round(2)
    cov["pct_inpatient"] = (cov["n_inpatient"] / cov["n_tot"] * 100).round(2)
    return cov.rename(columns={"mese": "data"})[["data", "LABORATORIO", "pct_icu", "pct_inpatient"]]


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    qualitative_columns = antibiotic_qualitative_columns(df)
    frames = []

    for qualitative_col in qualitative_columns:
        antibiotic = qualitative_col.removesuffix("_QUALITATIVO")
        temp = df[["DATA_PRELIEVO", "patogeno", "LABORATORIO", qualitative_col]].copy()
        temp = temp.rename(columns={qualitative_col: "risultato"})
        temp["antibiotico"] = antibiotic
        temp = temp.dropna(subset=["risultato"])

        if temp.empty:
            continue

        counts = temp.groupby(
            [
                pd.Grouper(key="DATA_PRELIEVO", freq="ME"),
                "patogeno",
                "LABORATORIO",
                "antibiotico",
                "risultato",
            ]
        ).size().unstack(fill_value=0)

        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        monthly = pd.merge(
            counts.reset_index(),
            percentages.reset_index(),
            on=["DATA_PRELIEVO", "patogeno", "LABORATORIO", "antibiotico"],
            suffixes=("_count", "_pct"),
        )
        frames.append(monthly)

    if not frames:
        raise ValueError("Nessun risultato antibiotico aggregabile trovato.")

    aggregated = pd.concat(frames, ignore_index=True)
    aggregated = aggregated.rename(
        columns={
            "R_count": "Conteggio_R",
            "I_count": "Conteggio_I",
            "S_count": "Conteggio_S",
            "R_pct": "resistenti",
            "I_pct": "intermedi",
            "S_pct": "sensibili",
            "DATA_PRELIEVO": "data",
        }
    )

    for column in ["Conteggio_R", "Conteggio_I", "Conteggio_S", "resistenti", "intermedi", "sensibili"]:
        if column not in aggregated.columns:
            aggregated[column] = 0.0

    aggregated["Totale_Campioni"] = (
        aggregated["Conteggio_R"] + aggregated["Conteggio_I"] + aggregated["Conteggio_S"]
    )
    aggregated["data"] = pd.to_datetime(aggregated["data"])

    # Filtra mesi con campioni insufficienti per una stima percentuale affidabile
    aggregated = aggregated[aggregated["Totale_Campioni"] >= MIN_SAMPLES_PER_MONTH].copy()

    aggregated["combinazione_unica"] = (
        aggregated["patogeno"].astype(str)
        + "_"
        + aggregated["LABORATORIO"].astype(str)
        + "_"
        + aggregated["antibiotico"].astype(str)
    )

    # Unisci covariati ICU/inpatient se disponibili nel dataframe originale
    if "REPARTO_DI_RICOVERO" in df.columns and "PAZIENTE_RICOVERATO" in df.columns:
        covariates = compute_covariates(df)
        aggregated = aggregated.merge(covariates, on=["data", "LABORATORIO"], how="left")
    else:
        aggregated["pct_icu"] = np.nan
        aggregated["pct_inpatient"] = np.nan

    return aggregated.sort_values(["combinazione_unica", "data"]).reset_index(drop=True)


def load_and_aggregate(
    paths: list[str | Path],
    conflict_output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    raw = read_excel_files(paths)
    clean, report = prepare_clean_dataframe(raw, conflict_output_path=conflict_output_path)
    # Passa il raw (non clean) per calcolare i covariati su tutti i campioni del mese,
    # inclusi quelli senza esito antibiotico (es. colture negative)
    aggregated = aggregate_monthly(clean)
    if "REPARTO_DI_RICOVERO" in raw.columns and "PAZIENTE_RICOVERATO" in raw.columns:
        raw_dated = raw.copy()
        raw_dated["DATA_PRELIEVO"] = pd.to_datetime(
            raw_dated["DATA_PRELIEVO"], format="mixed", dayfirst=True, errors="coerce"
        )
        raw_dated = raw_dated[raw_dated["DATA_PRELIEVO"] >= START_DATE]
        covariates = compute_covariates(raw_dated)
        aggregated = aggregated.drop(columns=["pct_icu", "pct_inpatient"], errors="ignore")
        aggregated = aggregated.merge(covariates, on=["data", "LABORATORIO"], how="left")
    return aggregated, report
