"""Adapter that bridges the Flask app's aggregated DB format to the original
Prophet training pipeline in src/micare_sicilia/model.py."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from micare_sicilia.model import train_forecasts

logger = logging.getLogger(__name__)

_PROPHET_SUBFOLDER = "prophet"


def db_to_model_format(aggregated: pd.DataFrame) -> pd.DataFrame:
    """Convert aggregated DB DataFrame to the format expected by model.py.

    DB columns  -> model.py columns
    month       -> data
    sensitive_pct, intermediate_pct, resistant_pct -> sensibili, intermedi, resistenti
    combinazione_unica is created as "{pathogen}_{laboratory}_{antibiotic}"
    """
    df = aggregated.copy()
    df["data"] = pd.to_datetime(df["month"])
    df["sensibili"] = df["sensitive_pct"]
    df["intermedi"] = df["intermediate_pct"]
    df["resistenti"] = df["resistant_pct"]
    df["combinazione_unica"] = (
        df["pathogen"].astype(str)
        + "_"
        + df["laboratory"].astype(str)
        + "_"
        + df["antibiotic"].astype(str)
    )
    return df


def run_prophet_training(
    aggregated: pd.DataFrame,
    model_folder: str | Path,
    progress_callback: Callable[[int, str, str], None] | None = None,
) -> dict[str, Any]:
    """Run full per-combination Prophet training and return a summary dict."""
    output_dir = Path(model_folder) / _PROPHET_SUBFOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(35, "Prophet — preparazione dati", "Adatto il formato dati per Prophet.")

    model_df = db_to_model_format(aggregated)
    n_combinations = model_df["combinazione_unica"].nunique()
    logger.info("Avvio addestramento Prophet su %d combinazioni", n_combinations)

    if progress_callback:
        progress_callback(
            40,
            "Prophet — addestramento",
            f"Addestro modelli Prophet per {n_combinations} combinazioni. "
            "Operazione lenta (potenzialmente ore) — il progresso si aggiornerà al termine.",
            work_total=n_combinations,
            work_completed=0,
            work_label="combinazioni Prophet",
        )

    summary = train_forecasts(model_df, output_dir)

    result = {
        "prophet_combinations": summary.trained_resistant_models,
        "prophet_attempted": summary.attempted_combinations,
        "prophet_output_dir": str(summary.output_dir),
        "baseline_comparison": summary.baseline_comparison,
    }
    logger.info(
        "Prophet training completato: %d/%d combinazioni",
        summary.trained_resistant_models,
        summary.attempted_combinations,
    )
    return result


def prophet_output_dir(model_folder: str | Path) -> Path:
    return Path(model_folder) / _PROPHET_SUBFOLDER
