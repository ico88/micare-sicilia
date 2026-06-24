"""Test per le nuove funzionalità: covariati, soglia campioni, lettura multi-formato."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from micare_sicilia.config import MIN_SAMPLES_PER_MONTH, REPARTO_UTI_CODE
from micare_sicilia.data import aggregate_monthly, compute_covariates, read_excel_files


def _make_raw_df(n: int = 50, year: int = 2022) -> pd.DataFrame:
    """Crea un DataFrame grezzo minimale per i test."""
    rng = np.random.default_rng(42)
    months = pd.date_range(f"{year}-01-31", periods=12, freq="ME")
    dates = rng.choice(months, size=n)
    return pd.DataFrame(
        {
            "DATA_PRELIEVO": dates,
            "patogeno": rng.choice(["ESCCOL", "KLEPNE"], size=n),
            "LABORATORIO": rng.choice(["IT049", "IT133"], size=n),
            "REPARTO_DI_RICOVERO": rng.choice([REPARTO_UTI_CODE, 101, 43], size=n),
            "PAZIENTE_RICOVERATO": rng.choice([1, 5], size=n),
            "CODICE_CAMPIONE": [f"CAMP{i}" for i in range(n)],
            "AMK_QUALITATIVO": rng.choice(["S", "I", "R"], size=n),
            "MEM_QUALITATIVO": rng.choice(["S", "R"], size=n),
        }
    )


def test_compute_covariates_columns():
    raw = _make_raw_df()
    cov = compute_covariates(raw)
    assert "pct_icu" in cov.columns
    assert "pct_inpatient" in cov.columns
    assert "data" in cov.columns
    assert "LABORATORIO" in cov.columns


def test_compute_covariates_range():
    raw = _make_raw_df(200)
    cov = compute_covariates(raw)
    assert (cov["pct_icu"] >= 0).all()
    assert (cov["pct_icu"] <= 100).all()
    assert (cov["pct_inpatient"] >= 0).all()
    assert (cov["pct_inpatient"] <= 100).all()


def test_aggregate_monthly_min_samples_filter():
    """Tutti i record aggregati devono avere Totale_Campioni >= MIN_SAMPLES_PER_MONTH."""
    # Costruisce un df pulito con abbastanza campioni su alcune combo
    rng = np.random.default_rng(0)
    n = 500
    months = pd.date_range("2022-01-31", periods=12, freq="ME")
    raw = pd.DataFrame(
        {
            "DATA_PRELIEVO": rng.choice(months, size=n),
            "patogeno": ["ESCCOL"] * n,
            "LABORATORIO": ["IT049"] * n,
            "REPARTO_DI_RICOVERO": [REPARTO_UTI_CODE] * n,
            "PAZIENTE_RICOVERATO": [1] * n,
            "CODICE_CAMPIONE": [f"C{i}" for i in range(n)],
            "AMK_QUALITATIVO": rng.choice(["S", "I", "R"], size=n),
        }
    )
    agg = aggregate_monthly(raw)
    assert (agg["Totale_Campioni"] >= MIN_SAMPLES_PER_MONTH).all()


def test_aggregate_monthly_covariates_attached():
    """Il dataset aggregato deve includere colonne pct_icu e pct_inpatient."""
    raw = _make_raw_df(300)
    agg = aggregate_monthly(raw)
    assert "pct_icu" in agg.columns
    assert "pct_inpatient" in agg.columns


def test_aggregate_monthly_compositional_sum():
    """resistenti + intermedi + sensibili deve essere ~100 per ogni riga."""
    raw = _make_raw_df(300)
    agg = aggregate_monthly(raw)
    if agg.empty:
        pytest.skip("Nessun mese supera la soglia minima campioni con questo seed")
    total = agg["resistenti"] + agg["intermedi"] + agg["sensibili"]
    assert (total.round(1) == 100.0).all(), f"Somma compositionale fallita:\n{total[total.round(1) != 100.0]}"


def test_read_excel_files_normalizes_column_name(tmp_path):
    """Verifica che 'DATA PRELIEVO' (con spazio) venga normalizzato a DATA_PRELIEVO."""
    df = pd.DataFrame(
        {
            "DATA PRELIEVO": ["01/01/2022", "15/06/2022"],
            "MICROORGANISMO": ["ESCCOL", "KLEPNE"],
            "LABORATORIO": ["IT049", "IT049"],
        }
    )
    path = tmp_path / "test.xlsx"
    df.to_excel(path, index=False)
    result = read_excel_files([path])
    assert "DATA_PRELIEVO" in result.columns
    assert "DATA PRELIEVO" not in result.columns
    assert "patogeno" in result.columns
