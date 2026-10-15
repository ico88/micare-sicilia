import pandas as pd

from app.services import dataset_store
from app.services.series import build_series, cumulative_antibiogram


def _store(tmp_path):
    iso = pd.DataFrame([
        # isolate_id, lab, patient, pathogen, date, ward
        (0, "IT1", "P1", "ACISPP", "2022-01-10", "26"),
        (1, "IT1", "P1", "ACISPP", "2022-02-15", "49"),   # same patient, same quarter: not a first isolate
        (2, "IT1", "P2", "ACISPP", "2022-02-20", "49"),
        (3, "IT1", "P1", "ACISPP", "2022-07-01", "26"),   # same patient, next-but-one quarter: first again
        (4, "IT2", "P1", "ACISPP", "2022-03-01", "26"),   # same code in another lab: another patient
    ], columns=["isolate_id", "laboratorio", "codice_paziente", "microorganismo", "data_prelievo", "reparto_di_ricovero"])
    iso["data_prelievo"] = pd.to_datetime(iso["data_prelievo"])
    iso["codice_campione"] = iso["isolate_id"].astype(str)
    ast = pd.DataFrame([
        (0, "MEM", "R", "R"), (1, "MEM", "S", "S"), (2, "MEM", "S", "S"), (3, "MEM", "R", "R"),
        (4, "MEM", "R", "S"),
        (0, "AMP", "R", "R"),  # intrinsic for Acinetobacter: never in a series
    ], columns=["isolate_id", "antibiotic", "result", "result_harmonized"])
    ast["intrinsic"] = ast["antibiotic"].eq("AMP")
    dataset_store.save_dataset(tmp_path, 1, iso, ast, {}, pd.DataFrame())
    return tmp_path


def test_quarterly_series_counts_first_isolates_and_pools_wards(tmp_path):
    s = build_series(_store(tmp_path), "Q").set_index(["laboratorio", "antibiotic", "period"])
    q1 = s.loc[("IT1", "MEM", "2022Q1")]
    assert (q1["n_reported"], q1["r_reported"]) == (2, 1)  # P1 once (first: R) + P2 (S), wards 26 and 49 pooled
    assert q1["pct_r_reported"] == 50.0
    assert s.loc[("IT1", "MEM", "2022Q2"), "n_reported"] == 0  # explicit empty quarter
    assert s.loc[("IT1", "MEM", "2022Q3"), "r_reported"] == 1
    assert "AMP" not in s.index.get_level_values("antibiotic")
    # Readings differ where the MIC re-read changed the category
    it2 = s.loc[("IT2", "MEM", "2022Q1")]
    assert (it2["r_reported"], it2["r_harmonized"]) == (1, 0)
    assert not q1["reportable"]  # fewer than 30 isolates


def test_antibiogram_uses_first_isolate_per_year(tmp_path):
    a = cumulative_antibiogram(_store(tmp_path)).set_index(["laboratorio", "antibiotic", "anno"])
    row = a.loc[("IT1", "MEM", 2022)]
    assert (row["testati"], row["r"]) == (2, 1)  # P1 counted once in the year
    assert not row["riportabile"]


def test_blood_and_urine_are_separate_series(tmp_path):
    iso = pd.DataFrame([
        (0, "IT1", "P1", "ESCCOL", "2022-02-01", "1"),   # blood
        (1, "IT1", "P2", "ESCCOL", "2023-02-01", "1"),   # blood
        (2, "IT1", "P2", "ESCCOL", "2023-02-03", "3"),   # same patient, urine: first in its own population
        (3, "IT1", "P3", "ESCCOL", "2023-03-01", "3"),
    ], columns=["isolate_id", "laboratorio", "codice_paziente", "microorganismo", "data_prelievo", "materiale"])
    iso["data_prelievo"] = pd.to_datetime(iso["data_prelievo"])
    iso["codice_campione"] = iso["isolate_id"].astype(str)
    ast = pd.DataFrame([(i, "CTX", r, r) for i, r in [(0, "R"), (1, "R"), (2, "S"), (3, "S")]],
                       columns=["isolate_id", "antibiotic", "result", "result_harmonized"])
    ast["intrinsic"] = False
    dataset_store.save_dataset(tmp_path, 1, iso, ast, {}, pd.DataFrame())

    s = build_series(tmp_path, "Q").set_index(["popolazione", "period"])
    assert s.loc[("invasivi", "2023Q1"), "pct_r_reported"] == 100.0
    assert (s.loc[("urine", "2023Q1"), "n_reported"], s.loc[("urine", "2023Q1"), "r_reported"]) == (2, 0)
    # Urine exist only from 2023: no empty urine quarters in 2022
    assert "2022Q1" not in s.loc["urine"].index
    assert s.loc[("invasivi", "2022Q2"), "n_reported"] == 0
