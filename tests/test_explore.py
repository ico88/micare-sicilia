import io

import pandas as pd

from app import create_app
from app.services import dataset_store, explore


def _store(folder):
    iso = pd.DataFrame([
        # isolate_id, lab, patient, pathogen, date, ward, patient type, sex, age
        (0, "IT1", "P1", "KLEPNE", "2023-01-10", "49", "1", "M", 70),
        (1, "IT1", "P1", "KLEPNE", "2023-02-15", "26", "1", "M", 70),  # same patient and quarter: not counted
        (2, "IT1", "P2", "KLEPNE", "2023-02-20", "26", "3", "F", 40),
        (3, "IT2", "P3", "KLEPNE", "2023-04-01", "26", "1", "F", 85),
        (4, "IT2", "P4", "ESCCOL", "2023-05-01", "26", "1", "", None),
        (5, "IT2", "P5", "ESCCOL", "2023-05-02", "26", "5", "M", 30),  # non-hospitalised: other population
    ], columns=["isolate_id", "laboratorio", "codice_paziente", "microorganismo", "data_prelievo",
                "reparto_di_ricovero", "paziente_ricoverato", "sesso", "eta"])
    iso["data_prelievo"] = pd.to_datetime(iso["data_prelievo"])
    iso["codice_campione"] = iso["isolate_id"].astype(str)
    iso["materiale"] = "1"
    ast = pd.DataFrame([
        (0, "MEM", "R", "R"), (1, "MEM", "S", "S"), (2, "MEM", "S", "S"), (3, "MEM", "R", "S"),
        (0, "AMK", "S", "S"), (4, "MEM", "S", "S"), (5, "MEM", "S", "S"),
        (0, "VAN", "R", "R"),  # intrinsic: never counted
    ], columns=["isolate_id", "antibiotic", "result", "result_harmonized"])
    ast["intrinsic"] = ast["antibiotic"].eq("VAN")
    dataset_store.save_dataset(folder, 1, iso, ast, {}, pd.DataFrame())
    return folder


def test_cube_counts_first_isolates_on_every_dimension(tmp_path):
    results, counted = explore.build_cube(_store(tmp_path / "dataset"))
    kle = counted[counted["microorganismo"] == "KLEPNE"]
    assert kle["isolati"].sum() == 3  # P1 once in 2023Q1
    icu = kle[kle["reparto"] == explore.ICU]
    assert icu["isolati"].sum() == 1 and icu["eta"].iloc[0] == "65-79 anni"
    assert "VAN" not in set(results["antibiotic"].astype(str))
    mem = results[(results["antibiotic"] == "MEM") & (results["microorganismo"] == "KLEPNE")]
    assert (mem["n_reported"].sum(), mem["r_reported"].sum(), mem["r_harmonized"].sum()) == (3, 2, 1)
    assert set(results["popolazione"].astype(str)) == {"invasivi", "invasivi non ricoverati"}
    # aggregated counts only: no patient or sample field leaves the archive
    assert not {"codice_paziente", "codice_campione", "isolate_id"} & (set(results.columns) | set(counted.columns))


def test_query_answers_every_view_of_one_selection(tmp_path):
    results, counted = explore.build_cube(_store(tmp_path / "dataset"))
    answer = explore.query(results, counted, {"lettura": "reported"})
    assert answer["selezione"]["popolazione"] == "invasivi"
    assert answer["selezione"]["microorganismo"] == "KLEPNE"  # the pathogen with most isolates
    assert answer["selezione"]["antibiotico"] == "MEM"        # its most tested antibiotic
    combo = answer["numeri"]["combinazione"]
    assert (combo["testati"], combo["resistenti"], combo["riportabile"]) == (3, 2, False)
    assert [r["trimestre"] for r in answer["andamento"]] == ["2023Q1", "2023Q2"]
    assert [lab["laboratorio"] for lab in answer["laboratori"]] == ["IT2", "IT1"]  # highest %R first
    heat = {(c["antibiotic"], c["microorganismo"]): c["testati"] for c in answer["antibiogramma"]["celle"]}
    assert heat == {("MEM", "KLEPNE"): 3, ("AMK", "KLEPNE"): 1, ("MEM", "ESCCOL"): 1}

    one_lab = explore.query(results, counted, {"laboratorio": "IT1", "eta": "18-64 anni", "lettura": "reported"})
    assert one_lab["numeri"]["combinazione"]["testati"] == 1
    later = explore.query(results, counted, {"da": "2023Q2"})
    assert [r["trimestre"] for r in later["andamento"]] == ["2023Q2"]
    assert later["numeri"]["combinazione"]["resistenti"] == 0  # harmonised reading by default


def test_dashboard_pages(tmp_path):
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path}/t.sqlite",
                      "UPLOAD_FOLDER": str(tmp_path / "data"), "DATASET_FOLDER": str(tmp_path / "dataset"),
                      "FORECAST_FOLDER": str(tmp_path / "prev"), "EXPLORE_FOLDER": str(tmp_path / "cube")})
    client = app.test_client()
    assert "archivio è vuoto" in client.get("/consultazione").get_data(as_text=True)

    _store(tmp_path / "dataset")
    explore.load_cube(app.config["DATASET_FOLDER"], app.config["EXPLORE_FOLDER"])
    page = client.get("/consultazione").get_data(as_text=True)
    assert 'id="filters"' in page
    data = client.get("/consultazione/dati?microorganismo=ESCCOL").get_json()
    assert data["nomi"]["microorganismo"] == "Escherichia coli" and data["numeri"]["isolati"] == 1
    sheet = client.get("/consultazione/dati.xlsx")
    assert sheet.status_code == 200
    frames = pd.read_excel(io.BytesIO(sheet.data), sheet_name=None)
    assert set(frames) == {"Selezione", "Andamento", "Laboratori", "Antibiogramma"}

    # a new upload makes the stored cube stale
    assert explore.cube_ready(app.config["DATASET_FOLDER"], app.config["EXPLORE_FOLDER"])
    dataset_store.save_dataset(app.config["DATASET_FOLDER"], 2, *_frames_of_one_isolate(), {}, pd.DataFrame())
    assert not explore.cube_ready(app.config["DATASET_FOLDER"], app.config["EXPLORE_FOLDER"])


def _frames_of_one_isolate():
    iso = pd.DataFrame([(0, "IT3", "P9", "KLEPNE", pd.Timestamp("2023-03-01"), "0")],
                       columns=["isolate_id", "laboratorio", "codice_paziente", "microorganismo", "data_prelievo",
                                "codice_campione"])
    ast = pd.DataFrame([(0, "MEM", "S", "S", False)],
                       columns=["isolate_id", "antibiotic", "result", "result_harmonized", "intrinsic"])
    return iso, ast
