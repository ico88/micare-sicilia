import numpy as np
import pandas as pd
import pytest

from app import create_app, db
from app.services import dataset_store
from app.services.forecast_run import compute_forecasts, load_forecasts


def _store(folder, data_end="2024-12-30"):
    """Two laboratories, invasive E. coli tested for CTX every quarter 2019-2024 (about 30% R),
    plus a rare K. pneumoniae combination that cannot be forecast."""
    rng = np.random.default_rng(1)
    rows, ast = [], []
    quarters = pd.period_range("2019Q1", "2024Q4", freq="Q")
    for q in quarters:
        for lab in ("IT173", "IT174"):
            for k in range(25):
                iid = len(rows)
                day = q.start_time + pd.Timedelta(days=int(rng.integers(0, 85)))
                rows.append((iid, lab, f"{lab}-{q}-{k}", "ESCCOL", day, "1"))
                ast.append((iid, "CTX", "R" if rng.random() < 0.3 else "S"))
        iid = len(rows)
        rows.append((iid, "IT173", f"K-{q}", "KLEPNE", q.start_time, "1"))
        ast.append((iid, "MEM", "R"))
    iso = pd.DataFrame(rows, columns=["isolate_id", "laboratorio", "codice_paziente", "microorganismo",
                                      "data_prelievo", "materiale"])
    iso.loc[len(iso) - 1, "data_prelievo"] = pd.Timestamp(data_end)
    iso["codice_campione"] = iso["isolate_id"].astype(str)
    ast = pd.DataFrame(ast, columns=["isolate_id", "antibiotic", "result"])
    ast["result_harmonized"] = ast["result"]
    ast["intrinsic"] = False
    dataset_store.save_dataset(folder, 1, iso, ast, {}, pd.DataFrame())
    return folder


def test_forecasts_start_after_last_complete_quarter(tmp_path):
    result = compute_forecasts(_store(tmp_path))
    meta = result["meta"]
    assert meta["dati_fino_al"] == "30/12/2024"
    assert meta["ultimo_trimestre"] == "4° trimestre 2024"
    assert meta["trimestri_previsti"] == ["1° trimestre 2025", "2° trimestre 2025",
                                          "3° trimestre 2025", "4° trimestre 2025"]
    combos = {(c["microorganismo"], c["antibiotico"]): c for c in result["combinations"]}

    ecoli = combos[("ESCCOL", "CTX")]
    assert ecoli["ultimi_12_mesi"]["n"] == 200
    assert [f["orizzonte_mesi"] for f in ecoli["previsioni"]] == [3, 6, 9, 12]
    for f in ecoli["previsioni"]:
        assert f["low"] <= f["pct"] <= f["high"]
        assert 15 < f["pct"] < 45
        assert f["semaforo"] in ("verde", "giallo", "grigio")
    assert {lab["codice"] for lab in ecoli["laboratori"]} == {"IT173", "IT174"}

    rare = combos[("KLEPNE", "MEM")]
    assert rare["previsioni"] == []
    assert "Dati insufficienti" in rare["motivo_assenza"]


def test_incomplete_last_quarter_is_not_used_as_origin(tmp_path):
    folder = _store(tmp_path)
    # Add a few isolates in early 2025: the quarter is incomplete, the origin stays 2024Q4
    iso = pd.DataFrame([(10_000, "IT173", "late", "ESCCOL", pd.Timestamp("2025-01-20"), "1")],
                       columns=["isolate_id", "laboratorio", "codice_paziente", "microorganismo",
                                "data_prelievo", "materiale"])
    iso["codice_campione"] = "late"
    ast = pd.DataFrame([(10_000, "CTX", "S", "S", False)],
                       columns=["isolate_id", "antibiotic", "result", "result_harmonized", "intrinsic"])
    dataset_store.save_dataset(folder, 2, iso, ast, {}, pd.DataFrame())
    meta = compute_forecasts(folder)["meta"]
    assert meta["dati_fino_al"] == "20/01/2025"
    assert meta["ultimo_trimestre"] == "4° trimestre 2024"


@pytest.fixture
def client(tmp_path):
    app = create_app({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.sqlite'}",
        "UPLOAD_FOLDER": str(tmp_path / "data"),
        "MODEL_FOLDER": str(tmp_path / "models"),
        "DATASET_FOLDER": str(tmp_path / "dataset"),
        "FORECAST_FOLDER": str(tmp_path / "previsioni"),
    })
    yield app.test_client(), app
    with app.app_context():
        db.session.remove()
        db.engine.dispose()


def test_pages_before_and_after_computing(client, tmp_path):
    http, app = client
    page = http.get("/previsioni")
    assert page.status_code == 200
    assert "non sono ancora state calcolate" in page.get_data(as_text=True)

    _store(app.config["DATASET_FOLDER"])
    response = http.post("/previsioni/calcola")
    assert response.status_code == 302
    assert load_forecasts(app.config["FORECAST_FOLDER"])["meta"]["con_previsione"] == 1

    html = http.get("/previsioni?microrganismo=ESCCOL&orizzonte=6").get_data(as_text=True)
    assert "Cefotaxime" in html and "Dati fino al 30/12/2024" in html and "2° trimestre 2025" in html

    detail = http.get("/previsioni/ESCCOL/CTX").get_data(as_text=True)
    assert "Ultimi 12 mesi" in detail and "indicativo" in detail and "4° trimestre 2025" in detail
    assert http.get("/previsioni/KLEPNE/MEM").status_code == 200
    assert http.get("/previsioni/XXX/YYY").status_code == 404
    assert "sono cambiati" not in html

    # A new upload makes the stored forecasts stale until they are recomputed
    dataset_store.save_dataset(app.config["DATASET_FOLDER"], 2, pd.DataFrame({"isolate_id": [0]}),
                               pd.DataFrame({"isolate_id": [0]}), {}, pd.DataFrame())
    assert "sono cambiati" in http.get("/previsioni").get_data(as_text=True)


def test_legacy_pages_carry_the_not_validated_banner(client):
    http, _ = client
    for url in ("/pipeline", "/clinical"):
        response = http.get(url)
        assert response.status_code == 200
        assert "Strumento precedente, non validato" in response.get_data(as_text=True)
    assert "non validato" not in http.get("/previsioni").get_data(as_text=True).split("<main")[1]


def test_reset_removes_stored_forecasts(client):
    http, app = client
    _store(app.config["DATASET_FOLDER"])
    http.post("/previsioni/calcola")
    assert load_forecasts(app.config["FORECAST_FOLDER"]) is not None
    http.post("/reset-db")
    assert load_forecasts(app.config["FORECAST_FOLDER"]) is None
    assert "non sono ancora state calcolate" in http.get("/previsioni").get_data(as_text=True)
