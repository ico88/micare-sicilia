import io
import numpy as np
import pandas as pd
import pytest

from app import create_app, db
from app.services import jobs
from app.services import dataset_store
from app.services.forecast_run import compute_forecasts, load_forecasts


def _store(folder, data_end="2024-12-30", last_quarter="2024Q4"):
    """Two laboratories, invasive E. coli tested for CTX every quarter 2019-2024 (about 30% R),
    plus a rare K. pneumoniae combination that cannot be forecast."""
    rng = np.random.default_rng(1)
    rows, ast = [], []
    quarters = pd.period_range("2019Q1", last_quarter, freq="Q")
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
    iso["paziente_ricoverato"] = "1"
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
    assert "Non ci sono ancora dati" in page.get_data(as_text=True)

    _store(app.config["DATASET_FOLDER"])
    response = http.post("/previsioni/calcola")
    jobs.wait("previsioni")  # the computation runs on the server, in background
    assert response.status_code == 302
    assert load_forecasts(app.config["FORECAST_FOLDER"])["meta"]["con_previsione"] == 1

    html = http.get("/previsioni").get_data(as_text=True)
    assert "Dati fino al 30/12/2024" in html and "2° trimestre 2025" in html and 'id="listBoxes"' in html
    assert http.get("/previsioni?microrganismo=ESCCOL").headers["Location"].endswith("/previsioni?microorganismo=ESCCOL")

    board = http.get("/previsioni/dati?microorganismo=ESCCOL&orizzonte=6").get_json()
    assert board["selezione"] == {"microorganismo": "ESCCOL", "antibiotico": "CTX", "orizzonte": "6",
                                  "laboratorio": [], "affidabilita": []}
    assert board["meta"]["orizzonte"] == "2° trimestre 2025 (6 mesi)"
    assert [r["nome"] for r in board["tabella"]] == ["Cefotaxime"]
    detail = board["dettaglio"]
    assert detail["ultimi_12_mesi"]["n"] > 0 and len(detail["trimestrali"]) == 4
    assert detail["trimestrali"][-1]["trimestre"] == "4° trimestre 2025"
    assert detail["dodici_mesi"] is not None  # regional, next 12 months
    assert detail["laboratori"] and any(lab["previsione"] for lab in detail["laboratori"])  # per laboratory
    assert board["antibiogramma"]["celle"][0]["antibiotic"] == "CTX"

    lab = detail["laboratori"][0]["laboratorio"]
    one = http.get(f"/previsioni/dati?microorganismo=ESCCOL&orizzonte=6&laboratorio={lab}").get_json()
    assert one["selezione"]["orizzonte"] == "anno"  # per laboratory only the next 12 months
    assert one["dettaglio"]["laboratorio"]["laboratorio"] == lab
    states = {i["valore"]: i["stato"] for i in one["liste"]["orizzonte"]}
    assert states["anno"] == "selezionato" and states["3"] == "escluso"
    reliable = http.get("/previsioni/dati?affidabilita=assente").get_json()
    assert reliable["tabella"] == [] and reliable["antibiogramma"]["celle"] == []

    year = load_forecasts(app.config["FORECAST_FOLDER"])["combinations"]
    assert any(c.get("prossimi_12_mesi") for c in year)
    assert http.get("/previsioni/KLEPNE/MEM").headers["Location"].endswith("microorganismo=KLEPNE&antibiotico=MEM")
    assert http.get("/previsioni/XXX/YYY").status_code == 404
    assert "sono cambiati" not in html

    # A new upload makes the stored forecasts stale until they are recomputed
    dataset_store.save_dataset(app.config["DATASET_FOLDER"], 2, pd.DataFrame({"isolate_id": [0]}),
                               pd.DataFrame({"isolate_id": [0]}), {}, pd.DataFrame())
    assert "sono cambiati" in http.get("/previsioni").get_data(as_text=True)


def test_earlier_pipeline_is_gone(client):
    http, _ = client
    for url in ("/pipeline", "/clinical", "/dashboard", "/explore"):
        assert http.get(url).status_code == 404
    home = http.get("/").get_data(as_text=True)
    assert "Strumenti precedenti" not in home and "non validat" not in home


def test_reset_removes_stored_forecasts(client):
    http, app = client
    _store(app.config["DATASET_FOLDER"])
    http.post("/previsioni/calcola")
    jobs.wait("previsioni")  # the computation runs on the server, in background
    assert load_forecasts(app.config["FORECAST_FOLDER"]) is not None
    http.post("/reset-db")
    assert load_forecasts(app.config["FORECAST_FOLDER"]) is None
    assert "Non ci sono ancora dati" in http.get("/previsioni").get_data(as_text=True)


def _mark_imported(app, folder, upload_id=1, date_to="2024-12-30"):
    """An imported upload with its validation report, as the upload route leaves it."""
    from app.models import UploadedFile

    with app.app_context():
        db.session.add(UploadedFile(id=upload_id, filename="x.xlsx", original_name="rete_mic.xlsx", status="imported"))
        db.session.commit()
    report = {"errors": [], "warnings": [], "info": [],
              "stats": {"rows_read": 1203, "isolates_valid": 1200, "rows_excluded": 3, "ast_results": 1200,
                        "mic_values": 0, "antibiotics": ["CTX"], "date_from": "2019-01-02", "date_to": date_to,
                        "laboratories": 2, "hospitals": 2, "isolates_per_year": {2019: 200}}}
    dataset_store.save_report(folder, upload_id, report, pd.DataFrame())


def test_guide_shows_one_step_to_do_now(client):
    http, app = client
    html = http.get("/").get_data(as_text=True)
    assert "Percorso guidato" in html and "Carica il file Excel esportato" in html
    assert html.count("Fai questo ora") == 1

    folder = app.config["DATASET_FOLDER"]
    _store(folder)
    _mark_imported(app, folder)
    html = http.get("/").get_data(as_text=True)
    assert "Calcola le previsioni (passo 3)" in html and html.count("Fai questo ora") == 1
    assert "1.200 isolati" in html and "30/12/2024" in html

    http.post("/previsioni/calcola")

    jobs.wait("previsioni")  # the computation runs on the server, in background
    html = http.get("/").get_data(as_text=True)
    # Data end long before today: ready, but the operator is told to load newer data
    assert "carica l'export aggiornato" in html
    assert "Apri le previsioni" in html
    assert 'class="steps-bar' in http.get("/previsioni").get_data(as_text=True)


def test_workflow_state_order():
    from datetime import date

    from app.services.workflow import workflow_state

    empty = workflow_state("/nonexistent", "/nonexistent", [], today=date(2025, 1, 1))
    assert empty["current"] == "carica"
    assert empty["status"] == {"carica": "ora", "controlla": "dopo", "calcola": "dopo", "leggi": "dopo"}
    assert empty["data"] is None


def test_validation_page_points_to_next_step(client):
    http, app = client
    folder = app.config["DATASET_FOLDER"]
    _store(folder)
    _mark_imported(app, folder)
    html = http.get("/upload/1/validazione").get_data(as_text=True)
    assert "Prossimo passo: calcola le previsioni" in html
    assert 'action="/previsioni/calcola"' in html



def test_validation_scores_forecasts_made_with_data_up_to_2024(tmp_path):
    from app.services.out_of_sample import compute_validation, validation_possible

    assert not validation_possible(_store(tmp_path / "a"))
    assert not compute_validation(tmp_path / "a")["available"]

    folder = _store(tmp_path / "b", data_end="2025-06-29", last_quarter="2025Q2")
    assert validation_possible(folder)
    result = compute_validation(folder)
    assert result["available"]
    assert result["dati_fino_al"] == "31/12/2024"
    assert result["trimestri_osservati"] == ["1° trimestre 2025", "2° trimestre 2025"]
    rows = result["righe"]
    # Only E. coli / CTX has a forecast; K. pneumoniae has too few isolates
    assert {(r["microorganismo"], r["antibiotico"]) for r in rows} == {("ESCCOL", "CTX")}
    assert {r["periodo"] for r in rows} == {"2025Q1", "2025Q2"}
    for r in rows:
        assert r["orizzonte_mesi"] in (3, 6)
        assert r["errore"] < 15
        assert r["previsto_persistenza_12m"] is not None and r["previsto_prophet"] is not None
    # The forecasts are those made with data up to 2024: the same as computing with that archive
    then = compute_forecasts(tmp_path / "a")
    expected = {f["periodo"]: f["pct"] for c in then["combinations"] if c["microorganismo"] == "ESCCOL"
                for f in c["previsioni"]}
    for r in rows:
        assert abs(r["previsto"] - expected[r["periodo"]]) < 1e-9

    from app.services.diagnostics import _validation_sheet
    sheet = _validation_sheet(result)
    assert {"Previsto (%)", "Osservato (%)", "Scostamento (punti)", "Errore Prophet (modello del progetto originale)", "Dentro l'intervallo"} <= set(sheet.columns)
    assert len(sheet) == len(rows)


def test_validation_page_and_guide_panel(client):
    http, app = client
    html = http.get("/validazione").get_data(as_text=True)
    assert "Servono i dati successivi al 31/12/2024" in html

    folder = app.config["DATASET_FOLDER"]
    _store(folder, data_end="2025-06-29", last_quarter="2025Q2")
    _mark_imported(app, folder, date_to="2025-06-29")
    assert "Verifica sui dati nuovi" in http.get("/").get_data(as_text=True)
    assert http.post("/validazione/calcola").status_code == 302
    jobs.wait("verifica")
    html = http.get("/validazione").get_data(as_text=True)
    assert "Errore tipico delle previsioni mostrate" in html and "Prophet" in html
    assert "Il semaforo ha funzionato?" in html
    assert "Dove sbaglia il modello" in html and "Scarica l'analisi (Excel)" in html
    workbook = http.get("/validazione/analisi.xlsx")
    assert workbook.status_code == 200
    import openpyxl
    sheets = openpyxl.load_workbook(io.BytesIO(workbook.data)).sheetnames
    assert {"Leggimi analisi", "Scostamenti combinazione", "Contributi laboratori", "Modelli"} <= set(sheets)
    assert "hanno sbagliato in media di" in http.get("/").get_data(as_text=True)


def test_results_of_an_older_method_are_out_of_date(client):
    """After an update that changes the method the archive is the same, yet the stored
    forecasts and check must be redone: the pages say so and offer to redo them."""
    import json
    from pathlib import Path

    http, app = client
    folder = app.config["DATASET_FOLDER"]
    _store(folder, data_end="2025-06-29", last_quarter="2025Q2")
    _mark_imported(app, folder, date_to="2025-06-29")
    http.post("/previsioni/calcola")
    jobs.wait("previsioni")
    http.post("/validazione/calcola")
    jobs.wait("verifica")
    assert "metodo di calcolo" not in http.get("/").get_data(as_text=True)
    assert "Rifai la verifica" in http.get("/").get_data(as_text=True)  # always possible

    for name in ("previsioni/previsioni.json", "validazione.json"):
        path = Path(app.config["FORECAST_FOLDER"]) / Path(name).name
        stored = json.loads(path.read_text())
        (stored.get("meta") or stored).pop("metodo")  # as saved before the method version existed
        path.write_text(json.dumps(stored))

    home = http.get("/").get_data(as_text=True)
    assert "Il metodo di calcolo è stato aggiornato: ricalcola le previsioni" in home
    assert "Il metodo di calcolo è stato aggiornato dopo l'ultima verifica" in home
    assert "i numeri qui sotto non valgono più" in http.get("/validazione").get_data(as_text=True)
