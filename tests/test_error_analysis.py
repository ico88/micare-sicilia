import threading

import pandas as pd
import pytest

from app import create_app, db
from app.services import jobs
from app.services.error_analysis import analyse

BASE = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]
NOW = ["2025Q1", "2025Q2"]
MODELS = ("persistenza_12m", "trend")


def _labs(antibiotic, per_lab):
    """per_lab: {lab: ((n, r) per base quarter, (n, r) per observed quarter)}; None = absent."""
    rows = []
    for lab, (base, now) in per_lab.items():
        for periods, values in ((BASE, base), (NOW, now)):
            if values is None:
                continue
            for period in periods:
                rows.append({"microorganismo": "KLEPNE", "antibiotic": antibiotic, "laboratorio": lab,
                             "period": period, "n_harmonized": values[0], "r_harmonized": values[1],
                             "isolates": values[0]})
    return pd.DataFrame(rows)


def _rows(antibiotic, series, forecast_pct, trend_pct=None):
    now = series[series["period"].isin(NOW)].groupby("period")[["n_harmonized", "r_harmonized"]].sum()
    rows = []
    for h, (period, q) in enumerate(now.iterrows(), start=1):
        observed = 100 * q["r_harmonized"] / q["n_harmonized"]
        trend = forecast_pct if trend_pct is None else trend_pct
        rows.append({
            "microorganismo": "KLEPNE", "microorganismo_nome": "K. pneumoniae", "antibiotico": antibiotic,
            "antibiotico_nome": antibiotic, "trimestre": period, "periodo": period, "orizzonte_mesi": 3 * h,
            "semaforo": "verde", "modello": "persistenza_tarata", "n": int(q["n_harmonized"]),
            "previsto": forecast_pct, "basso": forecast_pct - 5, "alto": forecast_pct + 5, "osservato": observed,
            "errore": abs(observed - forecast_pct), "scostamento": observed - forecast_pct,
            "dentro": abs(observed - forecast_pct) <= 5,
            "previsto_persistenza_12m": forecast_pct, "errore_persistenza_12m": abs(observed - forecast_pct),
            "previsto_trend": trend, "errore_trend": abs(observed - trend),
        })
    return rows


@pytest.fixture
def analysis():
    mix = _labs("MEM", {"A": ((100, 20), (100, 20)), "B": (None, (100, 60))})      # new laboratory at 60%
    real = _labs("CTX", {"A": ((400, 80), (400, 140))})                              # 20% -> 35% in place
    noise = _labs("CIP", {"A": ((100, 20), (100, 21))})                              # 20% -> 21%
    series = pd.concat([mix, real, noise])
    rows = _rows("MEM", mix, 20.0) + _rows("CTX", real, 20.0, trend_pct=30.0) + _rows("CIP", noise, 20.0)
    return analyse(rows, series, pd.Period("2024Q4", freq="Q"), MODELS)


def test_deviation_is_split_into_laboratory_mix_and_unforeseen_change(analysis):
    combos = {c["antibiotico"]: c for c in analysis["combinazioni"]}
    assert {k: c["causa"] for k, c in combos.items()} == {"MEM": "mix", "CTX": "reale", "CIP": "rumore"}

    mem = combos["MEM"]
    assert mem["base"] == pytest.approx(20) and mem["osservato"] == pytest.approx(40)
    assert mem["scostamento"] == pytest.approx(20) and mem["di_cui_mix"] == pytest.approx(20)
    assert mem["laboratori_nuovi"] == ["B"] and mem["sottostime"] == 2 and mem["stesso_verso"]
    ctx = combos["CTX"]
    assert ctx["di_cui_mix"] == pytest.approx(0) and ctx["di_cui_non_previsto"] == pytest.approx(15)
    assert ctx["migliore_col_senno_di_poi"] == "trend"  # 30% was closer to 35% than 20%

    labs = pd.DataFrame(analysis["laboratori"])
    for antibiotic, c in combos.items():
        # exact: the laboratories add up to the regional change, the parts to the deviation
        assert labs[labs["antibiotico"] == antibiotic]["contributo"].sum() == pytest.approx(c["variazione_osservata"])
        assert c["di_cui_mix"] + c["di_cui_non_previsto"] == pytest.approx(c["scostamento"])
    new = labs[(labs["antibiotico"] == "MEM") & (labs["laboratorio"] == "B")].iloc[0]
    assert new["stato"] == "nuovo" and new["contributo_mix"] == pytest.approx(20)


def test_summary_and_models(analysis):
    s = analysis["sintesi"]
    assert s["cause"] == {"rumore": 1, "mix": 1, "reale": 1}
    assert s["sottostima"] == 3 and s["distorsione_trimestri"] > 0
    assert analysis["combinazioni"][0]["antibiotico"] == "MEM"  # largest deviation first
    models = {m["modello"]: m for m in analysis["modelli"]["complessivo"]}
    assert set(models) == {"mostrato", *MODELS}
    assert models["trend"]["migliore_in"] == 1
    lab_a = next(l for l in analysis["laboratori_sintesi"] if l["laboratorio"] == "A")
    assert lab_a["combinazioni"] == 3


@pytest.fixture
def app(tmp_path):
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 't.sqlite'}",
                      "UPLOAD_FOLDER": str(tmp_path / "data"), "MODEL_FOLDER": str(tmp_path / "models"),
                      "DATASET_FOLDER": str(tmp_path / "dataset"), "FORECAST_FOLDER": str(tmp_path / "prev")})
    yield app
    with app.app_context():
        db.session.remove()
        db.engine.dispose()


def test_one_job_at_a_time_shown_on_every_page(app):
    release = threading.Event()

    def slow(progress):
        progress("A metà strada", 50)
        release.wait(10)
        return {"message": "Fatto.", "link_endpoint": "forecasts.overview"}

    assert jobs.start("previsioni", app, slow) is None
    busy = jobs.start("verifica", app, lambda progress: {})
    assert busy["kind"] == "previsioni"  # refused: the running job is returned
    http = app.test_client()
    for url in ("/", "/previsioni", "/upload"):
        assert "Calcolo delle previsioni in corso" in http.get(url).get_data(as_text=True)
    assert http.get("/lavori/stato").get_json()["progress"] == 50

    release.set()
    jobs.wait("previsioni", timeout=10)
    html = http.get("/").get_data(as_text=True)
    assert "Calcolo delle previsioni completato" in html and 'href="/previsioni"' in html
    assert http.get("/lavori/stato").get_json()["status"] == "success"


def test_failed_job_is_reported(app):
    def broken(progress):
        raise RuntimeError("Nessun dato in archivio.")

    assert jobs.start("previsioni", app, broken) is None
    jobs.wait("previsioni", timeout=10)
    assert jobs.get("previsioni")["status"] == "error"
    assert "non riuscito" in app.test_client().get("/").get_data(as_text=True)
