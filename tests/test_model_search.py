import numpy as np
import pandas as pd
import pytest

from app.services import model_search as ms


def _combo(T=28, seed=0, labs=(("A", 120, 0.2), ("B", 40, 0.5), ("C", 8, 0.3))):
    rng = np.random.default_rng(seed)
    periods = [str(p) for p in pd.period_range("2019Q1", periods=T, freq="Q")]
    n = np.array([[size] * T for _, size, _ in labs], dtype=float)
    r = np.array([rng.binomial(size, p, T) for _, size, p in labs], dtype=float)
    return ms.Combination("KLEPNE", "MEM", periods, [code for code, _, _ in labs], n, r)


def test_no_candidate_sees_the_future():
    c = _combo()
    origin = 15
    changed = c.r.copy()
    changed[:, origin + 1:] = c.n[:, origin + 1:]  # everything resistant after the origin
    for name, model in ms.candidates().items():
        before, after = model(c.n, c.r, origin), model(c.n, changed, origin)
        assert np.allclose(before, after, equal_nan=True), name


def test_partial_pooling_pulls_small_laboratories_most():
    c = _combo(labs=(("A", 400, 0.2), ("B", 400, 0.25), ("C", 4, 0.9)))
    own = ms.lab_persistence(4)(c.n, c.r, 20)[:, 0]
    pooled = ms.lab_shrunk(4)(c.n, c.r, 20)[:, 0]
    regional = c.r[:, 17:21].sum() / c.n[:, 17:21].sum()
    # the tiny laboratory moves toward the region much more than the large ones
    assert abs(pooled[2] - regional) < abs(own[2] - regional)
    assert abs(pooled[0] - own[0]) < abs(pooled[2] - own[2])


def test_periods_never_mix():
    hold = pd.Period("2024Q4", freq="Q")
    q = lambda s: pd.Period(s, freq="Q")
    assert ms.period_of(q("2022Q4"), q("2022Q1"), hold) == "scelta"
    assert ms.period_of(q("2023Q2"), q("2022Q4"), hold) == "verifica"
    assert ms.period_of(q("2025Q2"), q("2024Q4"), hold) == "conferma"
    assert ms.period_of(q("2025Q1"), q("2024Q3"), hold) is None  # 2025 seen only from the end of 2024


def test_tests_and_leaderboard():
    combos = [_combo(seed=s) for s in range(3)]
    models = {ms.REFERENCE: ms.candidates()[ms.REFERENCE],
              "laboratorio · ultimi 12 mesi": ms.lab_persistence(4),
              "laboratorio · ultimi 12 mesi, verso la regione ×1": ms.lab_shrunk(4)}
    tests = ms.run_tests(combos, models)
    assert set(tests["obiettivo"]) == {"regionale", "pannello", "laboratorio"}
    assert set(tests["periodo"]) == {"scelta", "verifica", "conferma"}
    assert tests[tests["periodo"] == "conferma"]["origine"].eq("2024Q4").all()
    # laboratory C (8 a quarter, 32 a year) is a laboratory target; with 4 quarters pooled it reaches 30
    assert "C" in set(tests[tests["obiettivo"] == "laboratorio"]["laboratorio"])
    board = ms.leaderboard(tests)
    ref = board[(board["modello"] == ms.REFERENCE)]
    assert np.allclose(ref["errore_scelta"], ref["rif_scelta"])
    lab = board[board["obiettivo"] == "laboratorio"].set_index("modello")
    # laboratories truly differ (20%, 50%, 30%): their own figure beats the regional one
    assert lab.loc["laboratorio · ultimi 12 mesi", "errore_verifica"] < lab.loc[ms.REFERENCE, "errore_verifica"]
    chosen = ms.choose(board)
    assert set(chosen["obiettivo"]) == {"regionale", "pannello", "laboratorio"}


def test_gradient_boosting_learns_only_from_the_past():
    combos = [_combo(seed=s) for s in range(4)]
    table = ms.gbm_predictions(combos)
    assert table and all(v.shape == (3, 4) for v in table.values())
    # change everything after 2022Q4: forecasts made up to then must not move
    cut = combos[0].periods.index("2022Q4")
    changed = [ms.Combination(c.pathogen, c.antibiotic, c.periods, c.labs, c.n,
                              np.where(np.arange(c.n.shape[1]) > cut, c.n, c.r)) for c in combos]
    again = ms.gbm_predictions(changed)
    for key, value in table.items():
        if key[1] <= cut:
            assert np.allclose(value, again[key], equal_nan=True), key
    tests = ms.run_tests(combos, {ms.REFERENCE: ms.candidates()[ms.REFERENCE]}, precomputed={ms.GBM_NAME: table})
    assert ms.GBM_NAME in set(tests["modello"])


def test_search_page_and_job(tmp_path, monkeypatch):
    from app import create_app, db
    from app.services import jobs

    combos = [_combo(seed=s) for s in range(3)]
    models = {ms.REFERENCE: ms.candidates()[ms.REFERENCE], "laboratorio · ultimi 12 mesi": ms.lab_persistence(4)}

    def fake_search(folder, progress=None):
        tests = ms.run_tests(combos, models)
        board = ms.leaderboard(tests)
        return {"calcolato_il": "oggi", "popolazione": "invasivi", "lettura": "harmonized", "taglio": "2024Q4",
                "combinazioni": 3, "configurazioni": 2, "prove": tests, "classifica": board, "scelti": ms.choose(board)}

    monkeypatch.setattr(ms, "run_search", fake_search)
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 't.sqlite'}",
                      "UPLOAD_FOLDER": str(tmp_path / "d"), "MODEL_FOLDER": str(tmp_path / "m"),
                      "DATASET_FOLDER": str(tmp_path / "ds"), "FORECAST_FOLDER": str(tmp_path / "p")})
    http = app.test_client()
    assert "Nessuna ricerca ancora fatta" in http.get("/modelli").get_data(as_text=True)
    assert http.post("/modelli/ricerca").status_code == 302
    jobs.wait("ricerca", timeout=60)
    assert jobs.get("ricerca")["status"] == "success", jobs.get("ricerca")["error"]
    page = http.get("/modelli").get_data(as_text=True)
    assert "Cosa è stato scelto" in page and "laboratorio · ultimi 12 mesi" in page
    assert http.get("/modelli/ricerca.xlsx").status_code == 200
    with app.app_context():
        db.session.remove()
        db.engine.dispose()
