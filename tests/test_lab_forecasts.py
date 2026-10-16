import numpy as np
import pandas as pd

from app.services import lab_forecasts as lf


def _series(T=28, seed=1):
    rng = np.random.default_rng(seed)
    periods = [str(p) for p in pd.period_range("2019Q1", periods=T, freq="Q")]
    rows = []
    for antibiotic, base in (("MEM", 0.4), ("CTX", 0.7)):
        for lab, size, shift in (("IT001", 60, 0.0), ("IT002", 30, 0.15), ("IT003", 6, -0.2)):
            for period in periods:
                n = size
                rows.append({"microorganismo": "KLEPNE", "antibiotic": antibiotic, "laboratorio": lab,
                             "period": period, "n_harmonized": n,
                             "r_harmonized": rng.binomial(n, np.clip(base + shift, 0.01, 0.99))})
    return pd.DataFrame(rows)


def test_quantile_dispersion_reaches_80_percent():
    rng = np.random.default_rng(0)
    p, n = 0.3, 100
    obs = np.clip(p + rng.normal(0, 2 * np.sqrt(p * (1 - p) / n), 4000), 0, 1)  # twice the binomial spread
    phi = lf.dispersion(pd.DataFrame({"previsto": p, "osservato": obs, "n": n}))
    assert 3.5 < phi < 4.5
    inside = np.abs(obs - p) <= lf.INTERVAL_Z * np.sqrt(phi * p * (1 - p) / n)
    assert abs(inside.mean() - 0.8) < 0.02


def test_laboratory_forecasts():
    out = lf.forecast_laboratories(_series(), "harmonized", {("KLEPNE", "MEM"), ("KLEPNE", "CTX")})
    block = out[("KLEPNE", "MEM")]
    labs = {f["codice"]: f for f in block["laboratori"]}
    assert set(labs) == {"IT001", "IT002", "IT003"}
    # the small laboratory (24 a year, true 20%) is pulled toward the region and stays grey
    assert labs["IT003"]["semaforo"] == "grigio"
    assert labs["IT003"]["pct"] > 20
    assert labs["IT002"]["pct"] > labs["IT001"]["pct"]  # real differences survive the pooling
    for f in block["laboratori"]:
        assert f["low"] <= f["pct"] <= f["high"]
    assert block["verifica"]["prove"] >= 10


def test_constant_panel_trend_ignores_a_new_laboratory():
    from app.services.forecast_run import _constant_panel_trend, _trend_words

    rows = []
    for period in pd.period_range("2024Q1", "2025Q4", freq="Q"):
        new = period.year == 2025
        for lab, n, r in (("A", 100, 20), ("B", 400 if new else 0, 40 if new else 0)):
            rows.append({"microorganismo": "ESCCOL", "antibiotic": "CTX", "laboratorio": lab, "period": str(period),
                         "n_harmonized": n, "r_harmonized": r})
    labs = pd.DataFrame(rows)
    assert _trend_words(80 + 160, 400 + 1600, 80, 400).startswith("in calo")  # pooled: 20% -> 12%
    assert _constant_panel_trend(labs, "ESCCOL", "CTX", pd.Period("2025Q4", freq="Q")) == "stabile"
