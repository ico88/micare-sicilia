import numpy as np
import pandas as pd

from app.services.benchmark import overview, run_bench, summarize


def _regional(p_by_quarter, n=400, seed=0):
    rng = np.random.default_rng(seed)
    periods = pd.period_range("2019Q1", periods=len(p_by_quarter), freq="Q")
    r = rng.binomial(n, p_by_quarter)
    return pd.DataFrame({
        "microorganismo": "KLEPNE", "antibiotic": "MEM", "period": periods.astype(str),
        "period_start": periods.start_time, "n_harmonized": n, "r_harmonized": r,
    })


def test_stable_series_persistence_matches_noise_and_coverage():
    frames = [_regional([0.4] * 24, seed=s).assign(antibiotic=f"A{s}") for s in range(30)]
    s = summarize(run_bench(pd.concat(frames)))
    pers = s[(s.modello == "persistenza_12m") & (s.orizzonte_mesi == 3)]
    assert pers["errore_su_rumore"].median() < 1.3          # only sampling noise left
    assert 70 <= pers["copertura_80"].mean() <= 92            # honest interval on a stable series


def test_rising_series_persistence_underestimates():
    s = summarize(run_bench(_regional(np.linspace(0.05, 0.45, 24))))
    pers = s[(s.modello == "persistenza_12m")].set_index("orizzonte_mesi")
    assert pers.loc[12, "distorsione"] < -5                   # lags behind the trend
    assert pers.loc[12, "errore_medio"] > pers.loc[3, "errore_medio"]


def test_origins_use_only_past_and_skip_small_quarters():
    data = _regional([0.3] * 12)
    data.loc[10, "n_harmonized"] = 10                         # not reportable: not a test target
    bench = run_bench(data)
    assert bench["origine"].min() == "2020Q4"                 # eight quarters of history first
    assert "2021Q3" not in set(bench["trimestre"])
    assert (bench["trimestre"] > bench["origine"]).all()
    assert set(overview(summarize(bench))["orizzonte_mesi"]) == {3, 6, 9, 12}
