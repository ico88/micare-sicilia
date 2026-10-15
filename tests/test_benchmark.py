import numpy as np
import pandas as pd

from app.services.benchmark import choose_and_verify, overview, run_bench, summarize


def _regional(p_by_quarter, n=400, seed=0):
    rng = np.random.default_rng(seed)
    periods = pd.period_range("2019Q1", periods=len(p_by_quarter), freq="Q")
    r = rng.binomial(n, p_by_quarter)
    return pd.DataFrame({
        "laboratorio": "IT1", "microorganismo": "KLEPNE", "antibiotic": "MEM", "period": periods.astype(str),
        "period_start": periods.start_time, "n_harmonized": n, "r_harmonized": r,
    })


def test_stable_series_persistence_matches_noise_and_coverage():
    frames = [_regional([0.4] * 24, seed=s).assign(antibiotic=f"A{s}") for s in range(30)]
    s = summarize(run_bench(pd.concat(frames)))
    pers = s[(s.modello == "persistenza_12m") & (s.orizzonte_mesi == 3)]
    assert pers["errore_su_rumore"].median() < 1.3          # only sampling noise left
    assert 70 <= pers["copertura_80"].mean() <= 92            # honest interval on a stable series
    tarata = s[(s.modello == "persistenza_tarata") & (s.orizzonte_mesi == 3)]
    assert 70 <= tarata["copertura_80"].mean() <= 95


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


def test_trend_model_wins_on_a_steady_trend_and_is_chosen():
    frames = [_regional(np.linspace(0.05, 0.50, 24), seed=s).assign(antibiotic=f"A{s}") for s in range(5)]
    bench = run_bench(pd.concat(frames))
    s = summarize(bench)
    at12 = s[s.orizzonte_mesi == 12].groupby("modello")["errore_medio"].mean()
    assert at12["trend"] < 0.7 * at12["persistenza_12m"]
    verdict = choose_and_verify(bench)
    assert (verdict.loc[verdict.orizzonte_mesi == 12, "mostrato"] != "persistenza_tarata").mean() >= 0.8


def test_laboratory_model_ignores_a_change_in_laboratory_mix():
    # Two labs with stable but different %R; lab B starts sending ten times more isolates
    periods = pd.period_range("2019Q1", periods=24, freq="Q")
    rng = np.random.default_rng(1)
    rows = []
    for i, per in enumerate(periods):
        for lab, p, n in [("A", 0.6, 200), ("B", 0.2, 40 if i < 16 else 400)]:
            rows.append({"laboratorio": lab, "microorganismo": "KLEPNE", "antibiotic": "MEM", "period": str(per),
                         "period_start": per.start_time, "n_harmonized": n, "r_harmonized": rng.binomial(n, p)})
    bench = run_bench(pd.DataFrame(rows))
    after = bench[(bench.origine >= "2023Q2") & (bench.orizzonte_mesi == 3)]
    err = after.assign(e=after.errore.abs()).groupby("modello")["e"].mean()
    assert err["trend_laboratori"] < err["trend"]


def test_traffic_light_never_improves_further_ahead():
    frames = [_regional([0.4] * 24, n=60, seed=s).assign(antibiotic=f"A{s}") for s in range(20)]
    verdict = choose_and_verify(run_bench(pd.concat(frames)))
    order = {"verde": 0, "giallo": 1, "grigio": 2}
    for _, combo in verdict.sort_values("orizzonte_mesi").groupby("antibiotic"):
        levels = combo["semaforo"].map(order).tolist()
        assert levels == sorted(levels)
        assert (combo["semaforo"].map(order) >= combo["semaforo_orizzonte"].map(order)).all()
