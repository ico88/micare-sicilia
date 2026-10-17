# MIC RES Sicilia

**Piattaforma di previsione della resistenza antimicrobica per la Rete MIC Siciliana**  
*AMR forecasting platform for the Sicilian Minimum Inhibitory Concentration Network*

Developed at the **University of Catania (UniCT)** in collaboration with Sicilian clinical microbiology laboratories.

---

## What it does

MIC RES Sicilia forecasts antimicrobial resistance for the Sicilian Rete MIC.

**Guided path (home page).** Four steps in a fixed order: upload the Rete MIC export, read the data check, compute the forecasts, read them. The step to do now is highlighted; its state is derived from the validation reports and the stored forecasts (`app/services/workflow.py`).

**Consultation dashboard (page "Consultazione dati").** The archive sliced like the regional Qlik dashboard, with the rules of the forecasts: first isolate per patient and quarter, intrinsic resistances left out, harmonised EUCAST reading (or as reported), a percentage reportable only with at least 30 isolates tested. Filters are Qlik-style list boxes: click values to select them (green), the values the other selections leave possible stay white and the excluded ones turn grey, each with its count of isolates; search inside a box (Enter selects what is found), several values per field (year, quarter, laboratory, patient type, intensive care, sex, age band), one at a time for population, pathogen and antibiotic. A selection bar on top lists the selections, with back, forward and clear. Four views follow the selection: key figures, %R per quarter with its 95% interval and the isolates tested, laboratories compared with the figure of the whole selection, and the cumulative antibiogram pathogen × antibiotic; a click on a laboratory or a cell selects it. The **Previsioni** switch shows the stored forecasts on the same views: the regional quarters and next 12 months after the observed series, the next 12 months per laboratory, the antibiogram expected over the next 12 months, with their 80% interval and reliability (unreliable forecasts are not shown, as on the forecast pages). Forecasts exist for invasive isolates, harmonised reading, the region or one laboratory, without patient type, ward, sex or age filters; otherwise the page says why and offers to drop those filters. The page reads a cube of counts built once per archive version (`app/services/explore.py`, `instance/consultazione/`): no patient-level record reaches it. **Scarica la vista (Excel)** exports the tables of the current selection, with the forecast when shown.

**Validated forecasts (page "Previsioni di resistenza").** Regional invasive isolates (blood and CSF) of hospitalised, emergency, day-hospital and long-term-care patients; invasive isolates of non-hospitalised patients (`PAZIENTE_RICOVERATO` = 5, "altro") are a population of their own, outside the forecasts (in 2025 one laboratory started sending them by the thousand, which looked like a fall of resistance; without them the 2025 out-of-sample error fell from 4.1 to 3.2 points and the bias from −2.1 to −0.3), the next four quarters after the last complete quarter of data. For each pathogen × antibiotic one forecaster is chosen on the 2020-22 rolling-origin tests and judged on 2023-24 only (`app/services/benchmark.py`). The operator sees the value in words, an 80% interval, the last 12 months, the typical past error and a traffic light (affidabile / indicativa / non affidabile) that never improves with the horizon. Forecasts are computed with "Calcola previsioni" and stored in `instance/previsioni/previsioni.json` (`app/services/forecast_run.py`). The page is a dashboard like the consultation page (same list boxes, selection bar, back / forward, `app/static/js/dashboard.js`, `app/services/forecast_board.py`): pick pathogen, antibiotic, horizon (a forecast quarter or the next 12 months), one laboratory and the reliability classes to show; it shows the key figures, the regional series with the forecasts, why the forecast is what it is (model, traffic light, past tests), every antibiotic of the pathogen at the horizon, the laboratories (forecast and observed) and the antibiogram expected at the horizon. A click on a row, a card, a laboratory or a cell selects it.

**Out-of-sample check (page "Verifica sui dati nuovi").** Once the archive holds complete quarters after 2024, the forecasts are recomputed with data cut at 31/12/2024 and scored against what was observed, next to "as the last 12 months" and Prophet (`app/services/out_of_sample.py`). Prophet, the model of the original project, is a candidate in the test bench like the others.

**Where the model goes wrong (same page).** For every combination the observed quarters are pooled and set against the last year before the cutoff; the deviation (observed − forecast) is split exactly into *laboratory mix* (laboratories joining the network or weighing more or less) and *unforeseen change* of resistance inside the laboratories, or called noise when within 1.96 standard errors (`app/services/error_analysis.py`). The page lists the largest deviations, the laboratories that move the regional figure most and how every candidate model would have done; **Scarica l'analisi (Excel)** exports it all, aggregated figures only, to review and improve the models.

**Per-laboratory forecasts (forecast page, laboratory panel).** The next 12 months for every laboratory: its last 12 months pulled toward the regional figure, more strongly the fewer isolates it has (partial pooling, empirical Bayes, `app/services/lab_forecasts.py`). The 80% interval is calibrated on older tests and judged on the last two years. On 2025, with data cut at 2024: 5.3 points of error against 6.1 for the laboratory's own last year and 5.6 for the regional figure (385 comparisons). Beside the regional trend, the same trend at constant laboratories (those present in both years, earlier weights), as ECDC does.

**Regional forecast of the next 12 months (forecast page, horizon "Prossimi 12 mesi", the default).** The yearly %R to expect, beside the quarterly forecasts: over a year the error is about a quarter lower (2.9 against 3.7 points in 2023-2024), interval and traffic light calibrated as for laboratories.

**Model search (page "Ricerca dei modelli", `python scripts/ricerca_modelli.py`).** 86 configurations (recent levels over 6-24 months, pooling toward the region at several strengths, trends with a level per laboratory, blends, gradient boosting across all combinations) on four targets: regional %R per quarter and over 12 months, regional %R at constant panel, laboratory %R over 12 months; each model also with the last quarter's weights, rerun on laboratory × intensive care and laboratory × patient type strata, and a trend shared by antibiotics of the same mechanism group. Ranked on targets up to 2022, confirmed on 2023-2024; the forecasts made at the end of 2024 for 2025 are shown beside, never used to choose (`app/services/model_search.py`). Result: nothing beats "as the last 12 months" for the regional figure by the required 5% (the class trend did better on 2025 but worse on 2023-2024, so it is not used); pooling toward the region wins for laboratories; forecasting 12 months instead of a quarter cuts the error by about a quarter.

**Background jobs.** Downloading from the dashboard, computing the forecasts, the out-of-sample check, the model search and preparing the consultation data run on the server (`app/services/jobs.py`, one at a time): the browser can be closed, every page shows the progress and the outcome.

---

## Tech stack

| Layer | Library |
|---|---|
| Web framework | Flask 3 + Gunicorn |
| Database | SQLite via SQLAlchemy (list of uploaded files) |
| Archive | Parquet (pyarrow), one partition per upload |
| Forecasting | own binomial models (numpy), gradient boosting (scikit-learn), Prophet as a candidate |
| Frontend | Bootstrap 5, Chart.js 4 |

---

## Quick start (local)

```bash
git clone https://github.com/ico88/micare-sicilia.git
cd micare-sicilia

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

flask --app run:app run --debug
```

Open http://localhost:5000 → step 1 "Scarica dal cruscotto" (or upload the Rete MIC export) → step 3 "Calcola le previsioni".

---

## Server deployment

Run the installer (Ubuntu 20+, Debian 11+, RHEL 8+, macOS):

```bash
bash installer.sh
```

The script:
1. Installs system dependencies (`python3`, `build-essential`, Stan toolchain)
2. Creates a Python virtual environment and installs all packages
3. Starts the app on **port 5555** via Gunicorn (single worker, 8 threads)
4. Registers a **systemd service** (`mic-res-sicilia`) that starts automatically on boot

```bash
sudo systemctl status mic-res-sicilia
# open http://<server-ip>:5555
```

---

## Data format

### Rete MIC isolate-level export

**Source.** The data come from the public Rete MIC dashboard of the Sicilian regional
health service (Qualità Sicilia SSR, Qlik Sense, anonymous access):
<https://qlik.qualitasiciliassr.it/anonimo/single/?appid=85ada16c-4b41-4bc6-9ca1-405b8243d0c2&sheet=6ad6f3ac-3369-41c5-bd72-792243f9091b&opt=ctxmenu,currsel>.
The records can be downloaded straight from it, without going through Excel:

- **in the app**: step 1 → **Scarica dal cruscotto** (runs in the background, about 10 minutes,
  progress shown on the page);
- **from the command line**: `python scripts/estrai_qlik_retemic.py [file.csv]`, then upload the file.

`app/services/qlik_source.py` reads the dashboard table `Dati_Grezzi` (one row per isolate, same
columns as the export) through the Qlik Engine API with the anonymous session a browser gets:
no browser and no login. It drops the fields the dashboard adds on load (`KEY_CAMPIONE`,
`ETA_PZ`, `CHK_*`, …) and turns the `"<code> - <label>"` cells (`MATERIALE`, `OSPEDALE`,
`PAZIENTE_RICOVERATO`, `REPARTO_DI_RICOVERO`) back into codes. The dashboard holds the whole
history, so each refresh **replaces** the previous dashboard downloads (and, if ticked, the files
uploaded by hand from the same dashboard) instead of adding to them. The CSV holds
patient-level records and stays in `data/`, which is not committed.

The network export (one row per isolate, `<ABX>_QUALITATIVO` / `<ABX>_QUANTITATIVO`
column pairs, e.g. `DATA_PRELIEVO`, `CODICE_PAZIENTE`, `REPARTO_DI_RICOVERO`, `MICROORGANISMO`)
is uploaded once (a 40 MB workbook imports in about a minute) and:

1. **validated**: required columns, dates (no day/month swaps), age, sex, S/I/R values,
   MIC syntax, duplicate rows/samples, repeat isolates, intrinsic-resistance tests.
   The report is shown after upload and row-level anomalies can be downloaded as CSV;
2. **archived in full** as Parquet under `instance/dataset/` (codes keep leading zeros,
   MIC split into operator + value);
3. read by the forecasts as quarterly series per laboratory (`app/services/series.py`).

Each result is kept in two readings: as reported by the laboratory (breakpoints in force
that year) and harmonised (MIC re-read with EUCAST v16.1, `app/resources/breakpoints/`).
To update the breakpoints, download the official EUCAST workbook and run
`python scripts/import_eucast_breakpoints.py <file.xlsx> <version>`; review the chosen rows
in the generated CSV.

`app/services/series.py` builds the quarterly/monthly series per laboratory × pathogen ×
antibiotic (first isolate, wards pooled, intrinsic resistance excluded) and the annual
cumulative antibiogram. **Gestione dati → Scarica file diagnostico** exports them, with the
breakpoint effects and import anomalies, as one Excel file of aggregated figures only.

Any model can reuse the full records:

```python
from app.services.dataset_store import load_isolates, load_ast_results

isolates = load_isolates("instance/dataset", first_isolate_only=True)   # one row per isolate
ast = load_ast_results("instance/dataset", first_isolate_only=True)     # isolate x antibiotic, with patient/ward fields
```

---

## Project structure

```
app/
  routes/               guide (home), upload (data), explore (consultation), forecasts (forecasts, check, model search)
  services/
    qlik_source.py      download from the regional dashboard
    dataset_store.py    validation and Parquet archive of the isolates
    series.py           quarterly series per laboratory, populations, strata
    explore.py          aggregated cube and queries of the consultation dashboard
    forecast_board.py   the stored forecasts sliced for the forecast dashboard
    benchmark.py        rolling-origin test bench, model choice, traffic light
    forecasting.py      candidate forecasters
    forecast_run.py     forecasts shown to the operator
    lab_forecasts.py    per-laboratory and regional 12-month forecasts
    out_of_sample.py    check on the data after the cutoff
    error_analysis.py   where the forecasts went wrong (mix vs real change)
    model_search.py     search over model configurations
    diagnostics.py      aggregated workbooks
    jobs.py             background jobs
  templates/            Jinja2 pages
src/micare_sicilia/     reference tables (network start, antibiotic classes)
scripts/                command-line download, model search, EUCAST import
instance/               SQLite list of uploads, archive, stored results (not committed)
data/                   downloaded and uploaded files (not committed)
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*Università degli Studi di Catania · Dipartimento di Farmacia · Rete MIC Sicilia*
