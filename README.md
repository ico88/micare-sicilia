# MIC RES Sicilia

**Piattaforma di previsione della resistenza antimicrobica per la Rete MIC Siciliana**  
*AMR forecasting platform for the Sicilian Minimum Inhibitory Concentration Network*

Developed at the **University of Catania (UniCT)** in collaboration with Sicilian clinical microbiology laboratories.

---

## What it does

MIC RES Sicilia forecasts antimicrobial resistance for the Sicilian Rete MIC.

**Guided path (home page).** Four steps in a fixed order: upload the Rete MIC export, read the data check, compute the forecasts, read them. The step to do now is highlighted; its state is derived from the validation reports and the stored forecasts (`app/services/workflow.py`).

**Validated forecasts (page "Previsioni di resistenza").** Regional invasive isolates (blood and CSF), the next four quarters after the last complete quarter of data. For each pathogen × antibiotic one forecaster is chosen on the 2020-22 rolling-origin tests and judged on 2023-24 only (`app/services/benchmark.py`). The operator sees the value in words, an 80% interval, the last 12 months, the typical past error and a traffic light (affidabile / indicativa / non affidabile) that never improves with the horizon. Forecasts are computed with "Calcola previsioni" and stored in `instance/previsioni/previsioni.json` (`app/services/forecast_run.py`); laboratory figures are shown as observed, indicative values.

**Earlier tools (not validated).** The original pipeline trains **Meta Prophet** models on monthly data per combination and forecasts S / I / R up to 5 years ahead. It is kept under "Strumenti precedenti (non validati)" and supports:

| Feature | Details |
|---|---|
| **Data ingestion** | Upload Excel/CSV files with raw antibiogram records; automated normalisation and aggregation |
| **Expanding-window cross-validation** | Walk-forward validation to estimate real-world forecast error before final training |
| **ARIMA baseline** | Parallel SARIMAX(1,1,1) baseline for each combination; Prophet is compared against it |
| **Pre-computed forecasts** | All combinations × 60 future months stored in SQLite for instant clinical queries |
| **Confidence intervals** | Prophet 80% CI propagated through normalization to S/I/R percentages |
| **Clinical query UI** | Doctor-facing form: select pathogen + antibiotic + laboratory + month → get forecast with CI |
| **Async pipeline** | Training runs in a background thread; closing the browser does not interrupt it; state persists across server restarts |
| **Dashboard** | Historical trends, annual summaries, model-comparison tables |

---

## Tech stack

| Layer | Library |
|---|---|
| Web framework | Flask 3 + Gunicorn |
| Database | SQLite via SQLAlchemy |
| Time-series forecasting | [Meta Prophet](https://facebook.github.io/prophet/) |
| Baseline model | statsmodels SARIMAX |
| Parallelism | joblib `Parallel` |
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

Open http://localhost:5000 → upload your CSV data → run the pipeline.

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

## Pipeline steps

```
Step 1  Load data          ← upload CSV/Excel via web UI
Step 2  Expanding CV       ← train on first 3 years, validate on remaining
Step 3  Refined CV         ← train on all years minus last, validate on last year
Step 4  Final training     ← train on all data, pre-compute 60-month forecasts
```

All steps run asynchronously. If the browser is closed, the process continues in the background. On page reload the live status is restored from the database.

---

## Data format

Input files should be Excel or CSV with columns (names are flexible; the system maps them):

| Column | Example |
|---|---|
| Date / Month | `2023-04` or `01/04/2023` |
| Pathogen | `Escherichia coli` |
| Antibiotic / Material code | `AMC` / `Amoxicillina-clavulanato` |
| Laboratory | `Laboratorio Catania` |
| Ward (optional) | `Terapia Intensiva` |
| Result | `S` / `I` / `R` |
| Sample count | `42` |

### Rete MIC isolate-level export

The network export (one row per isolate, `<ABX>_QUALITATIVO` / `<ABX>_QUANTITATIVO`
column pairs, e.g. `DATA_PRELIEVO`, `CODICE_PAZIENTE`, `REPARTO_DI_RICOVERO`, `MICROORGANISMO`)
is uploaded once (a 40 MB workbook imports in about a minute) and:

1. **validated**: required columns, dates (no day/month swaps), age, sex, S/I/R values,
   MIC syntax, duplicate rows/samples, repeat isolates, intrinsic-resistance tests.
   The report is shown after upload and row-level anomalies can be downloaded as CSV;
2. **archived in full** as Parquet under `instance/dataset/` (codes keep leading zeros,
   MIC split into operator + value);
3. aggregated into the monthly S/I/R table used by the forecasting pipeline.

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
app/                    Flask application
  routes/               Blueprints: dashboard, upload, pipeline
  services/             pipeline_runner, prediction, prophet_training_adapter
  templates/            Jinja2 HTML templates
  static/css/           Custom styles
src/micare_sicilia/     Core ML library (model.py, config.py, preprocessing.py, …)
instance/               SQLite database (auto-created)
models/                 Trained Prophet artifacts (auto-created)
data/                   Uploaded data files (auto-created)
installer.sh            One-shot server installer
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*Università degli Studi di Catania · Dipartimento di Farmacia · Rete MIC Sicilia*
