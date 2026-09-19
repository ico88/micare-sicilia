# MIC RES Sicilia

**Piattaforma di previsione della resistenza antimicrobica per la Rete MIC Siciliana**  
*AMR forecasting platform for the Sicilian Minimum Inhibitory Concentration Network*

Developed at the **University of Catania (UniCT)** in collaboration with Sicilian clinical microbiology laboratories.

---

## What it does

MIC RES Sicilia ingests monthly antibiogram data (pathogen × antibiotic × laboratory × ward), trains **Meta Prophet** time-series models for each combination, and forecasts the percentage of Susceptible / Intermediate / Resistant isolates up to **5 years ahead** — with 80% confidence intervals.

The system supports:

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
