# mic-res-sicilia

Applicazione web Python per importare, pulire, aggregare e interrogare dati della Rete MIC siciliana, con previsioni aggregate S/I/R per sorveglianza epidemiologica e antimicrobial stewardship.

Il codice Colab originale non e stato cancellato:

- `notebooks/prophet-mic-v3.ipynb`
- `legacy/legacy_colab.ipynb`
- `scripts/prophet_mic_v3_colab.py`

## Stack

- Python 3.11+
- Flask, SQLite, SQLAlchemy
- Pandas, NumPy, scikit-learn
- Bootstrap 5 e Chart.js
- joblib per artefatti modello
- pytest per test

## Installazione

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Avvio

```bash
python run.py
```

Poi apri `http://127.0.0.1:5000`.

## Workflow MVP

1. Upload CSV/XLSX dalla pagina `/upload`.
2. Normalizzazione colonne e valori S/I/R.
3. Salvataggio osservazioni e aggregati mensili su SQLite.
4. Training baseline storica, HistGradientBoosting e RandomForest.
5. Dashboard con filtri per patogeno, antibiotico, laboratorio e reparto.
6. Previsione S/I/R per mese futuro con livello di affidabilita.
7. Visualizzazione metriche MAE, RMSE, MAPE, accuracy e F1 macro.

## Formato dataset atteso

Sono supportati due formati principali:

### Formato wide MIC

Colonne minime:

- `DATA_PRELIEVO`
- `MICROORGANISMO` o `patogeno`
- `LABORATORIO`
- colonne antibiotico come `AMK_QUALITATIVO`, `GEN_QUALITATIVO`, ecc.

### Formato long/aggregato

Colonne minime:

- data: `data`, `DATA_PRELIEVO`, `mese`
- patogeno: `patogeno`, `MICROORGANISMO`, `specie`
- antibiotico: `antibiotico`
- laboratorio: `LABORATORIO`, `laboratorio`

Opzionali:

- `reparto`
- `risultato` con valori S/I/R
- conteggi `Conteggio_S`, `Conteggio_I`, `Conteggio_R`
- `Totale_Campioni`

## Database

Le tabelle principali sono:

- `uploaded_files`
- `observations`
- `aggregated_observations`
- `trained_models`
- `predictions`
- `validation_metrics`
- `breakpoints`

Il database locale viene creato in `instance/mic_res_sicilia.sqlite`.

## Limiti clinici

Il sistema stima trend e probabilita aggregate di resistenza. Non suggerisce terapia antibiotica individuale.

Disclaimer: sistema di supporto epidemiologico; non sostituisce antibiogramma del singolo paziente, valutazione microbiologica o giudizio clinico.

## Test

```bash
pytest
```

## Note legacy

Prophet resta disponibile come codice legacy/opzionale, ma l'applicazione web usa come MVP:

- baseline storica;
- HistGradientBoosting;
- RandomForest.
