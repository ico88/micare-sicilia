# micare-sicilia

Applicazione web Python per importare, normalizzare, aggregare e analizzare dati microbiologici RETEMIC/MIC della Sicilia, con modelli previsionali S/I/R per sorveglianza epidemiologica, antimicrobial stewardship e valutazione scientifica dell'affidabilita predittiva.

Il sistema lavora su dati aggregati per mese, patogeno, antibiotico, laboratorio e reparto. Le previsioni sono strumenti di supporto epidemiologico e non sostituiscono antibiogramma del singolo paziente, valutazione microbiologica o giudizio clinico.

## Funzioni principali

- Upload di file Excel/CSV RETEMIC in formato wide o long.
- Accodamento delle nuove osservazioni nel database.
- Deduplica dei file gia importati, utile se lo stesso file viene caricato piu volte.
- Rigenerazione degli aggregati partendo da tutto lo storico disponibile.
- Dashboard filtrabile per patogeno, antibiotico, laboratorio, reparto e intervallo mesi.
- Training asincrono dei modelli con avanzamento, ETA, pausa, ripresa e stop.
- Previsione mensile per la combinazione selezionata.
- Previsione annuale globale: 12 mesi previsionali e sintesi annuale.
- Modalita "Tutti i modelli" per confrontare i risultati previsionali.
- Confronto automatico con dati reali quando il mese o anno previsto e gia presente nello storico.
- Report di validazione con metriche globali e backtest rolling.
- Export CSV di dati aggregati, previsioni e report.

## Stack

- Python 3.10+
- Flask
- SQLite
- SQLAlchemy
- Pandas, NumPy
- scikit-learn
- Prophet opzionale/legacy
- Bootstrap 5
- Chart.js
- pytest

## Installazione

Clona il repository e crea un ambiente virtuale:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Avvia l'applicazione:

```bash
python run.py
```

Poi apri:

```text
http://127.0.0.1:5000
```

Se la porta 5000 e occupata:

```bash
FLASK_APP=run.py flask run --port 5001
```

## Workflow operativo

1. Apri la pagina Upload.
2. Carica uno o piu file RETEMIC/Excel/CSV.
3. Il sistema salva le osservazioni raw nel database.
4. Se il file e gia stato importato, viene saltato.
5. Gli aggregati mensili vengono rigenerati su tutto lo storico.
6. Avvia il training modelli.
7. Attendi il completamento del training prima di usare i risultati aggiornati.
8. Usa la dashboard per filtrare combinazioni patogeno-antibiotico-laboratorio-reparto.
9. Genera previsioni mensili, annuali o su tutti i modelli.
10. Consulta validazione, backtest rolling ed export CSV.

## Training modelli

Il training e asincrono: il browser non resta bloccato durante l'addestramento.

La schermata training mostra:

- stato del job;
- fase corrente;
- progress bar;
- tempo trascorso;
- ETA stimato;
- unita di lavoro completate/totali;
- riepilogo finale;
- eventuale traceback in caso di errore.

Sono disponibili tre comandi:

- **Pausa**: sospende il training al primo checkpoint sicuro.
- **Riprendi**: continua il job sospeso.
- **Stop**: interrompe definitivamente il job; per ripartire serve un nuovo training.

Nota: pausa e stop sono cooperativi. Se scikit-learn sta gia eseguendo un `fit()`, l'applicazione attende la fine del target corrente per evitare artefatti corrotti.

## Modelli previsionali

Il sistema supporta piu strategie:

- `baseline_historical`
- `hist_gradient_boosting`
- `random_forest`
- `rf_quant_hgb_class`
- `ensemble_rf_hgb`
- `auto_hierarchical`
- `prophet`

### RF quantitativo + HGB classe

Strategia ibrida:

- Random Forest per le percentuali quantitative S/I/R.
- HistGradientBoosting per la classe finale suggerita.

### Ensemble RF + HGB

Media pesata:

- 60% Random Forest;
- 40% HistGradientBoosting.

La classe finale resta guidata dal modello decisionale disponibile.

### Modelli gerarchici

Il training gerarchico prova modelli separati su diversi livelli:

- regionale: patogeno + antibiotico;
- laboratorio: patogeno + antibiotico + laboratorio;
- esatto: patogeno + antibiotico + laboratorio + reparto.

Il sistema usa il modello piu specifico disponibile e torna a livelli piu generali quando i dati non sono sufficienti.

## Previsioni

### Previsione mensile

Dato un mese, il sistema prevede:

- percentuale sensibili;
- percentuale intermedi;
- percentuale resistenti;
- classe finale suggerita;
- confidenza;
- livello di affidabilita.

La previsione usa sempre la combinazione selezionata nella dashboard:

- patogeno;
- antibiotico;
- laboratorio, oppure tutti i laboratori;
- reparto, oppure tutti i reparti.

### Previsione annuale

La previsione annuale genera 12 previsioni mensili, da gennaio a dicembre, e poi calcola:

- media annuale S/I/R;
- classe prevalente dell'anno;
- mese con resistenza piu alta;
- tabella mensile;
- confronto con dati reali se l'anno e presente nel database.

### Tutti i modelli

La modalita "Tutti i modelli" genera la stessa previsione con tutti i modelli confrontabili.

Per previsione annuale:

```text
6 modelli x 12 mesi = 72 righe previsionali
```

Questa modalita e utile per confrontare stabilita, divergenze e affidabilita tra modelli.

## Validazione scientifica

La pagina Validazione include:

- riepilogo dataset;
- copertura per patogeno, antibiotico e laboratorio;
- metriche training;
- metriche di regressione: MAE, RMSE, MAPE;
- metriche classificative: accuracy, F1 macro;
- backtest rolling;
- export CSV delle tabelle.

Il backtest rolling confronta i modelli usando mesi finali dello storico come test temporale.

## Formati dati supportati

### Formato wide RETEMIC

Colonne tipiche:

- `DATA_PRELIEVO`
- `MICROORGANISMO`
- `LABORATORIO`
- `REPARTO` o `REPARTO_DI_RICOVERO`
- colonne antibiotico qualitative, ad esempio `AMK_QUALITATIVO`, `CIP_QUALITATIVO`

### Formato long/aggregato

Colonne attese:

- data o mese;
- patogeno;
- antibiotico;
- laboratorio;
- reparto opzionale;
- risultato S/I/R oppure conteggi S/I/R.

## Database locale

Il database SQLite viene creato automaticamente in:

```text
instance/mic_res_sicilia.sqlite
```

Tabelle principali:

- `uploaded_files`
- `observations`
- `aggregated_observations`
- `trained_models`
- `predictions`
- `validation_metrics`
- `breakpoints`

## File non versionati

Il repository non deve includere dati sensibili, database locali o modelli generati.

Sono ignorati da `.gitignore`:

- `data/`
- `instance/`
- `models/`
- `.venv/`
- `.idea/`
- `__pycache__/`
- `.pytest_cache/`
- file `.xlsx`, `.xls`, `.sqlite`, `.db`, `.joblib`, `.csv`

Per mantenere Git pulito:

```bash
git status --short --ignored
```

## Test

Esegui:

```bash
pytest
```

Oppure:

```bash
python -m pytest
```

## Materiale legacy

Il codice Colab originale e mantenuto come riferimento:

- `notebooks/prophet-mic-v3.ipynb`
- `legacy/legacy_colab.ipynb`
- `scripts/prophet_mic_v3_colab.py`

## Disclaimer clinico

Il sistema produce previsioni aggregate a fini epidemiologici e di supporto alla stewardship antimicrobica. Non fornisce indicazioni terapeutiche individuali e non sostituisce referto microbiologico, antibiogramma, linee guida cliniche o giudizio medico.
