"""Downloadable diagnostic workbook.

One Excel file with aggregated figures only (no patient or sample codes), so
it can be shared to review data quality and, from phase 2, model reliability.
Each sheet starts with a plain-language line saying what it shows.
"""
from __future__ import annotations

import io
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .breakpoints import BREAKPOINTS_VERSION, breakpoint_table
from .dataset_store import first_isolate_mask, load_ast_results, load_isolates
from .reference import ADMISSION_LABELS, MATERIAL_LABELS, label, laboratory_names
from .benchmark import choose_and_verify, overview, run_bench, summarize
from .series import MIN_ISOLATES_PER_PERIOD, build_series, cumulative_antibiogram, regional_series

DIAGNOSTIC_VERSION = 2
# Regional combinations entering the test bench: almost complete quarterly history
MIN_REPORTABLE_REGIONAL_QUARTERS = 20
# A combination is worth modelling per quarter when most recent quarters are reportable
MIN_REPORTABLE_QUARTERS = 12

SHEET_NOTES = {
    "Leggimi": "Cosa ha prodotto questo file: versioni, soglie e dati inclusi.",
    "Combinazioni": "Una riga per laboratorio x microrganismo x antibiotico: quanti dati ci sono e se bastano per una previsione trimestrale.",
    "Serie trimestrali": "Resistenza per trimestre, separata per popolazione: invasivi (sangue, liquor), invasivi non ricoverati («altro», a parte, fuori dalle previsioni), urine (dal 2023), altri materiali. Primo isolato per paziente, reparti sommati, intrinseche escluse.",
    "Serie mensili": "Come le serie trimestrali, per mese. Utile solo dove i numeri sono alti.",
    "Semaforo": "Per ogni combinazione e orizzonte: quale previsione verrà mostrata e con che affidabilità. Il modello è scelto sulle prove 2020-22 e giudicato solo sulle prove 2023-24. Un orizzonte non ha mai un semaforo migliore di quelli più vicini (colonna \"Semaforo del solo orizzonte\": il giudizio prima di questa regola).",
    "Banco di prova sintesi": "Quanto sbaglia la previsione più semplice (\"come gli ultimi 12 mesi\") sugli invasivi regionali, per orizzonte. È il riferimento che ogni modello deve battere.",
    "Banco di prova": "Dettaglio per combinazione e orizzonte. Errore medio e rumore in punti di % R; copertura = % di valori reali dentro l'intervallo dell'80% (accettabile 75-85%).",
    "Serie regionali invasivi": "Laboratori sommati, solo sangue e liquor, esclusi i pazienti non ricoverati \"altro\" (popolazione a parte nelle serie trimestrali): la serie su cui si costruisce la previsione regionale.",
    "Composizione": "Primi isolati per anno, per materiale e per tipo di paziente: un cambio di composizione cambia la % R senza che i batteri cambino.",
    "Antibiogramma": "Antibiogramma cumulativo annuale (CLSI M39): da confrontare con i report dei laboratori.",
    "Breakpoint": f"Riga EUCAST v{BREAKPOINTS_VERSION} usata per ogni coppia e quanto la lettura armonizzata differisce dal refertato.",
    "Breakpoint per anno": "Quota di risultati in cui R/non-R cambia rileggendo la MIC con la versione attuale: i salti indicano cambi di breakpoint.",
    "Anomalie import": "Riepilogo delle anomalie trovate nella validazione di ogni file caricato.",
    "Verifica dati nuovi": "Previsioni fatte con i soli dati fino al 31/12/2024 confrontate con i trimestri osservati dopo (almeno 30 isolati). Scostamento = osservato - previsto (positivo: il modello ha sottostimato). Accanto, cosa avrebbero previsto gli altri modelli candidati sugli stessi trimestri.",
    "Leggimi analisi": "Analisi degli scostamenti della verifica sui dati nuovi: cosa contiene questo file e come leggerlo. Solo dati aggregati per laboratorio, nessun codice paziente o campione.",
    "Sintesi scostamenti": "I numeri chiave: quante combinazioni sbagliano e in che verso, e quanta parte dello scostamento viene dal mix dei laboratori o da variazioni della resistenza non previste.",
    "Scostamenti combinazione": "Una riga per microrganismo x antibiotico, trimestri osservati sommati, dal più grande scostamento al più piccolo. Scostamento = osservato - previsto = di cui mix laboratori + di cui variazione non prevista (in punti di % R).",
    "Contributi laboratori": "Per ogni combinazione e laboratorio, diviso per tipo di paziente (ricoverati, pronto soccorso, non ricoverati...): peso e resistenza nell'anno prima del taglio e nei trimestri osservati, e quanto il laboratorio ha spostato il dato regionale. La somma dei contributi di una combinazione è la variazione regionale osservata.",
    "Laboratori": "Per laboratorio: in quante combinazioni è quello che sposta di più il dato regionale e di quanto in media. I laboratori nuovi o molto cresciuti sono la prima cosa da guardare.",
    "Modelli": "Errore di ogni modello candidato sugli stessi confronti. \"Migliore in\" = combinazioni in cui avrebbe sbagliato meno, col senno di poi: è una diagnosi, non un criterio di scelta.",
    "Modello mostrato": "Errore delle previsioni mostrate, raggruppate per il modello che il sistema aveva scelto.",
}

CAUSE_LABELS = {"rumore": "Nella variabilità casuale", "mix": "Mix dei laboratori",
                "reale": "Resistenza cambiata in modo non previsto"}

COLUMN_LABELS = {
    "popolazione": "Popolazione", "laboratorio": "Laboratorio", "ente": "Ente", "microorganismo": "Microrganismo", "antibiotic": "Antibiotico",
    "period": "Periodo", "isolates": "Primi isolati del microrganismo",
    "n_reported": "Testati (refertato)", "r_reported": "Resistenti (refertato)",
    "pct_r_reported": "% R refertato", "low_reported": "% R refertato min (IC95)", "high_reported": "% R refertato max (IC95)",
    "n_harmonized": "Testati (armonizzato)", "r_harmonized": "Resistenti (armonizzato)",
    "pct_r_harmonized": "% R armonizzato", "low_harmonized": "% R armonizzato min (IC95)", "high_harmonized": "% R armonizzato max (IC95)",
    "tested_share": "Quota testata", "reportable": f"Almeno {MIN_ISOLATES_PER_PERIOD} testati",
}


def build_diagnostic(dataset_folder: str | Path, uploads: list[dict], app_version: str,
                     validation: dict | None = None) -> bytes:
    """Return the diagnostic workbook as .xlsx bytes. validation: stored out-of-sample check, if any."""
    folder = Path(dataset_folder)
    isolates = load_isolates(folder)
    sheets: dict[str, pd.DataFrame] = {}

    if isolates.empty:
        sheets["Leggimi"] = _readme(app_version, uploads, None, None)
        return _write(sheets)

    quarterly = build_series(folder, "Q")
    monthly = build_series(folder, "M")
    sheets["Leggimi"] = _readme(app_version, uploads, isolates, quarterly)
    regional = regional_series(quarterly, "invasivi")
    bench_overview, bench_detail, traffic = _bench_sheets(quarterly, regional)
    sheets["Semaforo"] = traffic
    sheets["Banco di prova sintesi"] = bench_overview
    sheets["Banco di prova"] = bench_detail
    sheets["Serie regionali invasivi"] = _regional_sheet(regional)
    sheets["Combinazioni"] = _combinations(quarterly)
    sheets["Composizione"] = _composition(isolates)
    sheets["Serie trimestrali"] = _series_sheet(quarterly)
    sheets["Serie mensili"] = _series_sheet(monthly)
    sheets["Antibiogramma"] = _antibiogram_sheet(cumulative_antibiogram(folder))
    by_pair, by_year = _breakpoint_sheets(folder)
    sheets["Breakpoint"] = by_pair
    sheets["Breakpoint per anno"] = by_year
    sheets["Anomalie import"] = _anomalies(folder, uploads)
    if validation and validation.get("righe"):
        sheets["Verifica dati nuovi"] = _validation_sheet(validation)
        sheets.update(_analysis_sheets(validation))
    return _write(sheets)


def _readme(app_version: str, uploads: list[dict], isolates: pd.DataFrame | None,
            quarterly: pd.DataFrame | None) -> pd.DataFrame:
    rows = [
        ("Generato il", datetime.now().strftime("%d/%m/%Y %H:%M")),
        ("Versione file diagnostico", DIAGNOSTIC_VERSION),
        ("Versione software", f"{app_version} ({_git_commit()})"),
        ("Breakpoint per la lettura armonizzata", f"EUCAST v{BREAKPOINTS_VERSION}"),
        ("Regola primo isolato", "primo isolato per laboratorio x paziente x microrganismo nel periodo (anno, trimestre o mese)"),
        ("Soglia percentuale riportabile", f"almeno {MIN_ISOLATES_PER_PERIOD} isolati testati nel periodo"),
        ("Soglia per modellare una combinazione", f"almeno {MIN_REPORTABLE_QUARTERS} trimestri riportabili"),
        ("Contenuto", "solo dati aggregati: nessun codice paziente o campione"),
    ]
    if isolates is not None and not isolates.empty:
        rows += [
            ("Isolati validi in archivio", len(isolates)),
            ("Periodo dei dati", f"{isolates['data_prelievo'].min():%d/%m/%Y} - {isolates['data_prelievo'].max():%d/%m/%Y}"),
            ("Laboratori", ", ".join(sorted(isolates["laboratorio"].unique()))),
            ("Microrganismi", ", ".join(sorted(isolates["microorganismo"].unique()))),
            ("Combinazioni laboratorio x microrganismo x antibiotico", int(quarterly[["laboratorio", "microorganismo", "antibiotic"]].drop_duplicates().shape[0])),
        ]
    for u in uploads:
        rows.append((f"File caricato #{u['id']}", f"{u['original_name']} ({u['uploaded_at']}, {u['status']}, {u.get('row_count') or 0} osservazioni)"))
    rows += [(f"Foglio: {name}", note) for name, note in SHEET_NOTES.items() if name != "Leggimi"]
    return pd.DataFrame(rows, columns=["Voce", "Valore"])


MODEL_NAMES = {
    "persistenza_12m": "Come gli ultimi 12 mesi",
    "persistenza_tarata": "Come gli ultimi 12 mesi, intervallo tarato",
    "trend": "Trend smorzato",
    "trend_laboratori": "Trend con effetto laboratorio",
    "prophet": "Prophet (modello del progetto originale)",
}
TRAFFIC_LABELS = {
    "verde": "Affidabile",
    "giallo": "Indicativa",
    "grigio": "Non affidabile",
}


def _bench_sheets(quarterly: pd.DataFrame, regional: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reportable = regional.groupby(["microorganismo", "antibiotic"])["reportable"].sum()
    eligible = set(reportable[reportable >= MIN_REPORTABLE_REGIONAL_QUARTERS].index)
    labs = quarterly[quarterly["popolazione"] == "invasivi"]
    labs = labs[[k in eligible for k in zip(labs["microorganismo"], labs["antibiotic"])]]
    bench = run_bench(labs)
    if bench.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    summary = summarize(bench)
    over = overview(summary).assign(modello=lambda d: d["modello"].map(MODEL_NAMES))
    over = _round(over.rename(columns={
        "modello": "Previsione", "orizzonte_mesi": "Orizzonte (mesi)", "combinazioni": "Combinazioni",
        "errore_medio": "Errore tipico (punti % R, mediana)", "errore_peggiore_10pct": "Errore nel 10% peggiore",
        "copertura_80": "Copertura intervallo 80% (mediana)", "rumore_atteso": "Rumore inevitabile (punti)",
        "errore_su_rumore": "Errore / rumore (mediana)",
    }))
    detail = summary.assign(modello=lambda d: d["modello"].map(MODEL_NAMES)).sort_values(
        ["modello", "orizzonte_mesi", "errore_medio"], ascending=[True, True, False])
    detail = _round(detail.rename(columns={
        "microorganismo": "Microrganismo", "antibiotic": "Antibiotico", "modello": "Previsione",
        "orizzonte_mesi": "Orizzonte (mesi)", "prove": "Prove", "errore_medio": "Errore medio (punti)",
        "distorsione": "Distorsione (+ = sovrastima)", "copertura_80": "Copertura intervallo 80% (%)",
        "rumore_atteso": "Rumore inevitabile (punti)", "testati_medi": "Testati medi per trimestre",
        "errore_su_rumore": "Errore / rumore",
    }))
    verdict = choose_and_verify(bench)
    verdict = verdict.assign(
        semaforo=verdict["semaforo"].map(lambda v: f"{v} - {TRAFFIC_LABELS[v]}"),
        semaforo_orizzonte=verdict["semaforo_orizzonte"].map(lambda v: f"{v} - {TRAFFIC_LABELS[v]}"),
        scelto_su_2020_22=verdict["scelto_su_2020_22"].map(MODEL_NAMES),
        mostrato=verdict["mostrato"].map(MODEL_NAMES),
    ).sort_values(["orizzonte_mesi", "semaforo", "microorganismo", "antibiotic"])
    traffic = _round(verdict.rename(columns={
        "microorganismo": "Microrganismo", "antibiotic": "Antibiotico", "orizzonte_mesi": "Orizzonte (mesi)",
        "semaforo": "Semaforo",
        "semaforo_orizzonte": "Semaforo del solo orizzonte",
        "mostrato": "Previsione mostrata", "scelto_su_2020_22": "Modello scelto sulle prove 2020-22",
        "prove_verifica": "Prove di verifica (2023-24)", "errore_verifica": "Errore in verifica (punti)",
        "errore_riferimento_verifica": "Errore di 'come gli ultimi 12 mesi' (punti)",
        "guadagno_pct": "Guadagno rispetto al riferimento (%)", "copertura_verifica": "Copertura 80% in verifica (%)",
        "rumore_verifica": "Rumore inevitabile (punti)", "errore_su_rumore": "Errore / rumore",
    }))
    return over, detail, traffic


def _regional_sheet(regional: pd.DataFrame) -> pd.DataFrame:
    cols = ["microorganismo", "antibiotic", "period", "laboratori", "n_reported", "r_reported", "pct_r_reported",
            "n_harmonized", "r_harmonized", "pct_r_harmonized", "low_harmonized", "high_harmonized", "reportable"]
    out = regional[cols].copy()
    out["reportable"] = np.where(out["reportable"], "sì", "no")
    return _round(out.rename(columns=COLUMN_LABELS | {"laboratori": "Laboratori con dati"}))


def _combinations(quarterly: pd.DataFrame) -> pd.DataFrame:
    keys = ["popolazione", "laboratorio", "microorganismo", "antibiotic"]
    last_year = quarterly["period_start"] >= quarterly["period_start"].max() - pd.DateOffset(months=9)
    grouped = quarterly.groupby(keys)
    table = grouped.agg(
        trimestri=("period", "size"),
        trimestri_con_dati=("n_harmonized", lambda n: int((n > 0).sum())),
        trimestri_riportabili=("reportable", "sum"),
        testati_medi=("n_harmonized", "mean"),
        quota_testata_media=("tested_share", "mean"),
    ).reset_index()
    recent = quarterly[last_year].groupby(keys)[["n_reported", "r_reported", "n_harmonized", "r_harmonized"]].sum()
    recent["pct_r_refertato_ultimi_12_mesi"] = 100 * recent["r_reported"] / recent["n_reported"].replace(0, np.nan)
    recent["pct_r_armonizzato_ultimi_12_mesi"] = 100 * recent["r_harmonized"] / recent["n_harmonized"].replace(0, np.nan)
    table = table.merge(recent[["pct_r_refertato_ultimi_12_mesi", "pct_r_armonizzato_ultimi_12_mesi"]].reset_index(), on=keys, how="left")
    table["modellabile_trimestrale"] = np.where(table["trimestri_riportabili"] >= MIN_REPORTABLE_QUARTERS, "sì", "no")
    table = table.sort_values(["modellabile_trimestrale", "testati_medi"], ascending=[False, False])
    table.insert(2, "ente", table["laboratorio"].map(laboratory_names()))
    return _round(table.rename(columns={
        "popolazione": "Popolazione", "laboratorio": "Laboratorio", "ente": "Ente", "microorganismo": "Microrganismo", "antibiotic": "Antibiotico",
        "trimestri": "Trimestri nel periodo", "trimestri_con_dati": "Trimestri con dati",
        "trimestri_riportabili": f"Trimestri con almeno {MIN_ISOLATES_PER_PERIOD} testati",
        "testati_medi": "Testati medi per trimestre", "quota_testata_media": "Quota testata media",
        "pct_r_refertato_ultimi_12_mesi": "% R refertato ultimi 12 mesi",
        "pct_r_armonizzato_ultimi_12_mesi": "% R armonizzato ultimi 12 mesi",
        "modellabile_trimestrale": "Dati sufficienti per previsione trimestrale",
    }))


def _composition(isolates: pd.DataFrame) -> pd.DataFrame:
    first = isolates[first_isolate_mask(isolates, "Y")].assign(anno=lambda d: d["data_prelievo"].dt.year)
    parts = []
    for column, name, labels in [("materiale", "Materiale", MATERIAL_LABELS),
                                 ("paziente_ricoverato", "Paziente ricoverato", ADMISSION_LABELS)]:
        table = first.pivot_table(index=["microorganismo", column], columns="anno", values="isolate_id",
                                  aggfunc="size", fill_value=0)
        table = table.reset_index().rename(columns={"microorganismo": "Microrganismo", column: "Codice"})
        table.insert(1, "Variabile", name)
        table.insert(3, "Descrizione", table["Codice"].map(lambda c: label(c, labels)))
        parts.append(table)
    out = pd.concat(parts, ignore_index=True)
    out.columns = [str(c) for c in out.columns]
    return out


def _series_sheet(series: pd.DataFrame) -> pd.DataFrame:
    series = series.assign(ente=series["laboratorio"].map(laboratory_names()))
    cols = [c for c in COLUMN_LABELS if c in series.columns]
    out = series[cols].copy()
    out["reportable"] = np.where(out["reportable"], "sì", "no")
    return _round(out.rename(columns=COLUMN_LABELS))


def _antibiogram_sheet(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["riportabile"] = np.where(out["riportabile"], "sì", "no")
    return _round(out.rename(columns={
        "popolazione": "Popolazione", "laboratorio": "Laboratorio", "microorganismo": "Microrganismo",
        "antibiotic": "Antibiotico", "anno": "Anno", "testati": "Testati (primo isolato)", "s": "S", "i": "I", "r": "R",
        "pct_s": "% S", "pct_r": "% R", "pct_r_low": "% R min (IC95)", "pct_r_high": "% R max (IC95)",
        "riportabile": f"Riportabile (almeno {MIN_ISOLATES_PER_PERIOD})",
    }))


def _breakpoint_sheets(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ast = load_ast_results(folder)
    ast = ast[~ast["intrinsic"].astype(bool)]
    compared = ast[ast["harmonized_source"].eq("mic") & ast["result"].notna()].copy()
    compared["discorde"] = compared["result"].eq("R") != compared["result_harmonized"].eq("R")
    compared["anno"] = compared["data_prelievo"].dt.year

    usage = ast.groupby(["microorganismo", "antibiotic"]).agg(
        risultati=("antibiotic", "size"),
        da_mic=("harmonized_source", lambda s: int(s.eq("mic").sum())),
        da_refertato=("harmonized_source", lambda s: int(s.eq("refertato").sum())),
    )
    disc = compared.groupby(["microorganismo", "antibiotic"])["discorde"].agg(["sum", "mean"])
    usage = usage.join(disc).reset_index().rename(columns={"sum": "discordi_r", "mean": "quota_discordi"})
    table = breakpoint_table()
    if not table.empty:
        table = table.rename(columns={"pathogen": "microorganismo"})
        usage = usage.merge(table[["microorganismo", "antibiotic", "eucast_agent", "s_le", "r_gt", "status", "bracketed"]],
                            on=["microorganismo", "antibiotic"], how="left")
    usage["quota_discordi"] = 100 * usage["quota_discordi"]
    by_pair = _round(usage.rename(columns={
        "microorganismo": "Microrganismo", "antibiotic": "Antibiotico", "risultati": "Risultati",
        "da_mic": "Letti dalla MIC", "da_refertato": "Dal refertato (MIC assente/non interpretabile)",
        "discordi_r": "R/non-R diverso dal refertato", "quota_discordi": "% diversi",
        "eucast_agent": "Riga EUCAST usata", "s_le": "S ≤", "r_gt": "R >", "status": "Stato breakpoint",
        "bracketed": "Valore tra parentesi in EUCAST",
    }))

    by_year = compared.groupby(["microorganismo", "antibiotic", "anno"])["discorde"].agg(["size", "mean"]).reset_index()
    by_year = by_year.pivot_table(index=["microorganismo", "antibiotic"], columns="anno", values="mean")
    by_year = (100 * by_year).round(1).reset_index()
    by_year.columns = ["Microrganismo", "Antibiotico"] + [f"% diversi {c}" for c in by_year.columns[2:]]
    return by_pair, by_year


def _anomalies(folder: Path, uploads: list[dict]) -> pd.DataFrame:
    rows = []
    for u in uploads:
        path = folder / "reports" / f"upload_{u['id']}.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        for item in report.get("issue_summary", []):
            rows.append({"File": u["original_name"], "Gravità": item["gravita"], "Problema": item["problema"],
                         "Colonna": item["colonna"], "Righe": item["righe"]})
    return pd.DataFrame(rows, columns=["File", "Gravità", "Problema", "Colonna", "Righe"])


def _round(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes("float").columns
    df[floats] = df[floats].round(2)
    return df


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "commit sconosciuto"
    except (OSError, subprocess.SubprocessError):
        return "commit sconosciuto"


def _write(sheets: dict[str, pd.DataFrame]) -> bytes:
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False, startrow=2)
            ws = writer.sheets[name]
            ws["A1"] = SHEET_NOTES.get(name, "")
            ws["A1"].font = Font(italic=True)
            for cell in ws[3]:
                cell.font = Font(bold=True)
            ws.freeze_panes = "A4"
            for i, col in enumerate(frame.columns, start=1):
                width = max([len(str(col))] + [len(str(v)) for v in frame[col].head(200)])
                ws.column_dimensions[get_column_letter(i)].width = min(60, max(10, width + 2))
    return buffer.getvalue()


def _validation_sheet(validation: dict) -> pd.DataFrame:
    table = pd.DataFrame(validation["righe"])
    table = table.assign(
        semaforo=table["semaforo"].map(lambda v: f"{v} - {TRAFFIC_LABELS[v]}"),
        modello=table["modello"].map(MODEL_NAMES),
        dentro=np.where(table["dentro"], "sì", "no"),
    ).drop(columns=["microorganismo_nome", "antibiotico_nome", "periodo"])
    names = {
        "microorganismo": "Microrganismo", "antibiotico": "Antibiotico", "trimestre": "Trimestre",
        "orizzonte_mesi": "Orizzonte (mesi)", "semaforo": "Semaforo di allora", "modello": "Previsione mostrata",
        "n": "Testati", "previsto": "Previsto (%)", "basso": "Intervallo 80% min", "alto": "Intervallo 80% max",
        "osservato": "Osservato (%)", "errore": "Errore (punti)", "scostamento": "Scostamento (punti)",
        "dentro": "Dentro l'intervallo",
    }
    for model, model_name in MODEL_NAMES.items():
        names[f"previsto_{model}"] = f"{model_name} (%)"
        names[f"errore_{model}"] = f"Errore {model_name}"
    return _round(table.rename(columns=names))


def build_validation_workbook(validation: dict, app_version: str) -> bytes:
    """The out-of-sample comparison and its error analysis, aggregated figures only."""
    sheets = {"Leggimi analisi": _analysis_readme(validation, app_version)}
    sheets.update(_analysis_sheets(validation))
    if validation.get("righe"):
        sheets["Verifica dati nuovi"] = _validation_sheet(validation)
    return _write(sheets)


def _analysis_readme(validation: dict, app_version: str) -> pd.DataFrame:
    analysis = validation.get("analisi") or {}
    rows = [
        ("Generato il", datetime.now().strftime("%d/%m/%Y %H:%M")),
        ("Versione applicazione", app_version),
        ("Commit", _git_commit()),
        ("Verifica calcolata il", validation.get("calcolato_il", "")),
        ("Previsioni fatte con i dati fino al", validation.get("dati_fino_al", "")),
        ("Trimestri osservati", ", ".join(validation.get("trimestri_osservati", []))),
        ("Anno di riferimento (prima del taglio)", analysis.get("periodo_base", "")),
        ("Popolazione", "isolati invasivi (sangue e liquor) esclusi i pazienti non ricoverati \"altro\", lettura armonizzata EUCAST, primo isolato"),
        ("Confronti contati", f"trimestri con almeno {MIN_ISOLATES_PER_PERIOD} isolati testati; solo previsioni "
                              "mostrate all'operatore (semaforo verde o giallo)"),
        ("Scostamento", "osservato - previsto, in punti di % R: positivo = il modello ha sottostimato"),
        ("Mix dei laboratori", "parte dello scostamento dovuta a laboratori che pesano di più o di meno, "
                               "o che sono entrati o usciti dalla rete"),
        ("Variazione non prevista", "variazione della resistenza dentro i laboratori, meno quella che il "
                                    "modello aveva previsto"),
        ("Nella variabilità casuale", "scostamento entro 1,96 errori standard del valore osservato: "
                                      "nessun modello può fare meglio"),
        ("Migliore col senno di poi", "solo diagnosi: scegliere il modello sui dati 2025 consumerebbe "
                                      "l'unica verifica che nessun modello ha visto"),
    ]
    return pd.DataFrame(rows, columns=["Voce", "Valore"])


def _analysis_sheets(validation: dict) -> dict[str, pd.DataFrame]:
    analysis = validation.get("analisi") or {}
    if not analysis.get("combinazioni"):
        return {}
    s = analysis["sintesi"]
    summary = [
        ("Combinazioni analizzate", s["combinazioni"]),
        ("Il modello ha sovrastimato", s["sovrastima"]),
        ("Il modello ha sottostimato", s["sottostima"]),
        ("Sempre nello stesso verso in tutti i trimestri", s["stesso_verso"]),
        ("Scostamento medio (punti, con segno)", s["scostamento_medio"]),
        ("Scostamento medio assoluto (punti)", s["scostamento_assoluto_medio"]),
        ("Quota dello scostamento dal mix dei laboratori (%)", s["quota_mix"]),
        ("Quota dello scostamento da variazioni non previste (%)", s["quota_non_previsto"]),
        *[(f"Causa principale: {CAUSE_LABELS[k]}", v) for k, v in s["cause"].items()],
        *[(f"Scostamento medio a {h} mesi (punti, con segno)", v) for h, v in s["distorsione_per_orizzonte"].items()],
    ]
    combos = []
    for c in analysis["combinazioni"]:
        row = {
            "Microrganismo": c["microorganismo"], "Antibiotico": c["antibiotico"],
            "Previsione mostrata": MODEL_NAMES.get(c["modello"], c["modello"]), "Trimestri": c["trimestri"],
            "Testati anno prima": c["n_base"], "Testati osservati": c["n_osservati"],
            "% R anno prima": c["base"], "% R previsto": c["previsto"], "% R osservato": c["osservato"],
            "Variazione prevista": c["variazione_prevista"], "Variazione osservata": c["variazione_osservata"],
            "Scostamento": c["scostamento"], "di cui mix laboratori": c["di_cui_mix"],
            "di cui variazione non prevista": c["di_cui_non_previsto"],
            "Errore casuale atteso (1 ES)": c["errore_casuale"], "Causa principale": CAUSE_LABELS[c["causa"]],
            "Trimestri sovrastimati": c["sovrastime"], "Trimestri sottostimati": c["sottostime"],
            "Stesso verso": "sì" if c["stesso_verso"] else "no",
            "Laboratori nuovi": ", ".join(c["laboratori_nuovi"]), "Laboratori usciti": ", ".join(c["laboratori_usciti"]),
            "Migliore col senno di poi": MODEL_NAMES.get(c["migliore_col_senno_di_poi"], c["migliore_col_senno_di_poi"]
                                                          if c["migliore_col_senno_di_poi"] != "mostrato" else "Previsione mostrata"),
        }
        for model, error in c["errori_modelli"].items():
            row[f"Errore {MODEL_NAMES.get(model, 'previsione mostrata')}"] = error
        combos.append(row)
    labs = pd.DataFrame(analysis["laboratori"]).rename(columns={
        "microorganismo": "Microrganismo", "antibiotico": "Antibiotico", "laboratorio": "Laboratorio",
        "stato": "Stato", "n_base": "Testati anno prima", "n_osservati": "Testati osservati",
        "peso_base": "Peso anno prima (%)", "peso_osservato": "Peso osservato (%)",
        "resistenza_base": "% R anno prima", "resistenza_osservata": "% R osservato",
        "contributo_mix": "Contributo mix (punti)", "contributo_reale": "Contributo variazione (punti)",
        "contributo": "Contributo totale (punti)",
    })
    names = laboratory_names()
    labs.insert(3, "Ente", labs["Laboratorio"].map(lambda unit: names.get(unit.split(" · ")[0], "")))
    lab_summary = pd.DataFrame(analysis["laboratori_sintesi"]).rename(columns={
        "laboratorio": "Laboratorio", "stato": "Stato", "isolati_base": "Primi isolati anno prima",
        "isolati_osservati": "Primi isolati nei trimestri osservati", "combinazioni": "Combinazioni",
        "primo_contributo": "Combinazioni in cui sposta di più", "contributo_medio_assoluto": "Spostamento medio (punti)",
        "contributo_mix_medio": "di cui mix (media, con segno)", "contributo_reale_medio": "di cui variazione (media, con segno)",
    })
    lab_summary.insert(1, "Ente", lab_summary["Laboratorio"].map(lambda unit: names.get(unit.split(" · ")[0], "")))
    model_names = {**MODEL_NAMES, "mostrato": "Previsione mostrata"}
    models = pd.DataFrame(analysis["modelli"]["complessivo"]).assign(
        modello=lambda d: d["modello"].map(model_names)).rename(columns={
        "modello": "Modello", "confronti": "Confronti", "errore": "Errore medio (punti)",
        "distorsione": "Scostamento medio con segno", "migliore_in": "Migliore in (combinazioni)"})
    shown = pd.DataFrame(analysis["modelli"]["per_modello_mostrato"]).assign(
        modello_mostrato=lambda d: d["modello_mostrato"].map(MODEL_NAMES)).rename(columns={
        "modello_mostrato": "Modello scelto", "confronti": "Confronti", "combinazioni": "Combinazioni",
        "errore": "Errore medio (punti)", "dentro": "Dentro l'intervallo"})
    return {
        "Sintesi scostamenti": _round(pd.DataFrame(summary, columns=["Voce", "Valore"]).assign(
            Valore=lambda d: d["Valore"].map(lambda v: round(v, 2) if isinstance(v, float) else v))),
        "Scostamenti combinazione": _round(pd.DataFrame(combos)),
        "Contributi laboratori": _round(labs),
        "Laboratori": _round(lab_summary),
        "Modelli": _round(models),
        "Modello mostrato": _round(shown),
    }
