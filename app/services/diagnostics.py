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
from .series import MIN_ISOLATES_PER_PERIOD, build_series, cumulative_antibiogram

DIAGNOSTIC_VERSION = 1
# A combination is worth modelling per quarter when most recent quarters are reportable
MIN_REPORTABLE_QUARTERS = 12

SHEET_NOTES = {
    "Leggimi": "Cosa ha prodotto questo file: versioni, soglie e dati inclusi.",
    "Combinazioni": "Una riga per laboratorio x microrganismo x antibiotico: quanti dati ci sono e se bastano per una previsione trimestrale.",
    "Serie trimestrali": "Resistenza per trimestre, separata per popolazione: invasivi (sangue, liquor), urine (dal 2023), altri materiali. Primo isolato per paziente, reparti sommati, intrinseche escluse.",
    "Serie mensili": "Come le serie trimestrali, per mese. Utile solo dove i numeri sono alti.",
    "Composizione": "Primi isolati per anno, per materiale e per tipo di paziente: un cambio di composizione cambia la % R senza che i batteri cambino.",
    "Antibiogramma": "Antibiogramma cumulativo annuale (CLSI M39): da confrontare con i report dei laboratori.",
    "Breakpoint": f"Riga EUCAST v{BREAKPOINTS_VERSION} usata per ogni coppia e quanto la lettura armonizzata differisce dal refertato.",
    "Breakpoint per anno": "Quota di risultati in cui R/non-R cambia rileggendo la MIC con la versione attuale: i salti indicano cambi di breakpoint.",
    "Anomalie import": "Riepilogo delle anomalie trovate nella validazione di ogni file caricato.",
}

COLUMN_LABELS = {
    "popolazione": "Popolazione", "laboratorio": "Laboratorio", "ente": "Ente", "microorganismo": "Microrganismo", "antibiotic": "Antibiotico",
    "period": "Periodo", "isolates": "Primi isolati del microrganismo",
    "n_reported": "Testati (refertato)", "r_reported": "Resistenti (refertato)",
    "pct_r_reported": "% R refertato", "low_reported": "% R refertato min (IC95)", "high_reported": "% R refertato max (IC95)",
    "n_harmonized": "Testati (armonizzato)", "r_harmonized": "Resistenti (armonizzato)",
    "pct_r_harmonized": "% R armonizzato", "low_harmonized": "% R armonizzato min (IC95)", "high_harmonized": "% R armonizzato max (IC95)",
    "tested_share": "Quota testata", "reportable": f"Almeno {MIN_ISOLATES_PER_PERIOD} testati",
}


def build_diagnostic(dataset_folder: str | Path, uploads: list[dict], app_version: str) -> bytes:
    """Return the diagnostic workbook as .xlsx bytes."""
    folder = Path(dataset_folder)
    isolates = load_isolates(folder)
    sheets: dict[str, pd.DataFrame] = {}

    if isolates.empty:
        sheets["Leggimi"] = _readme(app_version, uploads, None, None)
        return _write(sheets)

    quarterly = build_series(folder, "Q")
    monthly = build_series(folder, "M")
    sheets["Leggimi"] = _readme(app_version, uploads, isolates, quarterly)
    sheets["Combinazioni"] = _combinations(quarterly)
    sheets["Composizione"] = _composition(isolates)
    sheets["Serie trimestrali"] = _series_sheet(quarterly)
    sheets["Serie mensili"] = _series_sheet(monthly)
    sheets["Antibiogramma"] = _antibiogram_sheet(cumulative_antibiogram(folder))
    by_pair, by_year = _breakpoint_sheets(folder)
    sheets["Breakpoint"] = by_pair
    sheets["Breakpoint per anno"] = by_year
    sheets["Anomalie import"] = _anomalies(folder, uploads)
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
