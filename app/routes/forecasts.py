"""Operator pages: validated regional forecasts with traffic light and plain words."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import pandas as pd

from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, send_file, url_for

from app import APP_VERSION
from app.routes.guide import current_workflow
from app.services import jobs, model_search
from app.services.diagnostics import CAUSE_LABELS, MODEL_NAMES, build_validation_workbook
from app.services.out_of_sample import (
    COMPARED, compute_validation, cutoff_label, load_validation, save_validation, validation_possible,
)
from app.services.reference import laboratory_names
from app.services.forecast_run import compute_forecasts, load_forecasts, save_forecasts, stale_reason

bp = Blueprint("forecasts", __name__)

TRAFFIC = {
    "verde": {"label": "Affidabile", "badge": "success",
              "why": "Le previsioni fatte nel 2023-2024 sono state vicine al valore reale e l'intervallo è stato affidabile."},
    "giallo": {"label": "Indicativa", "badge": "warning",
               "why": "L'intervallo è affidabile, ma l'errore è stato più ampio di quanto si spiega con il solo "
                      "numero di isolati: usare come indicazione di massima."},
    "grigio": {"label": "Non affidabile", "badge": "secondary",
               "why": "Nelle prove sul passato l'intervallo non conteneva il valore reale abbastanza spesso, "
                      "oppure le prove sono troppo poche. Si mostra solo il dato degli ultimi 12 mesi."},
}
MODEL_WORDS = {
    "persistenza_tarata": "Previsione basata sul livello degli ultimi 12 mesi: nei dati non c'è una tendenza "
                          "che un modello riesca ad anticipare meglio.",
    "trend": "Previsione basata sulla tendenza recente, smorzata per non estrapolare troppo lontano.",
    "trend_laboratori": "Previsione basata sulla tendenza recente, al netto dei cambiamenti nel numero di isolati "
                        "inviati dai singoli laboratori.",
    "prophet": "Previsione del modello Prophet (tendenza e andamento stagionale), che nelle prove sul passato "
               "ha sbagliato meno degli altri.",
}
COMPARED_NAMES = {"persistenza_12m": "come gli ultimi 12 mesi", "prophet": "Prophet"}
HORIZONS = (3, 6, 9, 12)


def _folder():
    return current_app.config["FORECAST_FOLDER"]


def _forecast_for(combo: dict, horizon: int) -> dict | None:
    return next((f for f in combo["previsioni"] if f["orizzonte_mesi"] == horizon), None)


def past_test_sentence(f: dict) -> str:
    inside = round((f["copertura"] or 0) * f["prove"] / 100)
    return (f"Su {f['prove']} previsioni simili fatte nel 2023-2024 l'errore tipico è stato di "
            f"{f['errore_tipico']:.1f}".replace(".", ",") + " punti percentuali; il valore reale era dentro l'intervallo "
            f"in {inside} casi su {f['prove']}.")


@bp.get("/previsioni")
def overview():
    data = load_forecasts(_folder())
    if not data or not data["meta"].get("available"):
        return render_template("forecasts.html", data=None, wf=current_workflow())
    combos = data["combinations"]
    pathogens = sorted({(c["microorganismo"], c["microorganismo_nome"]) for c in combos}, key=lambda p: p[1])
    selected = request.args.get("microrganismo") or (pathogens[0][0] if pathogens else "")
    horizon = int(request.args.get("orizzonte", 12))
    if horizon not in HORIZONS:
        horizon = 12
    rows = []
    for c in combos:
        if c["microorganismo"] != selected:
            continue
        rows.append({"combo": c, "forecast": _forecast_for(c, horizon)})
    order = {"verde": 0, "giallo": 1, "grigio": 2}
    rows.sort(key=lambda r: (r["forecast"] is None, order.get((r["forecast"] or {}).get("semaforo"), 3),
                             r["combo"]["antibiotico_nome"]))
    stale = stale_reason(data["meta"], current_app.config["DATASET_FOLDER"])
    return render_template("forecasts.html", data=data, rows=rows, pathogens=pathogens, selected=selected,
                           horizon=horizon, horizons=HORIZONS, traffic=TRAFFIC, stale=stale, wf=current_workflow())


@bp.get("/previsioni/<pathogen>/<antibiotic>")
def detail(pathogen: str, antibiotic: str):
    data = load_forecasts(_folder())
    if not data:
        abort(404)
    combo = next((c for c in data["combinations"]
                  if c["microorganismo"] == pathogen and c["antibiotico"] == antibiotic), None)
    if combo is None:
        abort(404)
    forecasts = [dict(f, frase_prova=past_test_sentence(f)) for f in combo["previsioni"]]
    model = forecasts[0]["modello"] if forecasts else None
    return render_template("forecast_detail.html", data=data, combo=combo, forecasts=forecasts,
                           traffic=TRAFFIC, model_words=MODEL_WORDS.get(model))


@bp.post("/previsioni/calcola")
def compute():
    app = current_app._get_current_object()
    busy = jobs.start("previsioni", app, lambda progress: _compute_forecasts(app, progress))
    _flash_started("previsioni", busy)
    return redirect(request.referrer or url_for("guide.home"))


def _compute_forecasts(app, progress) -> dict:
    progress("Prove dei modelli sui dati passati e calcolo dei quattro trimestri successivi.")
    result = compute_forecasts(app.config["DATASET_FOLDER"])
    if not result["meta"].get("available"):
        raise RuntimeError("Nessun dato in archivio: carica prima l'export della Rete MIC (passo 1).")
    save_forecasts(result, app.config["FORECAST_FOLDER"])
    meta = result["meta"]
    return {"message": f"Previsioni calcolate per {meta['con_previsione']} combinazioni, dati fino al "
                       f"{meta['dati_fino_al']}. Ora scegli un microrganismo e leggi la tabella.",
            "link_endpoint": "forecasts.overview"}


def _flash_started(kind: str, busy: dict | None) -> None:
    if busy:
        flash(f"È già in corso: {busy['label'].lower()}. Attendi che finisca, poi riprova.", "warning")
    else:
        flash(f"{jobs.LABELS[kind]} avviato sul server: puoi chiudere la pagina, il lavoro continua. "
              "L'avanzamento è nella barra in alto.", "info")


@bp.get("/validazione")
def validation():
    result = load_validation(_folder())
    rows = (result or {}).get("righe", [])
    pathogens = sorted({(r["microorganismo"], r["microorganismo_nome"]) for r in rows}, key=lambda p: p[1])
    selected = request.args.get("microrganismo") or (pathogens[0][0] if pathogens else "")
    shown = sorted((r for r in rows if r["microorganismo"] == selected),
                   key=lambda r: (r["antibiotico_nome"], r["periodo"]))
    stale = stale_reason(result, current_app.config["DATASET_FOLDER"])
    analysis = (result or {}).get("analisi") or {}
    return render_template("validation_2025.html", result=result, rows=shown, pathogens=pathogens, stale=stale,
                           analysis=analysis, causes=CAUSE_LABELS, model_names=MODEL_NAMES,
                           lab_names=laboratory_names(),
                           selected=selected, traffic=TRAFFIC, compared=COMPARED, compared_names=COMPARED_NAMES,
                           cutoff=cutoff_label(), wf=current_workflow())


@bp.get("/validazione/analisi.xlsx")
def validation_workbook():
    result = load_validation(_folder()) or abort(404)
    return send_file(
        io.BytesIO(build_validation_workbook(result, APP_VERSION)),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"analisi_scostamenti_{datetime.now():%Y%m%d_%H%M}.xlsx",
    )


@bp.post("/validazione/calcola")
def compute_validation_route():
    if not validation_possible(current_app.config["DATASET_FOLDER"]):
        flash(f"Per la verifica servono dati successivi al {cutoff_label()}: carica l'export aggiornato.", "warning")
        return redirect(url_for("guide.home"))
    app = current_app._get_current_object()
    busy = jobs.start("verifica", app, lambda progress: _compute_validation(app, progress))
    _flash_started("verifica", busy)
    return redirect(url_for("forecasts.validation"))


def _compute_validation(app, progress) -> dict:
    progress(f"Previsioni rifatte con i soli dati fino al {cutoff_label()} e confronto con i trimestri successivi.")
    result = compute_validation(app.config["DATASET_FOLDER"])
    if not result.get("available"):
        raise RuntimeError(f"Per la verifica servono dati successivi al {cutoff_label()}.")
    save_validation(result, app.config["FORECAST_FOLDER"])
    return {"message": "Verifica completata: confronto fra le previsioni e i valori osservati, con l'analisi "
                       "degli scostamenti.", "link_endpoint": "forecasts.validation"}


@bp.get("/modelli")
def model_search_page():
    summary = model_search.load_summary(_folder())
    board = pd.DataFrame((summary or {}).get("classifica", []))
    tops = {target: g.sort_values("errore_verifica").head(8).to_dict("records")
            for target, g in board.groupby("obiettivo")} if not board.empty else {}
    has_workbook = (Path(_folder()) / model_search.WORKBOOK_FILE).exists()
    return render_template("models_search.html", summary=summary, tops=tops, has_workbook=has_workbook,
                           references=model_search.TARGET_REFERENCE, wf=current_workflow())


@bp.post("/modelli/ricerca")
def model_search_run():
    app = current_app._get_current_object()
    busy = jobs.start("ricerca", app, lambda progress: _run_model_search(app, progress))
    _flash_started("ricerca", busy)
    return redirect(url_for("forecasts.model_search_page"))


@bp.get("/modelli/ricerca.xlsx")
def model_search_workbook():
    path = Path(_folder()) / model_search.WORKBOOK_FILE
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"ricerca_modelli_{datetime.now():%Y%m%d_%H%M}.xlsx")


def _run_model_search(app, progress) -> dict:
    result = model_search.run_search(app.config["DATASET_FOLDER"], progress=lambda message: progress(message))
    folder = app.config["FORECAST_FOLDER"]
    model_search.write_workbook(result, Path(folder) / model_search.WORKBOOK_FILE)
    model_search.save_summary(result, folder)
    return {"message": f"Provate {result['configurazioni']} configurazioni su {result['combinazioni']} combinazioni "
                       f"({len(result['prove']):,} prove).".replace(",", "."),
            "link_endpoint": "forecasts.model_search_page"}
