"""Operator pages: validated regional forecasts with traffic light and plain words."""
from __future__ import annotations

from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, url_for

from app.routes.guide import current_workflow
from app.services.out_of_sample import (
    COMPARED, compute_validation, cutoff_label, load_validation, save_validation,
)
from app.services.forecast_run import archive_signature, compute_forecasts, load_forecasts, save_forecasts

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
    stale = data["meta"].get("archivio") != archive_signature(current_app.config["DATASET_FOLDER"])
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
    result = compute_forecasts(current_app.config["DATASET_FOLDER"])
    if not result["meta"].get("available"):
        flash("Nessun dato in archivio: carica prima l'export della Rete MIC (passo 1).", "warning")
        return redirect(url_for("guide.home"))
    save_forecasts(result, _folder())
    meta = result["meta"]
    flash(f"Passo 3 completato: previsioni calcolate per {meta['con_previsione']} combinazioni, "
          f"dati fino al {meta['dati_fino_al']}. Ora scegli un microrganismo e leggi la tabella.", "success")
    return redirect(url_for("forecasts.overview"))


@bp.get("/validazione")
def validation():
    result = load_validation(_folder())
    rows = (result or {}).get("righe", [])
    pathogens = sorted({(r["microorganismo"], r["microorganismo_nome"]) for r in rows}, key=lambda p: p[1])
    selected = request.args.get("microrganismo") or (pathogens[0][0] if pathogens else "")
    shown = sorted((r for r in rows if r["microorganismo"] == selected),
                   key=lambda r: (r["antibiotico_nome"], r["periodo"]))
    stale = bool(result) and result.get("archivio") != archive_signature(current_app.config["DATASET_FOLDER"])
    return render_template("validation_2025.html", result=result, rows=shown, pathogens=pathogens, stale=stale,
                           selected=selected, traffic=TRAFFIC, compared=COMPARED, compared_names=COMPARED_NAMES,
                           cutoff=cutoff_label(), wf=current_workflow())


@bp.post("/validazione/calcola")
def compute_validation_route():
    result = compute_validation(current_app.config["DATASET_FOLDER"])
    if not result.get("available"):
        flash(f"Per la verifica servono dati successivi al {cutoff_label()}: carica l'export aggiornato.", "warning")
        return redirect(url_for("guide.home"))
    save_validation(result, _folder())
    flash("Verifica completata: confronto fra le previsioni e i valori osservati.", "success")
    return redirect(url_for("forecasts.validation"))
