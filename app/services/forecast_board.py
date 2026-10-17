"""Data of the forecast dashboard (page "Previsioni di resistenza"): the stored
forecasts (forecast_run) sliced with the same list boxes as the consultation page.

Selections: pathogen and antibiotic (one each), a laboratory (one at a time: the
regional figure when none), the horizon (a forecast quarter or the next 12 months)
and the reliability classes shown in the overview table and the antibiogram.
"""
from __future__ import annotations

YEAR = "anno"  # horizon value of the next-12-months forecast
HORIZONS = ("3", "6", "9", "12")
RELIABILITY = {"verde": "affidabile", "giallo": "indicativa", "grigio": "non affidabile", "assente": "dati insufficienti"}
SINGLE = ["microorganismo", "antibiotico", "orizzonte"]
MULTI = ["laboratorio", "affidabilita"]
ONE_AT_A_TIME = ["laboratorio"]
LABELS = {"microorganismo": "Microrganismo", "antibiotico": "Antibiotico", "orizzonte": "Previsione per",
          "laboratorio": "Laboratorio", "affidabilita": "Affidabilità"}

TRAFFIC_WHY = {
    "verde": "Le previsioni fatte nel 2023-2024 sono state vicine al valore reale e l'intervallo è stato affidabile.",
    "giallo": "L'intervallo è affidabile, ma l'errore è stato più ampio di quanto si spiega con il solo numero di "
              "isolati: usare come indicazione di massima.",
    "grigio": "Nelle prove sul passato l'intervallo non conteneva il valore reale abbastanza spesso, oppure le prove "
              "sono troppo poche. Si mostra solo il dato degli ultimi 12 mesi.",
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


def _decimal(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def past_test_sentence(f: dict) -> str:
    inside = round((f["copertura"] or 0) * f["prove"] / 100)
    return (f"Su {f['prove']} previsioni simili fatte nel 2023-2024 l'errore tipico è stato di "
            f"{_decimal(f['errore_tipico'])} punti percentuali; il valore reale era dentro l'intervallo "
            f"in {inside} casi su {f['prove']}.")


def check_sentence(check: dict | None, subject: str) -> str:
    """The test of a 12-month forecast (regional or per laboratory) in words."""
    if not check or check.get("errore") is None:
        return ""
    text = f"Nelle prove degli ultimi due anni {subject} ha sbagliato in media di {_decimal(check['errore'])} punti"
    if check.get("copertura") is not None:
        text += f"; l'intervallo ha contenuto il valore reale nel {check['copertura']:.0f}% dei casi ({check['prove']} prove)"
    return text + "."


def _value(row: dict | None, check: dict | None = None) -> dict | None:
    if not row:
        return None
    return {"pct": round(row["pct"], 1), "basso": round(row["low"], 1), "alto": round(row["high"], 1),
            "semaforo": row["semaforo"], "errore": (check or {}).get("errore", row.get("errore_tipico"))}


def forecast_at(combo: dict, horizon: str, lab: str = "") -> dict | None:
    """The forecast of one combination at a horizon, regional or of one laboratory (12 months only)."""
    if lab:
        if horizon != YEAR:
            return None
        rows = combo.get("previsioni_laboratori") or {}
        return _value(next((r for r in rows.get("laboratori", []) if r["codice"] == lab), None), rows.get("verifica"))
    if horizon == YEAR:
        row = combo.get("prossimi_12_mesi")
        return _value(row, (row or {}).get("verifica"))
    return _value(next((f for f in combo.get("previsioni", []) if str(f["orizzonte_mesi"]) == horizon), None))


def _class(value: dict | None) -> str:
    return value["semaforo"] if value else "assente"


def _box(values: list[tuple[str, str, int | None, bool]], selected) -> list[dict]:
    chosen = set(selected if isinstance(selected, list) else [selected])
    return [{"valore": v, "nome": name, "conteggio": count,
             "stato": "selezionato" if v in chosen else "possibile" if possible else "escluso"}
            for v, name, count, possible in values]


def board(data: dict | None, filters: dict, stale: str | None = None) -> dict:
    """Everything the forecast dashboard shows for one selection."""
    if not data or not data.get("meta", {}).get("available"):
        return {"vuoto": True}
    meta, combos = data["meta"], data["combinations"]
    by_key = {(c["microorganismo"], c["antibiotico"]): c for c in combos}
    pathogens = sorted({c["microorganismo"]: c["microorganismo_nome"] for c in combos}.items(), key=lambda p: p[1])

    labs_known = meta.get("laboratori", {})
    lab = next(iter(filters.get("laboratorio") or []), "")
    lab = lab if lab in labs_known else ""
    horizon = filters.get("orizzonte") if filters.get("orizzonte") in HORIZONS + (YEAR,) else YEAR
    if lab:
        horizon = YEAR  # per laboratory only the next 12 months are forecast
    classes = [c for c in filters.get("affidabilita") or [] if c in RELIABILITY]

    pathogen = filters.get("microorganismo")
    if pathogen not in dict(pathogens):
        with_forecast = [p for p, _ in pathogens if any(c["previsioni"] for c in combos if c["microorganismo"] == p)]
        pathogen = with_forecast[0] if with_forecast else pathogens[0][0]
    of_pathogen = sorted((c for c in combos if c["microorganismo"] == pathogen), key=lambda c: c["antibiotico_nome"])
    antibiotic = filters.get("antibiotico")
    if (pathogen, antibiotic) not in by_key:
        ranked = sorted(of_pathogen, key=lambda c: (forecast_at(c, horizon, lab) is None,
                                                    -((c.get("ultimi_12_mesi") or {}).get("n") or 0)))
        antibiotic = ranked[0]["antibiotico"]
    combo = by_key[(pathogen, antibiotic)]

    def tested(c: dict) -> int:
        if lab:
            return next((r["n"] for r in c.get("laboratori", []) if r["codice"] == lab), 0)
        return (c.get("ultimi_12_mesi") or {}).get("n") or 0

    # Overview of the pathogen: every antibiotic at the horizon, filtered by reliability
    table = []
    for c in of_pathogen:
        value = forecast_at(c, horizon, lab)
        observed = c.get("ultimi_12_mesi")
        if lab:
            row = next((r for r in c.get("laboratori", []) if r["codice"] == lab), None)
            observed = {"pct": row["pct"], "n": row["n"], "riportabile": row["riportabile"]} if row else None
        table.append({"antibiotico": c["antibiotico"], "nome": c["antibiotico_nome"], "osservato": observed,
                      "tendenza": None if lab else c["tendenza"], "previsione": value, "classe": _class(value)})
    # as the earlier overview: reliable first, then indicative, unreliable, without forecast
    order = list(RELIABILITY)
    table.sort(key=lambda r: (order.index(r["classe"]), -((r["osservato"] or {}).get("n") or 0)))
    shown = [r for r in table if not classes or r["classe"] in classes]

    lists = {
        "microorganismo": _box([(p, name, sum(1 for c in combos if c["microorganismo"] == p and forecast_at(c, horizon, lab)),
                                 any(forecast_at(c, horizon, lab) for c in combos if c["microorganismo"] == p))
                                for p, name in pathogens], pathogen),
        "antibiotico": _box([(c["antibiotico"], c["antibiotico_nome"], tested(c), forecast_at(c, horizon, lab) is not None)
                             for c in sorted(of_pathogen, key=lambda c: -tested(c))], antibiotic),
        "orizzonte": _box([(YEAR, "Prossimi 12 mesi", None, True)]
                          + [(h, f"{label} ({h} mesi)", None, not lab)
                             for h, label in zip(HORIZONS, meta.get("trimestri_previsti", []))], horizon),
        "laboratorio": _box([(code, name, next((r["n"] for r in combo.get("laboratori", []) if r["codice"] == code), 0),
                              any(r["codice"] == code for r in combo.get("laboratori", [])))
                             for code, name in labs_known.items()], [lab] if lab else []),
        "affidabilita": _box([(k, name, sum(1 for r in table if r["classe"] == k), any(r["classe"] == k for r in table))
                              for k, name in RELIABILITY.items()], classes),
    }

    # The selected combination
    lab_rows = combo.get("previsioni_laboratori") or {}
    forecasts_by_lab = {r["codice"]: r for r in lab_rows.get("laboratori", [])}
    laboratories = [
        {"laboratorio": r["codice"], "nome": r["nome"], "testati": r["n"], "osservato": round(r["pct"], 1),
         "osservato_basso": round(r["low"], 1), "osservato_alto": round(r["high"], 1), "riportabile": r["riportabile"],
         "previsione": _value(forecasts_by_lab.get(r["codice"]), lab_rows.get("verifica"))}
        for r in combo.get("laboratori", [])]
    year = combo.get("prossimi_12_mesi")
    quarterly = [dict(_value(f), periodo=f["periodo"], trimestre=f["trimestre"], orizzonte=str(f["orizzonte_mesi"]),
                      perche=TRAFFIC_WHY[f["semaforo"]], prova=past_test_sentence(f) if f["errore_tipico"] is not None else "")
                 for f in combo.get("previsioni", [])]
    model = combo["previsioni"][0]["modello"] if combo.get("previsioni") else None
    lab_observed = next((r for r in laboratories if r["laboratorio"] == lab), None)
    detail = {
        "nomi": {"microorganismo": combo["microorganismo_nome"], "antibiotico": combo["antibiotico_nome"],
                 "laboratorio": labs_known.get(lab, "")},
        "ultimi_12_mesi": combo.get("ultimi_12_mesi"),
        "tendenza": combo.get("tendenza"), "tendenza_rete_costante": combo.get("tendenza_rete_costante"),
        "storico": combo.get("storico", []),
        "trimestrali": quarterly,
        "dodici_mesi": dict(_value(year, (year or {}).get("verifica")), perche=TRAFFIC_WHY[year["semaforo"]],
                            prova=check_sentence(year.get("verifica"), "la previsione annuale")) if year else None,
        "laboratorio": dict(lab_observed, prova=check_sentence(lab_rows.get("verifica"), "la previsione per laboratorio"))
                       if lab_observed else None,
        "selezionata": forecast_at(combo, horizon, lab),
        "modello": MODEL_WORDS.get(model, ""),
        "motivo_assenza": combo.get("motivo_assenza", ""),
        "laboratori": laboratories,
        "verifica_laboratori": check_sentence(lab_rows.get("verifica"), "la previsione per laboratorio"),
    }

    # Antibiogram expected at the horizon, every pathogen x antibiotic with a forecast
    cells = [dict(value, microorganismo=c["microorganismo"], antibiotic=c["antibiotico"])
             for c in combos if (value := forecast_at(c, horizon, lab)) and (not classes or _class(value) in classes)]
    columns = [p for p, _ in pathogens if any(cell["microorganismo"] == p for cell in cells)]
    names = {c["antibiotico"]: c["antibiotico_nome"] for c in combos}
    rows = sorted({cell["antibiotic"] for cell in cells}, key=lambda a: -sum(
        tested(by_key[(p, a)]) for p, _ in pathogens if (p, a) in by_key))
    horizon_name = next(i["nome"] for i in lists["orizzonte"] if i["valore"] == horizon)
    return {
        "vuoto": False,
        "selezione": {"microorganismo": pathogen, "antibiotico": antibiotic, "orizzonte": horizon,
                      "laboratorio": [lab] if lab else [], "affidabilita": classes},
        "liste": lists,
        "meta": {"dati_fino_al": meta.get("dati_fino_al"), "ultimo_trimestre": meta.get("ultimo_trimestre"),
                 "calcolato_il": meta.get("calcolato_il"), "trimestri_previsti": meta.get("trimestri_previsti", []),
                 "breakpoint": meta.get("breakpoint"), "da_ricalcolare": stale, "orizzonte": horizon_name},
        "tabella": shown,
        "dettaglio": detail,
        "antibiogramma": {"colonne": [{"valore": p, "nome": dict(pathogens)[p]} for p in columns],
                          "righe": [{"valore": a, "nome": names[a]} for a in rows], "celle": cells},
        "affidabilita": RELIABILITY,
    }
