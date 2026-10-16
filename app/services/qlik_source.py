"""Isolate-level Rete MIC records read from the public regional dashboard.

The dashboard (Qualità Sicilia SSR, Qlik Sense, anonymous access) loads the raw
network export into the table ``Dati_Grezzi``: one row per isolate, the same
``<ABX>_QUALITATIVO`` / ``<ABX>_QUANTITATIVO`` columns as the Excel export. The
table is read through the Qlik Engine API (JSON-RPC over a websocket) with the
same anonymous session a browser gets, and written as a CSV that the normal
upload accepts unchanged.

No browser is needed: the session cookie comes from the dashboard page and the
websocket only asks for the CSRF token served by ``/qps/csrftoken``.
"""
from __future__ import annotations

import csv
import http.cookiejar
import json
import os
import re
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote, urlparse

DASHBOARD_URL = (
    "https://qlik.qualitasiciliassr.it/anonimo/single/"
    "?appid=85ada16c-4b41-4bc6-9ca1-405b8243d0c2"
    "&sheet=6ad6f3ac-3369-41c5-bd72-792243f9091b&opt=ctxmenu,currsel"
)
RAW_TABLE = "Dati_Grezzi"
KEY_FIELD = "KEY_CAMPIONE"
# Fields the dashboard adds on load (keys, checks, derived dates): not part of the export
DERIVED_FIELDS = {
    KEY_FIELD, "DATA_CONTABILE_ANNO", "ETA_PZ", "CAMPIONE_UNICO", "CHK_NULLI",
    "ANNO_PRELIEVO", "MESE_PRELIEVO", "CHK_ANNO",
}
# The dashboard shows these as "<code> - <label>" ("3 - Urine", "$ - NA"); the export has the code
CODE_LABEL_FIELDS = {"OSPEDALE", "PAZIENTE_RICOVERATO", "REPARTO_DI_RICOVERO", "MATERIALE"}
_CODE_LABEL_RE = re.compile(r"^([^\s]*)\s+-(?:\s.*)?$", re.DOTALL)
MAX_CELLS = 10_000      # Engine API limit per data page
FIELDS_PER_CUBE = 24    # columns read together with the key
IN_FLIGHT = 8           # pages requested before reading the answers
TIMEOUT = 120

Progress = Callable[[str, float], None]


class QlikSourceError(RuntimeError):
    pass


def parse_dashboard_url(url: str) -> tuple[str, str, str]:
    """(origin, virtual proxy prefix, app id) from a ``.../<prefix>/single/?appid=...`` link."""
    parts = urlparse(url)
    app_id = parse_qs(parts.query).get("appid", [""])[0]
    segments = [s for s in parts.path.split("/") if s]
    if not app_id or "single" not in segments:
        raise QlikSourceError(f"Indirizzo del cruscotto non riconosciuto: {url}")
    prefix = "/".join(segments[: segments.index("single")])
    return f"{parts.scheme}://{parts.netloc}", f"/{prefix}" if prefix else "", app_id


class QlikEngine:
    """Minimal synchronous JSON-RPC client for the Qlik Engine API."""

    def __init__(self, dashboard_url: str, timeout: int = TIMEOUT):
        import websocket  # websocket-client

        origin, prefix, self.app_id = parse_dashboard_url(dashboard_url)
        jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(jar), urllib.request.HTTPSHandler(context=_ssl_context()))
        try:
            opener.open(dashboard_url, timeout=timeout).read()
            with opener.open(f"{origin}{prefix}/qps/csrftoken", timeout=timeout) as response:
                token = response.headers.get("qlik-csrf-token", "")
        except OSError as exc:
            raise QlikSourceError(f"Cruscotto non raggiungibile: {exc}") from exc
        cookie = "; ".join(f"{c.name}={c.value}" for c in jar)
        host = urlparse(origin).netloc
        ws_url = f"wss://{host}{prefix}/app/{self.app_id}"
        if token:
            ws_url += f"?qlik-csrf-token={quote(token)}"
        try:
            self._ws = websocket.create_connection(
                ws_url, cookie=cookie, origin=origin, timeout=timeout,
                sslopt={"context": _ssl_context()}, **_proxy_options())
        except Exception as exc:  # websocket-client raises its own hierarchy
            raise QlikSourceError(f"Collegamento al motore Qlik rifiutato: {exc}") from exc
        self._next_id = 0

    def close(self) -> None:
        self._ws.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def send(self, method: str, handle: int, params) -> int:
        self._next_id += 1
        self._ws.send(json.dumps({"jsonrpc": "2.0", "id": self._next_id, "method": method,
                                  "handle": handle, "params": params}))
        return self._next_id

    def receive(self, ids: list[int]) -> dict[int, dict]:
        """Results of the given requests; notifications in between are skipped."""
        wanted, results = set(ids), {}
        while wanted:
            message = json.loads(self._ws.recv())
            if message.get("id") in wanted:
                if "error" in message:
                    raise QlikSourceError(f"Errore del motore Qlik: {message['error'].get('message')}")
                results[message["id"]] = message["result"]
                wanted.discard(message["id"])
        return results

    def call(self, method: str, handle: int, params):
        request_id = self.send(method, handle, params)
        return self.receive([request_id])[request_id]


def open_doc(engine) -> int:
    handle = engine.call("OpenDoc", -1, [engine.app_id])["qReturn"]["qHandle"]
    engine.call("ClearAll", handle, [True])  # selections would filter the rows read
    return handle


def table_layout(engine, doc: int, table: str = RAW_TABLE) -> tuple[list[str], int]:
    """Field names (load order) and row count of a table of the data model."""
    result = engine.call("GetTablesAndKeys", doc, [{"qcx": 1000, "qcy": 1000}, {"qcx": 0, "qcy": 0}, 30, True, False])
    for entry in result["qtr"]:
        if entry["qName"] == table:
            return [f["qName"] for f in entry["qFields"]], int(entry["qNoOfRows"])
    raise QlikSourceError(f"Tabella {table} non trovata nel cruscotto: struttura cambiata?")


def read_columns(engine, doc: int, fields: list[str], key: str = KEY_FIELD) -> tuple[list[str], dict[str, list[str]]]:
    """(keys, {field: values}) for fields of one table, read with the key as first column.

    The key is unique per row, so the hypercube has one row per record, sorted by
    the key: every call returns the records in the same order. Null cells are
    written as "", rows without a key (combinations from other tables) are skipped.
    """
    dims = [{"qDef": {"qFieldDefs": [f], "qSortCriterias": [{"qSortByAscii": 1}]}, "qNullSuppression": False}
            for f in [key, *fields]]
    object_id = f"mic-export-{fields[0]}"
    obj = engine.call("CreateSessionObject", doc, [{"qInfo": {"qId": object_id, "qType": "mic-export"}, "qHyperCubeDef": {
        "qDimensions": dims, "qMeasures": [], "qSuppressZero": False, "qSuppressMissing": False, "qMode": "S",
        "qInterColumnSortOrder": list(range(len(dims))),
    }}])["qReturn"]["qHandle"]
    keys: list[str] = []
    columns: dict[str, list[str]] = {f: [] for f in fields}
    intern: dict[str, str] = {}
    try:
        n_rows = engine.call("GetLayout", obj, [])["qLayout"]["qHyperCube"]["qSize"]["qcy"]
        width = len(dims)
        height = MAX_CELLS // width
        tops = list(range(0, n_rows, height))
        for start in range(0, len(tops), IN_FLIGHT):
            ids = [engine.send("GetHyperCubeData", obj, ["/qHyperCubeDef",
                                                         [{"qTop": top, "qLeft": 0, "qWidth": width, "qHeight": height}]])
                   for top in tops[start:start + IN_FLIGHT]]
            results = engine.receive(ids)
            for request_id in ids:  # page order, not arrival order
                for row in results[request_id]["qDataPages"][0]["qMatrix"]:
                    if row[0].get("qIsNull") or not row[0].get("qText"):
                        continue
                    keys.append(row[0]["qText"])
                    for field, cell in zip(fields, row[1:]):
                        text = "" if cell.get("qIsNull") else (cell.get("qText") or "")
                        columns[field].append(intern.setdefault(text, text))
    finally:
        engine.call("DestroySessionObject", doc, [object_id])
    return keys, columns


def export_value(field: str, text: str) -> str:
    """Cell text as the network export has it: codes without the label the dashboard adds."""
    text = text.strip()
    if field in CODE_LABEL_FIELDS:
        code = _CODE_LABEL_RE.match(text)
        if code:
            text = code.group(1).strip()
        if text in {"$", "-"}:
            return ""
    return text


def extract_to_csv(dashboard_url: str, out_path: str | Path, progress: Progress | None = None,
                   engine_factory=QlikEngine) -> dict:
    """Write the dashboard's raw table as a ';' CSV in the Rete MIC export layout."""
    progress = progress or (lambda message, fraction: None)
    started = time.monotonic()
    progress("Collegamento al cruscotto", 0.0)
    with engine_factory(dashboard_url) as engine:
        doc = open_doc(engine)
        fields, n_rows = table_layout(engine, doc)
        export_fields = [f for f in fields if f not in DERIVED_FIELDS]
        if KEY_FIELD not in fields or not any(f.endswith("_QUALITATIVO") for f in export_fields):
            raise QlikSourceError("La tabella del cruscotto non ha il formato dell'export Rete MIC.")
        chunks = [export_fields[i:i + FIELDS_PER_CUBE] for i in range(0, len(export_fields), FIELDS_PER_CUBE)]
        keys: list[str] | None = None
        columns: dict[str, list[str]] = {}
        for number, chunk in enumerate(chunks, start=1):
            progress(f"Lettura dal cruscotto: blocco {number} di {len(chunks)}", (number - 1) / len(chunks))
            chunk_keys, chunk_columns = read_columns(engine, doc, chunk)
            if keys is not None and chunk_keys != keys:
                raise QlikSourceError("Righe non allineate tra i blocchi letti: il cruscotto è cambiato durante "
                                      "l'estrazione? Riprova.")
            keys = chunk_keys
            columns.update(chunk_columns)

    if len(keys) != n_rows:
        raise QlikSourceError(f"Lette {len(keys)} righe su {n_rows}: estrazione incompleta, riprova.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part")
    with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(export_fields)
        for i in range(len(keys)):
            writer.writerow([export_value(f, columns[f][i]) for f in export_fields])
    tmp.replace(out_path)
    progress("Estrazione completata", 1.0)
    return {"rows": len(keys), "columns": len(export_fields), "seconds": round(time.monotonic() - started, 1)}


def _ssl_context() -> ssl.SSLContext:
    bundle = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    return ssl.create_default_context(cafile=bundle if bundle and Path(bundle).exists() else None)


def _proxy_options() -> dict:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if not proxy:
        return {}
    parts = urlparse(proxy)
    options = {"http_proxy_host": parts.hostname, "http_proxy_port": parts.port or 8080, "proxy_type": "http"}
    if parts.username:
        options["http_proxy_auth"] = (parts.username, parts.password or "")
    return options
