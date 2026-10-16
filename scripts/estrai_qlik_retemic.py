"""Download the isolate-level Rete MIC records from the public regional dashboard.

    python scripts/estrai_qlik_retemic.py [file.csv] [--url <dashboard link>]

Writes a ';' CSV in the layout of the network export (one row per isolate,
<ABX>_QUALITATIVO / <ABX>_QUANTITATIVO columns) that the upload page accepts
as is. Default output: data/cruscotto_retemic_<date>.csv. The file holds
patient-level records: keep it on the server, never commit it.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.qlik_source import DASHBOARD_URL, QlikSourceError, extract_to_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", type=Path,
                        default=ROOT / "data" / f"cruscotto_retemic_{date.today():%Y%m%d}.csv")
    parser.add_argument("--url", default=DASHBOARD_URL, help="link to the dashboard sheet")
    args = parser.parse_args()
    try:
        stats = extract_to_csv(args.url, args.output,
                               progress=lambda message, fraction: print(f"[{fraction:4.0%}] {message}", flush=True))
    except QlikSourceError as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        return 1
    print(f"{stats['rows']} righe, {stats['columns']} colonne in {stats['seconds']} s -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
