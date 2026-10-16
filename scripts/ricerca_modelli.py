"""Model search on the archive: every candidate, three targets, three periods.

    python scripts/ricerca_modelli.py [cartella_archivio] [file.xlsx]

Defaults: instance/dataset and instance/previsioni/ricerca_modelli.xlsx. Uses the
invasive isolates and the harmonised reading, as the forecasts. The choice uses only
tests whose target falls before 2025 (scelta 2020-22, verifica 2023-24); the forecasts
made at the end of 2024 for 2025 (conferma) are reported beside, never used to choose.
The workbook holds aggregated figures only.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.model_search import run_search, write_workbook  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", nargs="?", type=Path, default=ROOT / "instance" / "dataset")
    parser.add_argument("output", nargs="?", type=Path, default=ROOT / "instance" / "previsioni" / "ricerca_modelli.xlsx")
    args = parser.parse_args()
    started = time.monotonic()
    result = run_search(args.dataset, progress=lambda message: print(message, flush=True))
    write_workbook(result, args.output)
    print(f"{result['configurazioni']} configurazioni, {len(result['prove']):,} prove, "
          f"{time.monotonic() - started:.0f} s -> {args.output}")
    for row in result["scelti"].itertuples():
        print(f"  {row.obiettivo:12} {row.modello}: verifica {row.errore_verifica:.2f} "
              f"(riferimento {row.errore_rif_verifica:.2f}), conferma 2025 {row.errore_conferma:.2f} "
              f"(riferimento {row.errore_rif_conferma:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
