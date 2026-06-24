from __future__ import annotations

import argparse
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="micare")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Aggrega i dati Excel e addestra i modelli Prophet.")
    train_parser.add_argument("excel", nargs="+", help="Uno o piu file Excel RETEMIC.")
    train_parser.add_argument("--output-dir", default="outputs", help="Cartella per pickle e aggregati generati.")
    train_parser.add_argument("--n-jobs", type=int, default=-1, help="Numero job paralleli per joblib.")
    train_parser.add_argument(
        "--conflicts-file",
        default=None,
        help="Percorso Excel per eventuali conflitti duplicati non risolvibili automaticamente.",
    )

    predict_parser = subparsers.add_parser("predict", help="Legge i pickle generati e calcola una previsione.")
    predict_parser.add_argument("output_dir", help="Cartella che contiene i pickle di previsione.")
    predict_parser.add_argument("patogeno")
    predict_parser.add_argument("laboratorio")
    predict_parser.add_argument("antibiotico")
    predict_parser.add_argument("anno", type=int)
    predict_parser.add_argument("mese", type=int)

    return parser


def train_command(args: argparse.Namespace) -> int:
    from .data import load_and_aggregate
    from .model import train_forecasts

    output_dir = Path(args.output_dir)
    conflicts_file = args.conflicts_file
    if conflicts_file is None:
        conflicts_file = output_dir / "campioni_conflitto_da_revisionare_VERI.xlsx"

    aggregated, report = load_and_aggregate(args.excel, conflict_output_path=conflicts_file)
    summary = train_forecasts(aggregated, output_dir=output_dir, n_jobs=args.n_jobs)

    print("Aggregazione completata.")
    print(f"Righe input dopo pulizia temporale/antibiotici: {report.total_rows}")
    print(f"Righe pulite dopo consolidamento: {report.clean_rows}")
    print(f"Campioni duplicati: {report.duplicate_samples}")
    print(f"Campioni consolidati: {report.consolidated_samples}")
    print(f"Conflitti veri da revisionare: {report.true_conflict_samples}")
    print("Addestramento completato.")
    print(f"Combinazioni tentate: {summary.attempted_combinations}")
    print(f"Modelli resistenti addestrati: {summary.trained_resistant_models}")
    print(f"Output: {summary.output_dir}")
    return 0


def predict_command(args: argparse.Namespace) -> int:
    from .model import predict_percentages

    values = predict_percentages(
        args.output_dir,
        pathogen=args.patogeno,
        laboratory=args.laboratorio,
        antibiotic=args.antibiotico,
        year=args.anno,
        month=args.mese,
    )

    print(f"Previsione {args.patogeno}_{args.laboratorio}_{args.antibiotico} {args.anno}-{args.mese:02d}")
    print(f"Resistenti: {values['resistenti']:.2f}%")
    print(f"Intermedi: {values['intermedi']:.2f}%")
    print(f"Sensibili: {values['sensibili']:.2f}%")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return train_command(args)
    if args.command == "predict":
        return predict_command(args)

    parser.error(f"Comando sconosciuto: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
