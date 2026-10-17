"""Import of a saved data file into the archive, shared by the upload page and the
dashboard refresh."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app import db
from app.models import UploadedFile
from app.services import dataset_store


@dataclass
class ImportOutcome:
    upload_id: int
    message: str
    report: dict
    replaced: list[int] = field(default_factory=list)


def import_file(saved_path: str | Path, original_name: str, dataset_folder: str | Path,
                replaces: list[int] | tuple[int, ...] = ()) -> ImportOutcome:
    """Validate and archive one Rete MIC export; `replaces` are imported uploads it supersedes.

    The superseded uploads are retired only once the new one is archived, so a failed
    import leaves them in place. Raises DatasetValidationError (report saved) or any
    other error (nothing kept).
    """
    uploaded = UploadedFile(filename=str(saved_path), original_name=original_name, status="uploaded")
    db.session.add(uploaded)
    db.session.commit()
    upload_id = uploaded.id
    try:
        raw = dataset_store.read_raw_file(saved_path)
        if not dataset_store.is_mic_export(raw.columns):
            raise ValueError("Il file non è un export della Rete MIC: mancano le colonne <ANTIBIOTICO>_QUALITATIVO. "
                             "Carica l'export degli isolati o usa \"Scarica dal cruscotto\".")
        isolates, ast, report, issues = dataset_store.build_and_validate(raw)
        dataset_store.save_dataset(dataset_folder, upload_id, isolates, ast, report, issues)
        uploaded.row_count = int(report["stats"]["isolates_valid"])
        uploaded.status = "imported"
        replaced = _retire(replaces, upload_id)
        uploaded.message = (
            f"{report['stats']['isolates_valid']} isolati validi su {report['stats']['rows_read']} righe, "
            f"{report['stats']['ast_results']} risultati di antibiogramma."
            + (f" Sostituiti {len(replaced)} file precedenti." if replaced else "")
        )
        db.session.commit()
    except dataset_store.DatasetValidationError as exc:
        db.session.rollback()
        dataset_store.save_report(dataset_folder, upload_id, exc.report)
        _mark_error(upload_id, f"Validazione fallita: {exc}")
        exc.upload_id = upload_id
        raise
    except Exception as exc:
        db.session.rollback()
        dataset_store.delete_upload_data(dataset_folder, upload_id)
        _mark_error(upload_id, str(exc))
        raise
    # Parquet partitions go only once the new archive is committed
    for old_id in replaced:
        dataset_store.delete_upload_data(dataset_folder, old_id)
    return ImportOutcome(upload_id, uploaded.message, report, replaced)


def _retire(upload_ids, new_id: int) -> list[int]:
    retired = []
    for old in UploadedFile.query.filter(UploadedFile.id.in_(list(upload_ids)),
                                         UploadedFile.status == "imported").all():
        old.status = "replaced"
        old.message = f"Sostituito dal caricamento n. {new_id}. {old.message}"
        retired.append(old.id)
    return retired


def _mark_error(upload_id: int, message: str) -> None:
    uploaded = db.session.get(UploadedFile, upload_id)
    uploaded.status = "error"
    uploaded.message = message
    db.session.commit()
