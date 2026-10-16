"""Import of a saved data file into the archive, shared by the upload page and the
dashboard refresh."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app import db
from app.models import Observation, UploadedFile
from app.services import dataset_store
from app.services.database import aggregated_to_db, observation_month_totals_from_db, observations_to_db
from app.services.preprocessing import aggregate_observations, load_tabular_file, normalize_uploaded_dataframe


@dataclass
class ImportOutcome:
    upload_id: int
    message: str
    report: dict | None
    columns: list[str] = field(default_factory=list)
    normalized: pd.DataFrame | None = None
    aggregated: pd.DataFrame | None = None
    replaced: list[int] = field(default_factory=list)


def import_file(saved_path: str | Path, original_name: str, dataset_folder: str | Path,
                replaces: list[int] | tuple[int, ...] = ()) -> ImportOutcome:
    """Validate and archive one file; `replaces` are imported uploads it supersedes.

    The superseded uploads leave the archive (isolates, observations) in the same
    transaction, so a failed import leaves them in place. Raises
    DatasetValidationError (report saved) or any other error (nothing kept).
    """
    uploaded = UploadedFile(filename=str(saved_path), original_name=original_name, status="uploaded")
    db.session.add(uploaded)
    db.session.commit()
    upload_id = uploaded.id
    report = None
    try:
        raw = dataset_store.read_raw_file(saved_path)
        if dataset_store.is_mic_export(raw.columns):
            # Isolate-level export: validate and keep the full record for any model
            isolates, ast, report, issues = dataset_store.build_and_validate(raw)
            dataset_store.save_dataset(dataset_folder, upload_id, isolates, ast, report, issues)
            normalized = dataset_store.to_observations(isolates, ast)
        else:
            raw = load_tabular_file(saved_path)
            normalized = normalize_uploaded_dataframe(raw)
        uploaded.row_count = len(normalized)
        uploaded.status = "imported"

        observations_to_db(normalized, uploaded)
        replaced = _retire(replaces, upload_id)
        db.session.flush()
        aggregated = aggregate_observations(observation_month_totals_from_db())
        validated = (
            f"{report['stats']['isolates_valid']} isolati validi su {report['stats']['rows_read']} righe, "
            if report else ""
        )
        uploaded.message = (
            f"{validated}Importate {len(normalized)} nuove osservazioni. "
            f"Storico aggregato: {len(aggregated)} righe mensili."
            + (f" Sostituiti {len(replaced)} file precedenti." if replaced else "")
        )
        aggregated_to_db(aggregated)
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
    return ImportOutcome(upload_id, uploaded.message, report, list(raw.columns), normalized, aggregated, replaced)


def _retire(upload_ids, new_id: int) -> list[int]:
    retired = []
    for old in UploadedFile.query.filter(UploadedFile.id.in_(list(upload_ids)),
                                         UploadedFile.status == "imported").all():
        Observation.query.filter_by(uploaded_file_id=old.id).delete(synchronize_session=False)
        old.status = "replaced"
        old.message = f"Sostituito dal caricamento n. {new_id}. {old.message}"
        retired.append(old.id)
    return retired


def _mark_error(upload_id: int, message: str) -> None:
    uploaded = db.session.get(UploadedFile, upload_id)
    uploaded.status = "error"
    uploaded.message = message
    db.session.commit()
