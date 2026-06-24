from __future__ import annotations

import json

import pandas as pd

from app import db
from app.models import (
    AggregatedObservation,
    Observation,
    Prediction,
    TrainedModel,
    UploadedFile,
    ValidationMetric,
)


def observations_to_db(df: pd.DataFrame, uploaded_file: UploadedFile) -> None:
    rows = []
    for item in df.to_dict(orient="records"):
        rows.append(
            Observation(
                uploaded_file_id=uploaded_file.id,
                observed_at=pd.to_datetime(item["date"]).date(),
                pathogen=item["pathogen"],
                antibiotic=item["antibiotic"],
                laboratory=item["laboratory"],
                ward=item.get("ward") or "",
                result=item.get("result"),
                samples=int(item.get("samples") or 1),
                sensitive_count=float(item.get("sensitive_count") or 0),
                intermediate_count=float(item.get("intermediate_count") or 0),
                resistant_count=float(item.get("resistant_count") or 0),
            )
        )
    db.session.bulk_save_objects(rows)


def aggregated_to_db(df: pd.DataFrame) -> None:
    AggregatedObservation.query.delete()
    rows = []
    for item in df.to_dict(orient="records"):
        rows.append(
            AggregatedObservation(
                month=pd.to_datetime(item["month"]).date(),
                pathogen=item["pathogen"],
                antibiotic=item["antibiotic"],
                laboratory=item["laboratory"],
                ward=item.get("ward") or "",
                samples=int(item.get("samples") or 0),
                sensitive_count=float(item.get("sensitive_count") or 0),
                intermediate_count=float(item.get("intermediate_count") or 0),
                resistant_count=float(item.get("resistant_count") or 0),
                sensitive_pct=float(item.get("sensitive_pct") or 0),
                intermediate_pct=float(item.get("intermediate_pct") or 0),
                resistant_pct=float(item.get("resistant_pct") or 0),
            )
        )
    db.session.bulk_save_objects(rows)


def aggregated_from_db() -> pd.DataFrame:
    rows = AggregatedObservation.query.order_by(AggregatedObservation.month.asc()).all()
    return pd.DataFrame(
        [
            {
                "month": row.month,
                "pathogen": row.pathogen,
                "antibiotic": row.antibiotic,
                "laboratory": row.laboratory,
                "ward": row.ward or "",
                "samples": row.samples,
                "sensitive_count": row.sensitive_count,
                "intermediate_count": row.intermediate_count,
                "resistant_count": row.resistant_count,
                "sensitive_pct": row.sensitive_pct,
                "intermediate_pct": row.intermediate_pct,
                "resistant_pct": row.resistant_pct,
            }
            for row in rows
        ]
    )


def save_training_summary(summary: dict) -> None:
    TrainedModel.query.delete()
    ValidationMetric.query.delete()

    model_rows = []
    for artifact in summary["artifacts"]:
        for target in ["sensitive_pct", "intermediate_pct", "resistant_pct"]:
            model_rows.append(
                TrainedModel(
                    model_name=artifact["model_name"],
                    target=target,
                    artifact_path=artifact["artifact_path"],
                    scope="regional",
                    metadata_json=json.dumps(artifact.get("metadata", {})),
                )
            )
    db.session.bulk_save_objects(model_rows)
    db.session.flush()

    metric_rows = []
    for metric in summary["metrics"]:
        metric_rows.append(
            ValidationMetric(
                model_name=metric["model_name"],
                target=metric["target"],
                mae=metric.get("mae"),
                rmse=metric.get("rmse"),
                mape=metric.get("mape"),
                accuracy=metric.get("accuracy"),
                f1_macro=metric.get("f1_macro"),
                metadata_json=json.dumps({}),
            )
        )
    db.session.bulk_save_objects(metric_rows)


def save_prediction(prediction: dict) -> Prediction:
    row = Prediction(
        prediction_month=prediction["prediction_month"],
        pathogen=prediction["pathogen"],
        antibiotic=prediction["antibiotic"],
        laboratory=prediction["laboratory"],
        ward=prediction.get("ward") or "",
        model_name=prediction["model_name"],
        sensitive_pct=prediction["sensitive_pct"],
        intermediate_pct=prediction["intermediate_pct"],
        resistant_pct=prediction["resistant_pct"],
        reliability=prediction["reliability"],
        reliability_reason=prediction["reliability_reason"],
    )
    db.session.add(row)
    return row
