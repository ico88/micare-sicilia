from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error


def regression_metrics(y_true, y_pred) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"mae": None, "rmse": None, "mape": None}

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    non_zero = y_true != 0
    mape = None
    if non_zero.any():
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def sir_classification_metrics(y_true_pct: np.ndarray, y_pred_pct: np.ndarray) -> dict[str, float | None]:
    if len(y_true_pct) == 0:
        return {"accuracy": None, "f1_macro": None}
    y_true = np.argmax(y_true_pct, axis=1)
    y_pred = np.argmax(y_pred_pct, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def reliability_level(
    samples: int,
    model_rmse: float | None,
    historical_std: float,
    fallback_scope: str,
) -> tuple[str, str]:
    reasons = []
    score = 0

    if samples >= 50:
        score += 2
    elif samples >= 15:
        score += 1
    else:
        reasons.append("numerosita campionaria bassa")

    if model_rmse is not None and model_rmse <= 15:
        score += 2
    elif model_rmse is not None and model_rmse <= 25:
        score += 1
    else:
        reasons.append("errore del modello elevato o non disponibile")

    if historical_std <= 15:
        score += 1
    else:
        reasons.append("serie storica instabile")

    if fallback_scope != "exact":
        reasons.append(f"fallback usato: {fallback_scope}")

    if score >= 4 and fallback_scope == "exact":
        return "alta", "serie sufficiente e modello stabile"
    if score >= 2:
        return "media", "; ".join(reasons) or "affidabilita intermedia"
    return "bassa", "; ".join(reasons) or "dati insufficienti"
