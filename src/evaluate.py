"""
Evaluation helpers for classification models.

All models should be evaluated with the same function so the final comparison is
fair and easy to combine in the results notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_score_values(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    """
    Return probability-like scores for ROC-AUC when available.

    Preference order:
    1. predict_proba positive-class probability
    2. decision_function scores
    3. None if neither is available
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores).ravel()

    return None


def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: Iterable[int],
    model_name: str,
    feature_set: str,
) -> Dict[str, float | int | str]:
    """
    Evaluate a trained binary classifier.

    Parameters
    ----------
    model:
        A fitted scikit-learn style classifier.
    X_test:
        Test features.
    y_test:
        True labels.
    model_name:
        Name to display in the results table.
    feature_set:
        Usually "with_time" or "without_time".
    """
    y_true = np.asarray(y_test)
    y_pred = model.predict(X_test)
    score_values = get_score_values(model, X_test)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    if score_values is not None and len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, score_values)
    else:
        roc_auc = np.nan

    return {
        "model": model_name,
        "feature_set": feature_set,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_multiple_models(
    model_specs: List[Dict[str, Any]],
    y_test: Iterable[int],
) -> pd.DataFrame:
    """
    Evaluate several fitted models and return one table.

    Each item in model_specs should have:
    - model: fitted model object
    - X_test: test feature DataFrame
    - model_name: display name
    - feature_set: feature set label
    """
    results = []

    for spec in model_specs:
        results.append(
            evaluate_classifier(
                model=spec["model"],
                X_test=spec["X_test"],
                y_test=y_test,
                model_name=spec["model_name"],
                feature_set=spec["feature_set"],
            )
        )

    return pd.DataFrame(results)


def format_metrics_table(metrics_df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """
    Return a cleaner rounded metrics table for display.
    """
    formatted = metrics_df.copy()
    numeric_cols = formatted.select_dtypes(include=["float", "float64", "float32"]).columns
    formatted[numeric_cols] = formatted[numeric_cols].round(decimals)
    return formatted


def save_metrics(metrics_df: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Save a metrics table as a CSV file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(path, index=False)
    return path


def load_metrics_files(paths: Iterable[str | Path]) -> pd.DataFrame:
    """
    Load and combine multiple metric CSV files.
    """
    frames = []

    for path in paths:
        path = Path(path)
        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError("No metric files were found to combine.")

    return pd.concat(frames, ignore_index=True)


def rank_models(
    metrics_df: pd.DataFrame,
    primary_metric: str = "roc_auc",
    secondary_metric: str = "recall",
) -> pd.DataFrame:
    """
    Sort models by a primary metric and then a secondary metric.
    """
    if primary_metric not in metrics_df.columns:
        raise ValueError(f"Column '{primary_metric}' is not in the metrics table.")

    if secondary_metric not in metrics_df.columns:
        raise ValueError(f"Column '{secondary_metric}' is not in the metrics table.")

    return metrics_df.sort_values(
        by=[primary_metric, secondary_metric],
        ascending=[False, False],
    ).reset_index(drop=True)
