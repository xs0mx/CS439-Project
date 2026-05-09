"""
Plotting helpers for the heart failure final project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix
from sklearn.inspection import permutation_importance


def _prepare_output_path(output_path: str | Path | None) -> Path | None:
    if output_path is None:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_current_figure(output_path: str | Path | None) -> Path | None:
    """
    Save the current matplotlib figure if an output path is provided.
    """
    path = _prepare_output_path(output_path)
    if path is not None:
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
    return path


def plot_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: Iterable[int],
    title: str,
    output_path: str | Path | None = None,
) -> Path | None:
    """
    Plot and optionally save a confusion matrix.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Death", "Death"],
    )
    display.plot(ax=ax, values_format="d")
    ax.set_title(title)

    path = save_current_figure(output_path)
    plt.close(fig)
    return path


def plot_roc_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: Iterable[int],
    title: str,
    output_path: str | Path | None = None,
) -> Path | None:
    """
    Plot and optionally save a ROC curve.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(title)

    path = save_current_figure(output_path)
    plt.close(fig)
    return path


def plot_feature_importance(
    feature_names: Iterable[str],
    importances: Iterable[float],
    title: str,
    output_path: str | Path | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Plot feature importances and return the sorted importance table.
    """
    importance_df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": list(importances),
        }
    ).sort_values("importance", ascending=False)

    plot_df = importance_df.head(top_n) if top_n is not None else importance_df
    plot_df = plot_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(title)

    save_current_figure(output_path)
    plt.close(fig)

    return importance_df.reset_index(drop=True)


def plot_logistic_coefficients(
    model: Any,
    feature_names: Iterable[str],
    title: str,
    output_path: str | Path | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Plot logistic regression coefficients and return the sorted coefficient table.
    """
    if not hasattr(model, "coef_"):
        raise ValueError("The model does not have coef_. Make sure it is fitted.")

    coefficients = np.asarray(model.coef_).ravel()
    coef_df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)

    plot_df = coef_df.head(top_n) if top_n is not None else coef_df
    plot_df = plot_df.sort_values("coefficient", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df["coefficient"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Coefficient")
    ax.set_title(title)

    save_current_figure(output_path)
    plt.close(fig)

    return coef_df.reset_index(drop=True)


def compute_permutation_importance_table(
    model: Any,
    X_test: pd.DataFrame,
    y_test: Iterable[int],
    scoring: str = "roc_auc",
    n_repeats: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance and return a sorted table.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return importance_df.reset_index(drop=True)


def plot_permutation_importance(
    importance_df: pd.DataFrame,
    title: str,
    output_path: str | Path | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Plot a permutation importance table that was already computed.
    """
    plot_df = importance_df.head(top_n) if top_n is not None else importance_df
    plot_df = plot_df.sort_values("importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df["importance_mean"])
    ax.set_xlabel("Mean decrease in score")
    ax.set_title(title)

    save_current_figure(output_path)
    plt.close(fig)

    return importance_df
