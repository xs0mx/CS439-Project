"""
Training helpers for the heart failure final project.

These functions are optional for the notebooks, but they keep model training more
organized and reusable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

try:
    from .evaluate import evaluate_classifier, save_metrics
except ImportError:
    from evaluate import evaluate_classifier, save_metrics


RANDOM_STATE = 42


def build_logistic_regression(random_state: int = RANDOM_STATE) -> LogisticRegression:
    """
    Create a logistic regression model.

    class_weight='balanced' helps account for the class imbalance in the target.
    """
    return LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
        solver="liblinear",
    )


def build_random_forest(random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    """
    Create a random forest classifier.
    """
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        max_depth=None,
    )


def build_decision_tree(random_state: int = RANDOM_STATE) -> DecisionTreeClassifier:
    """
    Create a decision tree classifier.
    """
    return DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced",
        max_depth=4,
    )


def build_gradient_boosting(random_state: int = RANDOM_STATE) -> GradientBoostingClassifier:
    """
    Create a gradient boosting classifier.
    """
    return GradientBoostingClassifier(
        random_state=random_state,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
    )


def fit_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Fit a model and return it.
    """
    model.fit(X_train, y_train)
    return model


def fit_and_evaluate(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    feature_set: str,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Fit one model and evaluate it on the test set.
    """
    fitted_model = fit_model(model, X_train, y_train)
    metrics = evaluate_classifier(
        model=fitted_model,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
        feature_set=feature_set,
    )
    return fitted_model, metrics


def save_model(model: Any, output_path: str | Path) -> Path:
    """
    Save a fitted model as a .joblib file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(model_path: str | Path) -> Any:
    """
    Load a saved .joblib model.
    """
    return joblib.load(model_path)


def train_all_baseline_models(
    data: Dict[str, Any],
    metrics_output_path: str | Path | None = None,
    random_state: int = RANDOM_STATE,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train the four planned baseline model families with and without time.

    Expected keys in data:
    - X_train_lr_with_time, X_test_lr_with_time
    - X_train_lr_without_time, X_test_lr_without_time
    - X_train_tree_with_time, X_test_tree_with_time
    - X_train_tree_without_time, X_test_tree_without_time
    - y_train, y_test
    """
    y_train = data["y_train"]
    y_test = data["y_test"]

    model_runs = [
        (
            "logistic_regression_with_time",
            build_logistic_regression(random_state),
            data["X_train_lr_with_time"],
            data["X_test_lr_with_time"],
            "Logistic Regression",
            "with_time",
        ),
        (
            "logistic_regression_without_time",
            build_logistic_regression(random_state),
            data["X_train_lr_without_time"],
            data["X_test_lr_without_time"],
            "Logistic Regression",
            "without_time",
        ),
        (
            "decision_tree_with_time",
            build_decision_tree(random_state),
            data["X_train_tree_with_time"],
            data["X_test_tree_with_time"],
            "Decision Tree",
            "with_time",
        ),
        (
            "decision_tree_without_time",
            build_decision_tree(random_state),
            data["X_train_tree_without_time"],
            data["X_test_tree_without_time"],
            "Decision Tree",
            "without_time",
        ),
        (
            "random_forest_with_time",
            build_random_forest(random_state),
            data["X_train_tree_with_time"],
            data["X_test_tree_with_time"],
            "Random Forest",
            "with_time",
        ),
        (
            "random_forest_without_time",
            build_random_forest(random_state),
            data["X_train_tree_without_time"],
            data["X_test_tree_without_time"],
            "Random Forest",
            "without_time",
        ),
        (
            "gradient_boosting_with_time",
            build_gradient_boosting(random_state),
            data["X_train_tree_with_time"],
            data["X_test_tree_with_time"],
            "Gradient Boosting",
            "with_time",
        ),
        (
            "gradient_boosting_without_time",
            build_gradient_boosting(random_state),
            data["X_train_tree_without_time"],
            data["X_test_tree_without_time"],
            "Gradient Boosting",
            "without_time",
        ),
    ]

    fitted_models: Dict[str, Any] = {}
    metric_rows = []

    for key, model, X_train, X_test, model_name, feature_set in model_runs:
        fitted_model, metrics = fit_and_evaluate(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            feature_set=feature_set,
        )
        fitted_models[key] = fitted_model
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows)

    if metrics_output_path is not None:
        save_metrics(metrics_df, metrics_output_path)

    return fitted_models, metrics_df
