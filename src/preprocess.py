"""
Preprocessing helpers for the heart failure final project.

The project compares models with and without the time feature. This file prepares
both versions of the dataset and keeps the train/test split consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_COL = "DEATH_EVENT"
RANDOM_STATE = 42
TEST_SIZE = 0.20

BINARY_COLS = [
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
]

CONTINUOUS_COLS_WITH_TIME = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time",
]

CONTINUOUS_COLS_WITHOUT_TIME = [
    col for col in CONTINUOUS_COLS_WITH_TIME if col != "time"
]

FEATURE_COLS_WITH_TIME = CONTINUOUS_COLS_WITH_TIME + BINARY_COLS
FEATURE_COLS_WITHOUT_TIME = CONTINUOUS_COLS_WITHOUT_TIME + BINARY_COLS


@dataclass
class PreprocessingOutput:
    """
    Container for all model-ready datasets.
    """

    X_train_lr_with_time: pd.DataFrame
    X_test_lr_with_time: pd.DataFrame
    X_train_lr_without_time: pd.DataFrame
    X_test_lr_without_time: pd.DataFrame
    X_train_tree_with_time: pd.DataFrame
    X_test_tree_with_time: pd.DataFrame
    X_train_tree_without_time: pd.DataFrame
    X_test_tree_without_time: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_idx: pd.Index
    test_idx: pd.Index
    scaler_with_time: StandardScaler
    scaler_without_time: StandardScaler


def clean_dataset(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Make a clean copy of the dataset.

    This removes exact duplicate rows and checks that the target exists. The heart
    failure dataset normally has no missing values, so this function raises an
    error if missing values are found instead of silently filling them in.
    """
    df_clean = df.copy().drop_duplicates().reset_index(drop=True)

    if target_col not in df_clean.columns:
        raise ValueError(f"Target column '{target_col}' was not found.")

    missing_total = int(df_clean.isna().sum().sum())
    if missing_total > 0:
        missing_by_column = df_clean.isna().sum()
        raise ValueError(
            "Missing values were found. Decide how to handle them before modeling:\n"
            f"{missing_by_column[missing_by_column > 0]}"
        )

    return df_clean


def validate_columns(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Check that all expected columns exist in the DataFrame.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")


def make_feature_target_sets(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Create X with time, X without time, and y.
    """
    all_needed = FEATURE_COLS_WITH_TIME + [target_col]
    validate_columns(df, all_needed)

    X_with_time = df[FEATURE_COLS_WITH_TIME].copy()
    X_without_time = df[FEATURE_COLS_WITHOUT_TIME].copy()
    y = df[target_col].copy()

    return X_with_time, X_without_time, y


def make_stratified_split_indices(
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.Index, pd.Index]:
    """
    Create one stratified train/test split using row indices.
    """
    train_idx, test_idx = train_test_split(
        y.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return pd.Index(train_idx), pd.Index(test_idx)


def split_by_indices(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: pd.Index,
    test_idx: pd.Index,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Apply an existing train/test split to X and y.
    """
    X_train = X.loc[train_idx].reset_index(drop=True)
    X_test = X.loc[test_idx].reset_index(drop=True)
    y_train = y.loc[train_idx].reset_index(drop=True)
    y_test = y.loc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def scale_continuous_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    continuous_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Standardize continuous columns using only the training data.

    Binary 0/1 columns are not changed.
    """
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

    return X_train_scaled, X_test_scaled, scaler


def preprocess_heart_failure_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> PreprocessingOutput:
    """
    Run the full preprocessing pipeline.

    Logistic regression uses scaled continuous features. Tree-based models use
    the unscaled feature values.
    """
    df_clean = clean_dataset(df, target_col=target_col)
    X_with_time, X_without_time, y = make_feature_target_sets(df_clean, target_col)

    train_idx, test_idx = make_stratified_split_indices(
        y, test_size=test_size, random_state=random_state
    )

    X_train_with_time, X_test_with_time, y_train, y_test = split_by_indices(
        X_with_time, y, train_idx, test_idx
    )
    X_train_without_time, X_test_without_time, _, _ = split_by_indices(
        X_without_time, y, train_idx, test_idx
    )

    X_train_lr_with_time, X_test_lr_with_time, scaler_with_time = scale_continuous_columns(
        X_train_with_time,
        X_test_with_time,
        CONTINUOUS_COLS_WITH_TIME,
    )

    (
        X_train_lr_without_time,
        X_test_lr_without_time,
        scaler_without_time,
    ) = scale_continuous_columns(
        X_train_without_time,
        X_test_without_time,
        CONTINUOUS_COLS_WITHOUT_TIME,
    )

    return PreprocessingOutput(
        X_train_lr_with_time=X_train_lr_with_time,
        X_test_lr_with_time=X_test_lr_with_time,
        X_train_lr_without_time=X_train_lr_without_time,
        X_test_lr_without_time=X_test_lr_without_time,
        X_train_tree_with_time=X_train_with_time.copy(),
        X_test_tree_with_time=X_test_with_time.copy(),
        X_train_tree_without_time=X_train_without_time.copy(),
        X_test_tree_without_time=X_test_without_time.copy(),
        y_train=y_train,
        y_test=y_test,
        train_idx=train_idx,
        test_idx=test_idx,
        scaler_with_time=scaler_with_time,
        scaler_without_time=scaler_without_time,
    )


def save_preprocessed_outputs(
    output: PreprocessingOutput,
    output_dir: str | Path = "results/tables",
) -> Dict[str, Path]:
    """
    Save all model-ready datasets as CSV files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    objects = {
        "X_train_lr_with_time.csv": output.X_train_lr_with_time,
        "X_test_lr_with_time.csv": output.X_test_lr_with_time,
        "X_train_lr_without_time.csv": output.X_train_lr_without_time,
        "X_test_lr_without_time.csv": output.X_test_lr_without_time,
        "X_train_tree_with_time.csv": output.X_train_tree_with_time,
        "X_test_tree_with_time.csv": output.X_test_tree_with_time,
        "X_train_tree_without_time.csv": output.X_train_tree_without_time,
        "X_test_tree_without_time.csv": output.X_test_tree_without_time,
        "y_train.csv": output.y_train.to_frame(name=TARGET_COL),
        "y_test.csv": output.y_test.to_frame(name=TARGET_COL),
    }

    saved_paths: Dict[str, Path] = {}
    for filename, obj in objects.items():
        path = out / filename
        obj.to_csv(path, index=False)
        saved_paths[filename] = path

    split_info = pd.DataFrame(
        {
            "original_index": list(output.train_idx) + list(output.test_idx),
            "split": ["train"] * len(output.train_idx) + ["test"] * len(output.test_idx),
        }
    )
    split_path = out / "train_test_split_indices.csv"
    split_info.to_csv(split_path, index=False)
    saved_paths["train_test_split_indices.csv"] = split_path

    return saved_paths


def load_preprocessed_outputs(
    tables_dir: str | Path = "results/tables",
    target_col: str = TARGET_COL,
) -> Dict[str, pd.DataFrame | pd.Series]:
    """
    Load the saved preprocessing outputs from CSV files.
    """
    folder = Path(tables_dir)

    def read_frame(name: str) -> pd.DataFrame:
        path = folder / name
        if not path.exists():
            raise FileNotFoundError(f"Missing preprocessing file: {path}")
        return pd.read_csv(path)

    data: Dict[str, pd.DataFrame | pd.Series] = {
        "X_train_lr_with_time": read_frame("X_train_lr_with_time.csv"),
        "X_test_lr_with_time": read_frame("X_test_lr_with_time.csv"),
        "X_train_lr_without_time": read_frame("X_train_lr_without_time.csv"),
        "X_test_lr_without_time": read_frame("X_test_lr_without_time.csv"),
        "X_train_tree_with_time": read_frame("X_train_tree_with_time.csv"),
        "X_test_tree_with_time": read_frame("X_test_tree_with_time.csv"),
        "X_train_tree_without_time": read_frame("X_train_tree_without_time.csv"),
        "X_test_tree_without_time": read_frame("X_test_tree_without_time.csv"),
        "y_train": read_frame("y_train.csv")[target_col],
        "y_test": read_frame("y_test.csv")[target_col],
    }

    return data
