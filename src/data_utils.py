"""
Utility functions for the heart failure final project.

This file keeps the basic dataset loading and checking code in one place so the
notebooks stay cleaner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


TARGET_COL = "DEATH_EVENT"
DEFAULT_DATASET_NAME = "heart_failure_clinical_records_dataset.csv"


def get_project_root(start: str | Path | None = None) -> Path:
    """
    Return the project root folder.

    If the current folder is named notebooks, src, paper, or results, the parent
    folder is treated as the project root. Otherwise, the current folder itself
    is used.
    """
    current = Path(start).resolve() if start is not None else Path.cwd().resolve()

    if current.is_file():
        current = current.parent

    if current.name in {"notebooks", "src", "paper", "results"}:
        return current.parent

    return current


def find_dataset(
    project_root: str | Path | None = None,
    filename: str = DEFAULT_DATASET_NAME,
    extra_paths: Iterable[str | Path] | None = None,
) -> Path:
    """
    Find the dataset CSV in common project locations.

    The main expected location is data/heart_failure_clinical_records_dataset.csv.
    This function also checks a few backup locations so the notebooks still work
    if the file was downloaded with a number in the name.
    """
    root = get_project_root(project_root)

    candidates = [
        root / "data" / filename,
        root / "data" / "heart_failure_clinical_records_dataset(4).csv",
        root / filename,
        root / "heart_failure_clinical_records_dataset(4).csv",
        Path.cwd() / filename,
        Path.cwd() / "heart_failure_clinical_records_dataset(4).csv",
        Path("/mnt/data") / filename,
        Path("/mnt/data") / "heart_failure_clinical_records_dataset(4).csv",
    ]

    if extra_paths is not None:
        candidates.extend(Path(p) for p in extra_paths)

    for path in candidates:
        if path.exists():
            return path.resolve()

    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find the heart failure dataset. Checked these locations:\n"
        f"{checked}"
    )


def load_heart_failure_data(file_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the heart failure dataset as a pandas DataFrame.

    Parameters
    ----------
    file_path:
        Optional path to the CSV file. If not provided, the function searches for
        the file in the project data folder and common backup locations.
    """
    path = Path(file_path).resolve() if file_path is not None else find_dataset()
    return pd.read_csv(path)


def basic_data_report(df: pd.DataFrame, target_col: str = TARGET_COL) -> None:
    """
    Print a simple data report for quick checking in a notebook.
    """
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    if target_col in df.columns:
        print("\nTarget counts:")
        print(df[target_col].value_counts().sort_index())

        print("\nTarget proportions:")
        print(df[target_col].value_counts(normalize=True).sort_index())


def get_data_summary(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Return a small summary table for the dataset.
    """
    rows = [
        {"item": "rows", "value": df.shape[0]},
        {"item": "columns", "value": df.shape[1]},
        {"item": "duplicate_rows", "value": int(df.duplicated().sum())},
        {"item": "missing_values_total", "value": int(df.isna().sum().sum())},
    ]

    if target_col in df.columns:
        counts = df[target_col].value_counts().sort_index()
        for label, count in counts.items():
            rows.append({"item": f"{target_col}_{label}_count", "value": int(count)})

    return pd.DataFrame(rows)


def ensure_directory(path: str | Path) -> Path:
    """
    Create a folder if it does not already exist and return the Path object.
    """
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder
