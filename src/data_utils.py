#Utility functions for loading and checking the heart failure dataset.

import pandas as pd


def load_heart_failure_data(file_path: str = "data/heart_failure_clinical_records_dataset.csv") -> pd.DataFrame:
    return pd.read_csv(file_path)


def basic_data_report(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    if "DEATH_EVENT" in df.columns:
        print("\nTarget counts:")
        print(df["DEATH_EVENT"].value_counts())

        print("\nTarget proportions:")
        print(df["DEATH_EVENT"].value_counts(normalize=True))
