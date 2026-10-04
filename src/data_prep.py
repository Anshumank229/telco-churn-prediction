"""Loading, cleaning and validating the raw Telco churn data."""
import numpy as np
import pandas as pd

from .config import RAW_DATA, TARGET, ID_COL


def load_raw(path=RAW_DATA) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw data.

    Fixes applied:
    * TotalCharges is stored as text; blank strings (new customers with
      tenure == 0) are converted to NaN and then filled with 0, because
      they have not been billed yet.
    * Exact duplicate rows are dropped.
    * Target is mapped to 1 (churned) / 0 (stayed).
    * SeniorCitizen is converted from 0/1 to No/Yes for consistency.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
    new_customers = df["tenure"] == 0
    df.loc[new_customers & df["TotalCharges"].isna(), "TotalCharges"] = 0.0

    df = df.drop_duplicates()
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df.reset_index(drop=True)


def validate(df: pd.DataFrame) -> dict:
    """Run data-quality checks. Returns a dict of check name -> bool."""
    checks = {
        "no_missing_values": bool(df.isna().sum().sum() == 0),
        "unique_customer_ids": bool(df[ID_COL].is_unique),
        "target_is_binary": bool(set(df[TARGET].unique()) <= {0, 1}),
        "tenure_non_negative": bool((df["tenure"] >= 0).all()),
        "monthly_charges_positive": bool((df["MonthlyCharges"] > 0).all()),
        "total_charges_non_negative": bool((df["TotalCharges"] >= 0).all()),
        "total_charges_numeric": bool(pd.api.types.is_numeric_dtype(df["TotalCharges"])),
    }
    return checks


def get_clean_data() -> pd.DataFrame:
    df = clean(load_raw())
    failed = [k for k, ok in validate(df).items() if not ok]
    if failed:
        raise ValueError(f"Data validation failed: {failed}")
    return df
