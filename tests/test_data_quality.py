import pandas as pd
from src.data_prep import clean, load_raw, validate


def test_clean_data_passes_validation():
    checks = validate(clean(load_raw()))
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}


def test_blank_total_charges_are_filled():
    df = clean(load_raw())
    assert df["TotalCharges"].isna().sum() == 0
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()


def test_target_is_binary_int():
    df = clean(load_raw())
    assert set(df["Churn"].unique()) == {0, 1}
