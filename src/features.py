"""Feature engineering and the preprocessing ColumnTransformer."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import SERVICE_COLS, TARGET, ID_COL


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain features. Kept inside the Pipeline so train/predict match."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["tenure_group"] = pd.cut(
            X["tenure"], bins=[-1, 12, 24, 48, 72],
            labels=["0-12m", "13-24m", "25-48m", "49-72m"],
        ).astype(str)
        X["num_services"] = (X[SERVICE_COLS] == "Yes").sum(axis=1)
        X["avg_monthly_spend"] = X["TotalCharges"] / X["tenure"].clip(lower=1)
        X["has_security_support"] = (
            (X["OnlineSecurity"] == "Yes") | (X["TechSupport"] == "Yes")
        ).astype(int)
        return X


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET]
    return X, y


NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "num_services", "avg_monthly_spend", "has_security_support"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
        ]
    )
