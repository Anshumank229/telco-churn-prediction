"""Score customers with the saved model.

Usage:
    python -m src.predict --input data/raw/telco_churn.csv --output reports/predictions.csv
    python -m src.predict --demo
"""
import argparse

import joblib
import pandas as pd

from .config import ID_COL, MODEL_PATH, TARGET
from .data_prep import clean

RISK_BANDS = [0, 0.3, 0.6, 1.0]
RISK_LABELS = ["Low", "Medium", "High"]


def score(df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    df = clean(df) if df["SeniorCitizen"].dtype != object else df
    X = df.drop(columns=[c for c in (TARGET, ID_COL) if c in df.columns])
    out = pd.DataFrame({ID_COL: df[ID_COL]}) if ID_COL in df.columns else pd.DataFrame(index=df.index)
    out["churn_probability"] = model.predict_proba(X)[:, 1].round(4)
    out["risk_band"] = pd.cut(out["churn_probability"], RISK_BANDS, labels=RISK_LABELS, include_lowest=True)
    return out.sort_values("churn_probability", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="CSV with the same columns as the training data")
    ap.add_argument("--output", default="reports/predictions.csv")
    ap.add_argument("--demo", action="store_true", help="score 5 sample customers and print")
    args = ap.parse_args()

    if args.demo:
        from .config import RAW_DATA
        df = pd.read_csv(RAW_DATA).sample(5, random_state=1)
        print(score(df).to_string(index=False))
    elif args.input:
        res = score(pd.read_csv(args.input))
        res.to_csv(args.output, index=False)
        print(f"Wrote {len(res)} predictions to {args.output}")
        print(res["risk_band"].value_counts().to_string())
    else:
        ap.error("provide --input or --demo")


if __name__ == "__main__":
    main()
