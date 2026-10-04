"""Train, compare and save churn models.  Run:  python -m src.train"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from . import config as C
from .data_prep import get_clean_data
from .features import FeatureEngineer, build_preprocessor, split_xy


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("fe", FeatureEngineer()),
        ("prep", build_preprocessor()),
        ("model", estimator),
    ])


def candidate_models() -> dict:
    # class_weight="balanced" handles the ~27% churn imbalance without SMOTE
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=C.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                n_jobs=-1, random_state=C.RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_depth=4,
                                                            class_weight="balanced", random_state=C.RANDOM_STATE),
    }


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = get_clean_data()
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)

    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    scoring = ["roc_auc", "f1", "precision", "recall"]
    results, fitted = {}, {}

    for name, est in candidate_models().items():
        pipe = make_pipeline(est)
        cvres = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "cv_roc_auc_mean": float(cvres["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(cvres["test_roc_auc"].std()),
            "cv_f1_mean": float(cvres["test_f1"].mean()),
            "test_accuracy": float(accuracy_score(y_te, pred)),
            "test_precision": float(precision_score(y_te, pred)),
            "test_recall": float(recall_score(y_te, pred)),
            "test_f1": float(f1_score(y_te, pred)),
            "test_roc_auc": float(roc_auc_score(y_te, proba)),
        }
        fitted[name] = (pipe, proba)
        print(f"{name:22s} CV AUC {results[name]['cv_roc_auc_mean']:.3f} | test AUC {results[name]['test_roc_auc']:.3f} "
              f"| P {results[name]['test_precision']:.2f} R {results[name]['test_recall']:.2f} F1 {results[name]['test_f1']:.2f}")

    # Models often tie on CV ROC-AUC (to 3 decimals). Tie-break: prefer the simpler,
    # more interpretable model (dict order = simplest first).
    order = list(results)
    best_name = max(results, key=lambda n: (round(results[n]["cv_roc_auc_mean"], 3), -order.index(n)))
    best_pipe, best_proba = fitted[best_name]
    print("Best model (by CV ROC-AUC):", best_name)

    # --- figures for the best model ---
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_te, (best_proba >= 0.5).astype(int),
                                            display_labels=["Stayed", "Churned"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, (pipe, _) in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_te, y_te, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=.4)
    ax.set_title("ROC curves (hold-out test set)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "roc_curves.png", dpi=150); plt.close(fig)

    results["_best_model"] = best_name
    C.METRICS_PATH.write_text(json.dumps(results, indent=2))
    joblib.dump(best_pipe, C.MODEL_PATH)
    print("Saved model ->", C.MODEL_PATH)


if __name__ == "__main__":
    main()
