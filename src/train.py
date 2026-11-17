"""Train the denial-prediction models (Logistic Regression + RandomForest).

Both models are scikit-learn Pipelines:
  categorical features -> OneHotEncoder(handle_unknown="ignore")
  numeric features     -> StandardScaler (Logistic Regression only;
                          RandomForest does not need scaling)
  classifier           -> LogisticRegression / RandomForestClassifier

class_weight="balanced" is used for both classifiers because only 19%
of claims are denied — without it the models would largely ignore the
minority (denied) class.

A second, honest variant is also trained for each model WITHOUT
days_in_ar (features: payer, CPT, ICD-10, billed_amount only). See
README.md for why: days_in_ar is measured after the claim has been
sitting unpaid, so a model that uses it is not a true at-submission
predictor. The no-days_in_ar variant is what could actually be used
when a claim is first submitted.

Outputs (in the project root):
  models/logistic_regression.joblib, models/random_forest.joblib
  (+ *_no_days_in_ar.joblib variants)
  models/metrics.json — test-set metrics for every model, from the real run

Run:  python src/train.py
"""
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prepare import (CATEGORICAL_FEATURES, DATA_PATH, FEATURES,
                     NUMERIC_FEATURES, load_claims, split_data)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def build_preprocessor(use_scaling: bool, cat_features=None,
                       num_features=None) -> ColumnTransformer:
    cat_features = cat_features if cat_features is not None else CATEGORICAL_FEATURES
    num_features = num_features if num_features is not None else NUMERIC_FEATURES
    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
    if use_scaling:
        transformers.append(("num", StandardScaler(), num_features))
    else:
        transformers.append(("num", "passthrough", num_features))
    return ColumnTransformer(transformers)


def build_models(feature_columns):
    """Return {name: unfitted Pipeline} for the two models in the brief."""
    cat = [c for c in CATEGORICAL_FEATURES if c in feature_columns]
    num = [c for c in NUMERIC_FEATURES if c in feature_columns]
    logreg = Pipeline([
        ("prep", build_preprocessor(use_scaling=True, cat_features=cat,
                                    num_features=num)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=42)),
    ])
    forest = Pipeline([
        ("prep", build_preprocessor(use_scaling=False, cat_features=cat,
                                    num_features=num)),
        ("clf", RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1,
            class_weight="balanced_subsample")),
    ])
    return {"logistic_regression": logreg, "random_forest": forest}


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
        "test_n": int(len(y_test)),
        "test_denied": int(y_test.sum()),
    }


def train_and_evaluate(feature_columns, tag: str) -> dict:
    """Train both models on the given feature subset; return metrics."""
    import pandas as pd

    df = load_claims()
    X = df[feature_columns]
    y = df["denied"]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    results = {}
    for name, model in build_models(list(X_train.columns)).items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        out = MODELS_DIR / f"{name}{tag}.joblib"
        joblib.dump(model, out)
        cm = metrics["confusion_matrix"]
        print(f"  [{tag or 'full':>15}] {name:20s} "
              f"acc={metrics['accuracy']:.3f} "
              f"prec={metrics['precision']:.3f} "
              f"rec={metrics['recall']:.3f} "
              f"F1={metrics['f1']:.3f} AUC={metrics['roc_auc']:.3f} "
              f"CM=[TN {cm['tn']} FP {cm['fp']} / "
              f"FN {cm['fn']} TP {cm['tp']}] -> {out.name}")
    return results


def main():
    import json

    MODELS_DIR.mkdir(exist_ok=True)
    df = load_claims()
    print(f"Source: {DATA_PATH} — {len(df)} claims, "
          f"{int(df['denied'].sum())} denied ({df['denied'].mean():.1%})")
    # Leakage guard: fail loudly if a forbidden column ever becomes a feature.
    forbidden = {"paid_amount", "denial_reason", "status",
                 "claim_id", "patient_id"}
    assert not (set(FEATURES) & forbidden), "LEAKAGE: forbidden feature!"

    print("Training full models (payer, CPT, ICD-10, billed_amount, "
          "days_in_ar):")
    all_metrics = {"full_features": train_and_evaluate(FEATURES, "")}

    print("Training honest at-submission models (no days_in_ar):")
    no_days = [c for c in FEATURES if c != "days_in_ar"]
    all_metrics["no_days_in_ar"] = train_and_evaluate(
        no_days, "_no_days_in_ar")

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {MODELS_DIR / 'metrics.json'}")
    return all_metrics


if __name__ == "__main__":
    main()
