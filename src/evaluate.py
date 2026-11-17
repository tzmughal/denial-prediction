"""Evaluate the trained models and build the charts.

Reads the saved models from models/ (run `python src/train.py` first),
prints test-set metrics, and writes four matplotlib PNGs to
screenshots/:
  - roc_curve.png            ROC curves, both full models
  - confusion_matrix.png     confusion matrices, both full models
  - feature_importance.png   RandomForest top-15 features (full model)
  - denial_rate_by_payer.png denial rate by payer, from the actual data
  - lr_coefficients.png      Logistic Regression coefficients (extra)

Also prints the honest no-days_in_ar comparison from models/metrics.json.

Run:  python src/evaluate.py
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             roc_auc_score)

from prepare import DATA_PATH, FEATURES, load_claims, split_data

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
SHOTS_DIR = BASE_DIR / "screenshots"
SHOTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


def get_feature_names(pipeline) -> list:
    """One-hot-expanded feature names, in pipeline order."""
    prep = pipeline.named_steps["prep"]
    return list(prep.get_feature_names_out())


def plot_roc(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, model in models.items():
        # RocCurveDisplay appends "(AUC = ...)" to the name itself,
        # so pass a plain name — no AUC in it, or it prints twice.
        RocCurveDisplay.from_estimator(
            model, X_test, y_test, ax=ax,
            name=name.replace("_", " ").title())
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC=0.500)")
    ax.set_title("ROC Curve — Denial Prediction (test set, n=160)\n"
                 "Synthetic data")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(SHOTS_DIR / "roc_curve.png")
    plt.close(fig)


def plot_confusion(models, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (name, model) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, ax=ax, cmap="Blues",
            display_labels=["Not denied", "Denied"])
        ax.set_title(name.replace("_", " ").title())
    fig.suptitle("Confusion Matrices — Test Set (n=160, 30 denied)\n"
                 "Synthetic data")
    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    fig.savefig(SHOTS_DIR / "confusion_matrix.png")
    plt.close(fig)


def plot_rf_importance(model):
    names = get_feature_names(model)
    importances = model.named_steps["clf"].feature_importances_
    order = np.argsort(importances)[::-1][:15]
    top_names = [names[i].replace("cat__", "").replace("num__", "")
                 for i in order[::-1]]
    top_vals = importances[order[::-1]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_names, top_vals, color="#2563eb")
    ax.set_xlabel("RandomForest feature importance")
    ax.set_title("Top 15 Features — RandomForest (full model)\n"
                 "Synthetic data")
    fig.tight_layout()
    fig.savefig(SHOTS_DIR / "feature_importance.png")
    plt.close(fig)
    return [(names[i], round(float(importances[i]), 4)) for i in order]


def plot_lr_coefficients(model):
    names = get_feature_names(model)
    coefs = model.named_steps["clf"].coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:15]
    top_names = [names[i].replace("cat__", "").replace("num__", "")
                 for i in order[::-1]]
    top_vals = coefs[order[::-1]]
    colors = ["#dc2626" if v > 0 else "#2563eb" for v in top_vals]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_names, top_vals, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Logistic Regression coefficient "
                  "(positive = higher denial odds; red)")
    ax.set_title("Top 15 Coefficients — Logistic Regression (full model)\n"
                 "Synthetic data")
    fig.tight_layout()
    fig.savefig(SHOTS_DIR / "lr_coefficients.png")
    plt.close(fig)
    return [(names[i], round(float(coefs[i]), 4)) for i in order]


def plot_denial_by_payer(df):
    rates = (df.assign(denied=df["status"] == "Denied")
             .groupby("payer")["denied"].agg(["mean", "count"])
             .sort_values("mean", ascending=False))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rates.index, rates["mean"] * 100, color="#2563eb")
    for i, (rate, n) in enumerate(zip(rates["mean"], rates["count"])):
        ax.text(i, rate * 100 + 0.6, f"{rate:.1%} (n={n})",
                ha="center", fontsize=9)
    ax.set_ylabel("Denial rate (%)")
    ax.set_ylim(0, max(rates["mean"]) * 100 + 5)
    ax.set_title("Denial Rate by Payer — all 800 synthetic claims\n"
                 "Overall denial rate 19.0% (152 of 800)")
    fig.tight_layout()
    fig.savefig(SHOTS_DIR / "denial_rate_by_payer.png")
    plt.close(fig)
    return rates


def main():
    df = load_claims()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train)} rows ({int(y_train.sum())} denied) | "
          f"Test: {len(X_test)} rows ({int(y_test.sum())} denied)")
    assert len(X_train) + len(X_test) == len(df) == 800

    models = {}
    for name in ["logistic_regression", "random_forest"]:
        path = MODELS_DIR / f"{name}.joblib"
        assert path.exists(), f"missing {path} — run src/train.py first"
        models[name] = joblib.load(path)
        # Leakage check on the loaded pipeline: no forbidden column
        # may appear among its input features.
        feats = list(models[name].feature_names_in_)
        forbidden = {"paid_amount", "denial_reason", "status",
                     "claim_id", "patient_id"}
        assert not (set(feats) & forbidden), f"LEAKAGE in {name}: {feats}"
        print(f"{name}: features in use = {feats} — no leakage features.")

    plot_roc(models, X_test, y_test)
    plot_confusion(models, X_test, y_test)
    rf_top = plot_rf_importance(models["random_forest"])
    lr_top = plot_lr_coefficients(models["logistic_regression"])
    rates = plot_denial_by_payer(df)

    print("\nRandomForest top 10 features (full model):")
    for n, v in rf_top[:10]:
        print(f"  {n:30s} {v:.4f}")
    print("\nLogistic Regression top 10 coefficients (by |value|):")
    for n, v in lr_top[:10]:
        print(f"  {n:30s} {v:+.4f}")
    print("\nDenial rate by payer (all 800 claims):")
    for payer, row in rates.iterrows():
        print(f"  {payer:20s} {row['mean']:.1%}  (n={int(row['count'])})")

    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        print(f"\nFull metrics table (from {metrics_path}):")
        print(json.dumps(json.loads(metrics_path.read_text()), indent=2))

    print("\nCharts written to screenshots/: roc_curve.png, "
          "confusion_matrix.png, feature_importance.png, "
          "lr_coefficients.png, denial_rate_by_payer.png")


if __name__ == "__main__":
    main()
