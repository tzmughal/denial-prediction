"""Prepare the modelling dataset for the Denial Prediction project.

Source: the SAME synthetic claims file as Project 1 (RCM Reporting
Dashboard) — ../rcm-reporting-dashboard/data/claims_raw.csv
(800 synthetic claims, seed 42). This script does NOT regenerate data,
so Project 1 and Project 2 always agree.

Target:  denied = 1 if status == "Denied" else 0
          (Pending claims count as 0 — they are not denied yet.)

Features (no leakage):
  - payer            (categorical)
  - cpt_code         (categorical — treated as a code, not a number)
  - icd10_code       (categorical)
  - billed_amount    (numeric, known at submission)
  - days_in_ar       (numeric — see the leakage note in README.md)

EXCLUDED on purpose (leakage / post-denial information):
  - paid_amount      — $0 for every denied claim; using it would leak the outcome
  - denial_reason    — only exists after a denial has happened
  - claim_id, patient_id — identifiers, not predictors
  - status           — this IS the target
  - service_date     — not used; the synthetic dates carry no denial signal
                       worth modelling and would invite spurious time fits

Split: stratified 80/20 train/test, random_state=42.
Run:  python src/prepare.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    BASE_DIR.parent / "rcm-reporting-dashboard" / "data" / "claims_raw.csv"
)
# Fallback: a local copy, if the Project 1 folder is ever moved.
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "data" / "claims_raw.csv"

CATEGORICAL_FEATURES = ["payer", "cpt_code", "icd10_code"]
NUMERIC_FEATURES = ["billed_amount", "days_in_ar"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Everything that must NEVER be a feature. Checked again in train.py.
LEAKAGE_COLUMNS = ["paid_amount", "denial_reason", "status",
                   "claim_id", "patient_id"]


def load_claims(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the Project 1 synthetic claims CSV and add the target."""
    df = pd.read_csv(path, parse_dates=["service_date"])
    df["denied"] = (df["status"] == "Denied").astype(int)
    # CPT is a code, not a quantity — keep it categorical.
    df["cpt_code"] = df["cpt_code"].astype(str)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the allowed feature columns, in a fixed order."""
    overlap = set(FEATURES) & set(LEAKAGE_COLUMNS)
    assert not overlap, f"leakage feature in FEATURES: {overlap}"
    missing = [c for c in FEATURES if c not in df.columns]
    assert not missing, f"missing feature columns: {missing}"
    return df[FEATURES].copy()


def split_data(df: pd.DataFrame, test_size: float = 0.20,
               random_state: int = 42):
    """Stratified 80/20 train/test split on the denied target."""
    X = build_features(df)
    y = df["denied"]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y,
    )


if __name__ == "__main__":
    df = load_claims()
    print(f"Source: {DATA_PATH}")
    print(f"Rows: {len(df)}  (billed ${df['billed_amount'].sum():,.2f}, "
          f"paid ${df['paid_amount'].sum():,.2f})")
    print(f"Status split:\n{df['status'].value_counts().to_string()}")
    print(f"Denied: {int(df['denied'].sum())} "
          f"({df['denied'].mean():.1%} of {len(df)})")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train)} rows, {int(y_train.sum())} denied "
          f"({y_train.mean():.1%})")
    print(f"Test:  {len(X_test)} rows, {int(y_test.sum())} denied "
          f"({y_test.mean():.1%})")
    print(f"Features used: {FEATURES}")
    print(f"Excluded (leakage): paid_amount, denial_reason, "
          f"claim_id, patient_id, status")
    assert len(X_train) + len(X_test) == len(df)
    assert int(y_train.sum()) + int(y_test.sum()) == int(df["denied"].sum())
    print("Split checks passed: train + test = 800, denied counts add up.")
