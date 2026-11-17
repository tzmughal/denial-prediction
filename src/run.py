"""Run the whole Denial Prediction pipeline end to end.

    python src/run.py

Steps: prepare (split check) -> train (both models, full + honest
no-days_in_ar variants) -> evaluate (metrics + charts).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate
import prepare
import train

if __name__ == "__main__":
    print("== 1. Prepare ==")
    df = prepare.load_claims()
    X_train, X_test, y_train, y_test = prepare.split_data(df)
    print(f"Train {len(X_train)} / Test {len(X_test)} "
          f"(denied: train {int(y_train.sum())}, test {int(y_test.sum())})")
    print("\n== 2. Train ==")
    train.main()
    print("\n== 3. Evaluate ==")
    evaluate.main()
    print("\nDone. See README.md for the results and screenshots/.")
