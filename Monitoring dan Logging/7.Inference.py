"""Inference script — calls the locally-served MLflow model and prints the prediction.
The serving process must be started first, e.g.:
    mlflow models serve -m "models:/rice-model/Production" -p 5005 --no-conda
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
TEST_CSV = ROOT / ".." / "Membangun_model" / "rice_preprocessing" / "test.csv"
SCALER   = ROOT / ".." / "Membangun_model" / "rice_preprocessing" / "scaler.joblib"
MODEL    = ROOT / ".." / "Membangun_model" / "rice_tuned.joblib"
SERVE    = os.getenv("SERVE_URL", "http://localhost:5005/invocations")
TIMEOUT  = int(os.getenv("SERVE_TIMEOUT", "60"))


def check_serving(url: str, retries: int = 30, delay: float = 2.0) -> bool:
    ping = url.replace("/invocations", "/ping")
    for i in range(retries):
        try:
            r = requests.get(ping, timeout=2)
            if r.status_code == 200:
                print(f"Serving ready at {ping}")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(delay)
        print(f"waiting for serving ({i+1}/{retries}) ...")
    return False


def serve_inline() -> int:
    """Fallback: load model locally and predict directly (no server required)."""
    if not TEST_CSV.exists() or not MODEL.exists():
        print("ERROR: missing artifacts", file=sys.stderr)
        return 1
    df = pd.read_csv(TEST_CSV)
    X = df.drop(columns=["Class"]).head(5)
    payload = {"dataframe_split": {
        "columns": list(X.columns),
        "data": X.values.tolist(),
    }}
    try:
        if check_serving(SERVE):
            r = requests.post(SERVE, json=payload,
                              headers={"Content-Type": "application/json"},
                              timeout=TIMEOUT)
            print("Remote response:", r.status_code, r.text[:500])
            return 0 if r.status_code == 200 else 2
    except Exception as e:
        print(f"Remote serving failed: {e}. Falling back to local load.")

    import joblib
    bundle = joblib.load(MODEL)
    model = bundle["model"]
    classes = bundle["classes"]
    preds = model.predict(X)
    proba = model.predict_proba(X)
    print("LOCAL inferences (first 5 rows):")
    for i, row in enumerate(X.itertuples(index=False), start=1):
        cls = preds[i-1]
        p = proba[i-1]
        print(f"  sample {i}: pred={cls} (proba {p[0]:.3f}/{p[1]:.3f})  "
              f"features={row}")
    return 0


if __name__ == "__main__":
    sys.exit(serve_inline())
