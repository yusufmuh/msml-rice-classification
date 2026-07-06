"""Fetch dataset from UCI ML Repository and save raw CSV."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATASET_ID = int(os.getenv("DATASET_ID", "545"))
DATASET_NAME = os.getenv("DATASET_NAME", "rice")
OUT_DIR = Path("Membangun_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / f"{DATASET_NAME}_raw.csv"


def main() -> int:
    try:
        from ucimlrepo import fetch_ucirepo  # type: ignore
    except ImportError:
        print("ERROR: ucimlrepo not installed. Run: pip install ucimlrepo==0.0.7", file=sys.stderr)
        return 1

    print(f"Fetching UCI dataset id={DATASET_ID} ...")
    ds = fetch_ucirepo(id=DATASET_ID)
    X = ds.data.features
    y = ds.data.targets
    print(f"  features shape: {X.shape}, target shape: {y.shape}")
    print(f"  target name: {y.columns.tolist()}")
    print(f"  classes: {y.iloc[:, 0].unique().tolist()}")

    import pandas as pd
    df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
