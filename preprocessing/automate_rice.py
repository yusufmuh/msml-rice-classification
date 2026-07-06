"""Automated preprocessing for the Rice dataset (UCI id=545).

Idempotent: re-running produces the same artifacts.

Outputs (in `Membangun_model/rice_preprocessing/`):
- train.csv, test.csv        (split features + label)
- scaler.joblib              (StandardScaler fitted on train)
- label_encoder.joblib       (LabelEncoder fitted on y)
-eda_class_distribution.png
- eda_feature_histograms.png
- eda_correlation.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

load_dotenv()

DATASET_NAME = os.getenv("DATASET_NAME", "rice")
ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "Membangun_model" / f"{DATASET_NAME}_raw.csv"
OUT_DIR = ROOT / "Membangun_model" / f"{DATASET_NAME}_preprocessing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 10
TEST_SIZE = 0.2


def _read_raw() -> pd.DataFrame:
    if not RAW_CSV.exists():
        print(f"ERROR: {RAW_CSV} not found. Run scripts/fetch_dataset.py first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(df)} rows from {RAW_CSV}")
    return df


def _eda(df: pd.DataFrame) -> None:
    """Produce 3+ EDA visualizations as required."""
    sns.set_style("whitegrid")
    target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=target_col, data=df, ax=ax, palette="viridis")
    ax.set_title(f"Class distribution ({target_col})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_class_distribution.png", dpi=110)
    plt.close(fig)

    # 2. Feature histograms
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = axes.flatten()
    for ax, col in zip(axes, feature_cols):
        ax.hist(df[col], bins=30, color="steelblue", edgecolor="black")
        ax.set_title(col)
    for ax in axes[len(feature_cols):]:
        ax.axis("off")
    fig.suptitle("Feature histograms")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_feature_histograms.png", dpi=110)
    plt.close(fig)

    # 3. Correlation matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    ax.set_title("Pearson correlation (numeric features)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_correlation.png", dpi=110)
    plt.close(fig)

    # 4. Bonus: feature boxplot for outlier overview
    fig, ax = plt.subplots(figsize=(10, 5))
    df[feature_cols].boxplot(ax=ax)
    ax.set_title("Feature boxplots (outlier screening)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_boxplots.png", dpi=110)
    plt.close(fig)

    print("EDA figures written ->", OUT_DIR)


def _clean_and_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    # handle missing + duplicates
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    print(f"After dedupe+dropna: {len(df)} rows")

    # encode label
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    X = df[feature_cols].copy()

    # train/test split (stratify untuk keseimbangan kelas)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # numeric scaling
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    train_out = X_train_s.copy()
    train_out[target_col] = le.inverse_transform(y_train)
    test_out = X_test_s.copy()
    test_out[target_col] = le.inverse_transform(y_test)

    return train_out, test_out, le, scaler


def main() -> int:
    df = _read_raw()
    _eda(df)
    train_df, test_df, le, scaler = _clean_and_split(df)

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    joblib.dump(le, OUT_DIR / "label_encoder.joblib")

    print(f"train rows: {len(train_df)}, test rows: {len(test_df)}")
    print(f"Artifacts -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
