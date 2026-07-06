"""Training script (Kriteria 2) — autolog + manual logging metrics + DagsHub tracking."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "rice_preprocessing"
EXPERIMENT_NAME = "rice_classification"
MODEL_NAME = "rice-model"

# MLflow: connect ke DagsHub kalau tersedia, kalau tidak fallback lokal
_dagshub_user = os.getenv("DAGSHUB_USERNAME")
_dagshub_token = os.getenv("DAGSHUB_TOKEN")
_dagshub_repo = os.getenv("DAGSHUB_REPO")
if _dagshub_user and _dagshub_token and _dagshub_repo:
    tracking_uri = (
        f"https://{_dagshub_user}:{_dagshub_token}"
        f"@dagshub.com/{_dagshub_user}/{_dagshub_repo}.mlflow"
    )
    mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment(EXPERIMENT_NAME)


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    target = "Class"
    Xtr = train.drop(columns=[target])
    ytr = train[target]
    Xte = test.drop(columns=[target])
    yte = test[target]
    return (
        Xtr.reset_index(drop=True),
        pd.DataFrame({"Class": ytr.values}),
        Xte.reset_index(drop=True),
        pd.DataFrame({"Class": yte.values}),
    )


def _plot_confusion_matrix(cm: np.ndarray, classes: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _plot_feature_importance(model, feature_names, out: Path) -> None:
    importances = model.feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(feature_names)[order], importances[order], color="teal")
    ax.set_xlabel("Importance")
    ax.set_title("Feature importance (RandomForest)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> int:
    X_train_df, y_train_df, X_test_df, y_test_df = _load_data()
    y_train = y_train_df["Class"]
    y_test = y_test_df["Class"]
    classes = sorted(y_train.unique().tolist())
    print(f"train: {X_train_df.shape}, test: {X_test_df.shape}, classes: {classes}")

    params = dict(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)

    with mlflow.start_run(run_name="rf_base") as run:
        mlflow.sklearn.autolog(log_input_examples=False)

        model = RandomForestClassifier(**params)
        model.fit(X_train_df, y_train)

        y_pred = model.predict(X_test_df)
        y_proba = model.predict_proba(X_test_df)[:, 1]
        y_test_bin = (y_test == classes[1]).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=classes[1], zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=classes[1], zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=classes[1], zero_division=0)
        try:
            roc = roc_auc_score(y_test_bin, y_proba)
        except Exception:
            roc = float("nan")
        mlflow.log_metrics(
            {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}
        )
        mlflow.set_tag("dataset", os.getenv("DATASET_NAME", "rice"))
        mlflow.set_tag("author", os.getenv("NAMA_SISWA", "Muhammad-Yusuf"))
        mlflow.set_tag("stage", "Staging")
        print(f"metrics: acc={acc:.4f} f1={f1:.4f} auc={roc:.4f}")

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cm_path = ROOT / "confusion_matrix.png"
        _plot_confusion_matrix(cm, classes, cm_path)
        mlflow.log_artifact(str(cm_path))

        fi_path = ROOT / "feature_importance.png"
        _plot_feature_importance(model, list(X_train_df.columns), fi_path)
        mlflow.log_artifact(str(fi_path))

        try:
            mv = mlflow.register_model(
                f"runs:/{run.info.run_id}/model",
                MODEL_NAME,
            )
            print(f"Registered: {MODEL_NAME} v{mv.version}")
        except Exception as e:
            print(f"register skipped: {e}")

        print(f"RUN_ID={run.info.run_id}")
        print(f"ARTIFACT_URI={run.info.artifact_uri}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
