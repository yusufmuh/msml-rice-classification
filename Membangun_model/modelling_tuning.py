"""Tuning script (Kriteria 2) — Optuna sweep across RF + GBM + LogReg families.

Goal: capai akurasi >= 0.92 (test) dengan autolog + manual logging via DagsHub.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "rice_preprocessing"
EXPERIMENT_NAME = "rice_classification_tuning"
MODEL_NAME = "rice-model"

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


def _load():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    target = "Class"
    return (
        train.drop(columns=[target]).reset_index(drop=True),
        train[target],
        test.drop(columns=[target]).reset_index(drop=True),
        test[target],
    )


def _factory(name: str, params: dict):
    if name == "rf":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    if name == "gbm":
        return GradientBoostingClassifier(random_state=42, **params)
    if name == "lr":
        return LogisticRegression(random_state=42, max_iter=2000, **params)
    raise ValueError(name)


def main() -> int:
    X_train, y_train, X_test, y_test = _load()
    classes = sorted(y_train.unique().tolist())

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        family = trial.suggest_categorical("family", ["rf", "gbm", "lr"])
        if family == "rf":
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 200, 500),
                max_depth=trial.suggest_int("max_depth", 6, 30),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 8),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            )
        elif family == "gbm":
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
            )
        else:  # lr
            params = dict(
                C=trial.suggest_float("C", 0.1, 10.0, log=True),
                solver="liblinear",
            )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
        try:
            scores = cross_val_score(
                _factory(family, params),
                X_train, y_train,
                scoring="accuracy", cv=cv, n_jobs=-1,
            )
        except Exception:
            return float("-inf")
        return float(scores.mean())

    with mlflow.start_run(run_name="optuna_sweep") as run:
        study.optimize(objective, n_trials=40, show_progress_bar=False)
        best = study.best_trial
        family = best.params.pop("family")
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metric("best_cv_accuracy", best.value)

        model = _factory(family, best.params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba_full = model.predict_proba(X_test)
        y_proba = y_proba_full[:, 1]
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
        mlflow.set_tag("best_family", family)

        mlflow.sklearn.log_model(model, artifact_path="model")

        if acc >= 0.92:
            try:
                client = mlflow.tracking.MlflowClient()
                try:
                    client.create_registered_model(MODEL_NAME)
                except Exception:
                    pass
                mv = client.create_model_version(
                    name=MODEL_NAME,
                    source=f"runs:/{run.info.run_id}/model",
                    run_id=run.info.run_id,
                )
                client.transition_model_version_stage(
                    name=MODEL_NAME, version=mv.version, stage="Production"
                )
                print(f"Promoted v{mv.version} to Production.")
            except Exception as e:
                print(f"register/transition skipped: {e}")
        else:
            print(f"Accuracy {acc:.4f} < 0.92; will not promote.")

        model_path = ROOT / "rice_tuned.joblib"
        joblib.dump({"model": model, "classes": classes, "family": family}, model_path)
        print(f"Saved best ({family}) model to {model_path}")
        print(f"FINAL_TEST_ACCURACY={acc:.4f}")
        print(f"RUN_ID={run.info.run_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
