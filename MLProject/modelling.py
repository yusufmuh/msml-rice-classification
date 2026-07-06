"""MLProject entry — runs Optuna tuning against the rice dataset."""
import argparse
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
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
# Search for the preprocessed data — try a few likely locations.
_candidates = [
    ROOT / "Membangun_model" / "rice_preprocessing",
    ROOT.parent / "Membangun_model" / "rice_preprocessing",
]
DATA_DIR = next((c for c in _candidates if (c / "train.csv").exists()), _candidates[1])
MODEL_NAME = "rice-model"

_dagshub_user = os.getenv("DAGSHUB_USERNAME")
_dagshub_token = os.getenv("DAGSHUB_TOKEN")
_dagshub_repo = os.getenv("DAGSHUB_REPO")

# CI env: MLFLOW_TRACKING_URI is exported as plain https://… (no creds in URL);
# MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD carry the basic-auth creds.
_tracking_uri_env = os.getenv("MLFLOW_TRACKING_URI")
if _tracking_uri_env:
    # Re-inject credentials into URL so the remote tracking backend accepts them.
    if _dagshub_user and _dagshub_token and "://" in _tracking_uri_env and "@" not in _tracking_uri_env:
        if _tracking_uri_env.startswith("https://"):
            tracking_uri = (
                f"https://{_dagshub_user}:{_dagshub_token}@{_tracking_uri_env[len('https://'):]}"
            )
        else:
            tracking_uri = _tracking_uri_env
    else:
        tracking_uri = _tracking_uri_env
    mlflow.set_tracking_uri(tracking_uri)

# Local fallback for non-CI runs that pass token only via env (not URL).
if not mlflow.get_tracking_uri() or "dagshub.com" not in mlflow.get_tracking_uri():
    if _dagshub_user and _dagshub_token and _dagshub_repo:
        tracking_uri = (
            f"https://{_dagshub_user}:{_dagshub_token}"
            f"@dagshub.com/{_dagshub_user}/{_dagshub_repo}.mlflow"
        )
        mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("rice_classification_ci")


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
        return LogisticRegression(random_state=42, max_iter=2000, solver="liblinear", **params)
    raise ValueError(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, default=545)
    parser.add_argument("--n-trials", type=int, default=40)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = _load()
    classes = sorted(y_train.unique().tolist())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
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
        else:
            params = dict(C=trial.suggest_float("C", 0.1, 10.0, log=True))
        try:
            return float(cross_val_score(_factory(family, params), X_train, y_train,
                                          scoring="accuracy", cv=cv, n_jobs=-1).mean())
        except Exception:
            return float("-inf")

    with mlflow.start_run(run_name="ci_optuna") as run:
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
        best = study.best_trial
        family = best.params.pop("family")
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metric("best_cv_accuracy", best.value)

        model = _factory(family, best.params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_test_bin = (y_test == classes[1]).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=classes[1], zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=classes[1], zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=classes[1], zero_division=0)
        try:
            roc = roc_auc_score(y_test_bin, y_proba)
        except Exception:
            roc = float("nan")
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec,
                            "f1": f1, "roc_auc": roc})
        mlflow.set_tag("dataset_id", args.dataset_id)
        mlflow.set_tag("ci", "github-actions")

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
                client.transition_model_version_stage(name=MODEL_NAME, version=mv.version,
                                                     stage="Production")
            except Exception as e:
                print(f"register/transition skipped: {e}", file=sys.stderr)

        print(f"CI_RUN accuracy={acc:.4f} run_id={run.info.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
