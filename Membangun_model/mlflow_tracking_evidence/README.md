# Bukti MLflow Tracking — Kriteria 2 (Membangun Model Machine Learning)

Folder ini berisi bukti tangkapan layar MLflow Tracking UI untuk setiap level kriteria 2,
sebagai pelengkap `screenshoot_dashboard.jpg` dan `screenshoot_artifak.jpg` di folder induk.

## `local/` — Basic & Skilled (MLflow Tracking UI lokal, `http://127.0.0.1:5000`)

| File | Menunjukkan |
|---|---|
| `mlflow_local_experiments_list.jpg` | Daftar experiment lokal (`rice_classification`, `rice_classification_tuning`) di `127.0.0.1:5000` |
| `mlflow_local_run_overview.jpg` | Run `rf_base` (dari `modelling.py`) dengan **autolog** — 19 params, 12 metrics otomatis |
| `mlflow_local_run_artifacts.jpg` | Artefak autolog: `confusion_matrix.png`, `estimator.html`, `training_*.png`, model (MLmodel/conda.yaml/model.pkl) |
| `mlflow_local_model_registry_production.jpg` | Model registry `rice-model` — Version 2 berstatus **Production** |
| `mlflow_local_tuning_experiment_list.jpg` | Experiment `rice_classification_tuning` dengan run `optuna_sweep` (dari `modelling_tuning.py`, hyperparameter tuning) |
| `mlflow_local_tuning_run_manual_logging.jpg` | Run `optuna_sweep` dengan **manual logging** (bukan autolog): parameter `best_C` + 6 metrik manual (accuracy, precision, recall, f1, roc_auc, best_cv_accuracy) |
| `mlflow_local_tuning_extra_artifacts.jpg` | 2 artefak tambahan hasil manual logging: `confusion_matrix_tuned.png` dan `optuna_optimization_history.png` |

## `dagshub_online/` — Advance (MLflow Tracking UI online via DagsHub)

Tracking URI: `https://dagshub.com/yusufmuh/msml-rice-classification.mlflow`
(catatan: server DagsHub menjalankan MLflow versi 3.5.1, berbeda dari versi lokal 2.19.0 —
terlihat jelas di pojok kiri atas setiap tangkapan layar sebagai bukti berjalan di server terpisah/online)

| File | Menunjukkan |
|---|---|
| `mlflow_dagshub_experiments_list.jpg` | Riwayat run `optuna_sweep` di experiment `rice_classification_tuning` yang tersimpan online di DagsHub, dengan model teregistrasi `rice-model` v16 |
| `mlflow_dagshub_tuning_run_overview.jpg` | Detail run online: `Created by yusufmuh` (akun DagsHub), manual logging metrics (6 metrik), param `best_C`, source `modelling_tuning.py` |
| `mlflow_dagshub_tuning_run_artifacts.jpg` | Artefak manual logging tersimpan online: `confusion_matrix_tuned.png` + `optuna_optimization_history.png` (memenuhi syarat "autolog + minimal 2 artefak tambahan") |
| `mlflow_dagshub_model_registry.jpg` | Model registry `rice-model` online di DagsHub — 16 versi terdaftar dengan timestamp riwayat lengkap |

## Ringkasan pemenuhan Kriteria 2

- **Basic**: `modelling.py` — autolog, MLflow Tracking UI lokal → `local/mlflow_local_run_overview.jpg`, `mlflow_local_run_artifacts.jpg`
- **Skilled**: `modelling_tuning.py` — hyperparameter tuning (Optuna, 40 trials) + manual logging, MLflow Tracking UI lokal → `local/mlflow_local_tuning_run_manual_logging.jpg`
- **Advance**: `modelling_tuning.py` dijalankan dengan tracking online ke DagsHub, manual logging + 2 artefak tambahan (`confusion_matrix_tuned.png`, `optuna_optimization_history.png`) → `dagshub_online/mlflow_dagshub_tuning_run_artifacts.jpg`
