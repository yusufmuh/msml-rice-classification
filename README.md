# SMSML Rice Classification — Muhammad Yusuf (APC005D6Y0216)

End-to-end MLOps pipeline untuk submission **Proyek Akhir MSML (Dicoding)** dengan target nilai **4.0 / Bintang 5**.

## Kriteria

1. **Eksperimen + otomasi preprocessing** — `preprocessing/automate_rice.py`, Notebook `notebook/Eksperimen_SML_Muhammad-Yusuf.ipynb`, GitHub Actions `preprocessing.yml`.
2. **Training + tuning + tracking online** — `Membangun_model/modelling.py` (autolog) & `modelling_tuning.py` (Optuna RF/GBM/LR → Logistic Regression best, accuracy **0.9541**). Tracking ke DagsHub MLflow. Screenshot: `screenshoot_dashboard.jpg`, `screenshoot_artifak.jpg`. Link: `DagsHub.txt`.
3. **CI + packaging** — `MLProject/` (MLProject + python_env.yaml + modelling.py), `.github/workflows/ci.yml` (mlflow run → build docker → push Docker Hub).
4. **Serving + monitoring + alerting** — `mlflow models serve` + `Inference.py`, Prometheus exporter dengan ≥3 metrik, Grafana dashboard bernama `APC005D6Y0216` + ≥3 alert rule, screenshot bukti ada.

## Dataset
Rice (Cammeo & Osmancik) — UCI ML Repository id **545** — 3810 sampel biner, 7 fitur morfologi.

## Cara menjalankan end-to-end

```bash
# Setup
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Load .env (isi kredensial)
set -a && source .env && set +a

# 1. Fetch + preprocess
python scripts/fetch_dataset.py
python preprocessing/automate_rice.py

# 2. Train + track + tune
python Membangun_model/modelling.py
python Membangun_model/modelling_tuning.py

# 3. MLProject (CI)
mlflow run MLProject -P n_trials=20 --no-conda

# 4. Serve locally (background)
mlflow models serve -m "models:/rice-model/Production" -p 5005 --no-conda &

# 5. Inference
python "Monitoring dan Logging/7.Inference.py"

# 6. Monitoring
python "Monitoring dan Logging/3.prometheus_exporter.py" &
docker run -d -p 9090:9090 -v "$(pwd)/Monitoring dan Logging/2.prometheus.yml":/etc/prometheus/prometheus.yml prom/prometheus

# 7. Capture bukti
python scripts/capture_evidence.py

# 8. ZIP final
python scripts/build_submission_zip.py
```

Atau cukup `bash run_all.sh` dari root project.

## Output
- Tracking MLflow online: `https://dagshub.com/yusufmuh/msml-rice-classification.mlflow`
- Docker image: `docker.io/yusufmuh/msml-rice-model:latest`
- Submission: `SMSML_Muhammad-Yusuf.zip`

> ⚠️ Kredensial di `.env` **TIDAK** di-commit. Lihat `.gitignore`.
