#!/usr/bin/env bash
# End-to-end orchestrator — runs all 4 criteria sequentially.
# Requires a populated .env in the project root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

set -a
# shellcheck disable=SC1091
source .env
set +a

log() { printf "\n\033[1;36m==[ %s ]==\033[0m %s\n" "run_all" "$*"; }

log "0/8  Install dependencies (Python 3.12)"
PY=python3.12
if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python
fi
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt

log "1/8  Fetch dataset + preprocess"
"$PY" scripts/fetch_dataset.py
"$PY" preprocessing/automate_rice.py

log "2/8  Train (baseline) + track online"
"$PY" Membangun_model/modelling.py
log "2b/8 Tune (Optuna RF/GBM/LR) + push best to Production"
"$PY" Membangun_model/modelling_tuning.py

log "3/8  mlflow serve (background) for inference"
nohup mlflow models serve -m "models:/rice-model/Production" -p 5005 --no-conda -h 127.0.0.1 > serve.log 2>&1 &
SERVE_PID=$!
echo "serve pid: $SERVE_PID"
trap 'kill $SERVE_PID 2>/dev/null || true' EXIT
for i in {1..30}; do
  if curl -fsS "http://127.0.0.1:5005/ping" >/dev/null 2>&1; then
    echo "MLflow serve ready."; break
  fi
  sleep 2
done

log "4/8  Inference + serving evidence"
"$PY" "Monitoring dan Logging/7.Inference.py"
"$PY" scripts/render_serving_proof.py

log "5/8  Prometheus exporter (background)"
nohup "$PY" "Monitoring dan Logging/3.prometheus_exporter.py" > "Monitoring dan Logging/exporter.log" 2>&1 &
EXP_PID=$!
trap 'kill $SERVE_PID $EXP_PID 2>/dev/null || true' EXIT
sleep 5
for i in {1..5}; do
  if curl -fsS "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
    echo "Exporter ready."; break
  fi
  sleep 2
done
# Trigger some predictions to populate metrics
for i in 1 2 3 4 5 6 7 8; do
  curl -fsS -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{}' >/dev/null || true
done
"$PY" scripts/render_metrics_proof.py

log "6/8  Provision Grafana Cloud + capture proof"
bash "Monitoring dan Logging/setup_grafana.sh" || echo "Grafana provisioning skipped/failed."
"$PY" scripts/capture_grafana_public.py

log "7/8  Capture DagsHub & general evidence"
"$PY" scripts/capture_evidence.py

log "8/8  Build ZIP submission"
"$PY" scripts/build_submission_zip.py

echo
echo "SELESAI. Hasil: ./SMSML_$NAMA_SISWA.zip"
