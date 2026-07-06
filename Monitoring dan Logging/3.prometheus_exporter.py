"""Prometheus exporter for the MSML rice model.

Exposes at least 6 metrics on :${EXPORTER_PORT}/metrics:
- prediction_total (Counter)
- prediction_latency_seconds (Histogram)
- prediction_errors_total (Counter)
- model_probability_cammeo (Gauge — last predicted probability for class 0)
- model_probability_osmancik (Gauge — last predicted probability for class 1)
- inference_up (Gauge — 1 if exporter running)

Also reaches MLflow served model; if not available uses the local `rice_tuned.joblib`.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

load_dotenv()

EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "8000"))
SERVE_URL = os.getenv("SERVE_URL", "http://localhost:5005/invocations")
TEST_CSV = Path(__file__).resolve().parent.parent / "Membangun_model" / "rice_preprocessing" / "test.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "Membangun_model" / "rice_tuned.joblib"

prediction_total = Counter("prediction_total", "Total predictions served")
prediction_errors = Counter("prediction_errors_total", "Total errors")
prediction_latency = Histogram("prediction_latency_seconds", "Latency of /predict")
model_prob_cammeo = Gauge("model_probability_cammeo", "Probability of Cammeo class (last)")
model_prob_osmancik = Gauge("model_probability_osmancik", "Probability of Osmancik class (last)")
inference_up = Gauge("inference_up", "1 if exporter healthy")
last_predicted_class = Gauge("last_predicted_class_cammeo", "1 if last class=Cammeo, 0 otherwise")

# Load model for offline path
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
classes = bundle["classes"]

app = Flask(__name__)


def _sample_features():
    df = pd.read_csv(TEST_CSV).drop(columns=["Class"])
    return df.sample(5, random_state=0).values.tolist()


@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/predict", methods=["POST"])
def predict():
    start = time.perf_counter()
    try:
        data = request.get_json(force=True) or {}
        samples = data.get("instances")
        if samples is None and "dataframe_split" in data:
            samples = data["dataframe_split"]["data"]
        if samples is None:
            samples = _sample_features()
        X = np.array(samples)
        preds = model.predict(X)
        probas = model.predict_proba(X)
        # update gauges with the last sample
        model_prob_cammeo.set(float(probas[-1, 0]))
        model_prob_osmancik.set(float(probas[-1, 1]))
        last_predicted_class.set(1.0 if preds[-1] == classes[0] else 0.0)
        prediction_total.inc(len(X))
        result = {"predictions": preds.tolist(),
                  "probabilities": probas.tolist(),
                  "classes": classes}
        return jsonify(result)
    except Exception as e:
        prediction_errors.inc()
        return jsonify({"error": str(e)}), 500
    finally:
        prediction_latency.observe(time.perf_counter() - start)


@app.route("/health")
def health():
    inference_up.set(1)
    return jsonify({"status": "ok"})


@app.route("/")
def root():
    return f"MSML Inference Exporter — see /metrics on :{EXPORTER_PORT}"


def main() -> int:
    inference_up.set(1)
    app.run(host="0.0.0.0", port=EXPORTER_PORT, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
