"""Capture a comprehensive set of DagsHub (online) MLflow screenshots via Playwright HTTP auth.

Produces, into Membangun_model/:
- mlflow_dagshub_experiments_list.jpg   (all 3 experiments, proves online tracking)
- mlflow_dagshub_tuning_run_overview.jpg (manual-logging run, online)
- mlflow_dagshub_tuning_run_artifacts.jpg (autolog+2 extra artifacts, online)
- mlflow_dagshub_model_registry.jpg     (Production stage versions, online)
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Membangun_model"

DAG_USER = os.environ["DAGSHUB_USERNAME"]
DAG_TOK = os.environ["DAGSHUB_TOKEN"]
DAG_REPO = os.environ["DAGSHUB_REPO"]
BASE = f"https://dagshub.com/{DAG_USER}/{DAG_REPO}.mlflow"

# IDs discovered via MlflowClient against the DagsHub tracking URI.
EXPERIMENTS_URL = f"{BASE}/#/experiments/2"  # rice_classification_tuning list (shows sidebar w/ all exps)
TUNING_RUN_ID = os.environ.get("DAGSHUB_TUNING_RUN_ID", "ae98d4644ac04c2f828ddcda3a21ebd0")
TUNING_EXP_ID = "2"
MODEL_NAME = "rice-model"

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    ctx = browser.new_context(
        viewport={"width": 1600, "height": 1000},
        http_credentials={"username": DAG_USER, "password": DAG_TOK},
    )
    page = ctx.new_page()

    # 1. Experiments list (sidebar shows all experiments -> proves multiple experiments tracked online)
    print("loading experiments list...")
    page.goto(EXPERIMENTS_URL, wait_until="domcontentloaded", timeout=60000)
    try:
        page.wait_for_selector("text=optuna_sweep", timeout=45000)
    except Exception as e:
        print("warn:", e)
    page.wait_for_timeout(4000)
    page.screenshot(path=str(OUT / "mlflow_dagshub_experiments_list.jpg"), full_page=True)
    print("saved experiments list")

    # 2. Tuning run overview (manual logging metrics)
    run_url = f"{BASE}/#/experiments/{TUNING_EXP_ID}/runs/{TUNING_RUN_ID}"
    print("loading run overview...", run_url)
    page.goto(run_url, wait_until="domcontentloaded", timeout=60000)
    try:
        page.wait_for_selector("text=Parameters", timeout=45000)
    except Exception as e:
        print("warn:", e)
    page.wait_for_timeout(4000)
    page.screenshot(path=str(OUT / "mlflow_dagshub_tuning_run_overview.jpg"), full_page=True)
    print("saved run overview")

    # 3. Tuning run artifacts (should show confusion_matrix_tuned.png + optuna_optimization_history.png)
    art_url = f"{run_url}/artifacts"
    print("loading run artifacts...", art_url)
    page.goto(art_url, wait_until="domcontentloaded", timeout=60000)
    try:
        page.wait_for_selector("text=optuna_optimization_history.png", timeout=45000)
    except Exception as e:
        print("warn (artifacts):", e)
    page.wait_for_timeout(4000)
    page.screenshot(path=str(OUT / "mlflow_dagshub_tuning_run_artifacts.jpg"), full_page=True)
    print("saved run artifacts")

    # 4. Model registry (Production stage)
    model_url = f"{BASE}/#/models/{MODEL_NAME}"
    print("loading model registry...", model_url)
    page.goto(model_url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(3000)
    # Dismiss the "New Model Registry UI" promo modal if present.
    try:
        page.click("text=Learn more >> xpath=../..//button[contains(., '×') or contains(@aria-label,'Close')]", timeout=3000)
    except Exception:
        pass
    try:
        page.keyboard.press("Escape")
    except Exception:
        pass
    page.wait_for_timeout(1000)
    # Toggle off the new registry UI to reveal the classic Stage column.
    try:
        page.click("text=New model registry UI", timeout=5000)
        page.wait_for_timeout(1000)
        page.click("text=Disable", timeout=5000)
        page.wait_for_timeout(2000)
    except Exception as e:
        print("warn (toggle registry UI):", e)
    try:
        page.wait_for_selector("text=Production", timeout=20000)
    except Exception as e:
        print("warn (registry):", e)
    page.wait_for_timeout(2000)
    page.screenshot(path=str(OUT / "mlflow_dagshub_model_registry.jpg"), full_page=True)
    print("saved model registry")

    browser.close()

print("DONE")
