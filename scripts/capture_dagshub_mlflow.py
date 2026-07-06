"""Capture DagsHub MLflow dashboard + artifacts via Playwright using HTTP basic auth."""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "Membangun_model" / "screenshoot_dashboard.jpg"
ARTIFAK = ROOT / "Membangun_model" / "screenshoot_artifak.jpg"

DAG_USER = os.environ["DAGSHUB_USERNAME"]
DAG_TOK = os.environ["DAGSHUB_TOKEN"]
DAG_REPO = os.environ["DAGSHUB_REPO"]
# IMPORTANT: no credentials embedded in the URL itself. If window.location contains
# user:token@host, the SPA's relative fetch() calls inherit those credentials and the
# browser's fetch() API refuses to build the request ("URL that includes credentials").
# Using an HTTP-auth context keeps the URL clean while still transparently attaching
# the Authorization header for every request (including XHR/fetch) on this origin.
BASE = f"https://dagshub.com/{DAG_USER}/{DAG_REPO}.mlflow"

BEST_EXPERIMENT_ID = os.getenv("DAGSHUB_BEST_EXPERIMENT_ID", "2")
BEST_RUN_ID = os.getenv("DAGSHUB_BEST_RUN_ID", "2378bbee6212411094e09e79c7a59aca")
BASELINE_EXPERIMENT_ID = os.getenv("DAGSHUB_BASELINE_EXPERIMENT_ID", "1")
BASELINE_RUN_ID = os.getenv("DAGSHUB_BASELINE_RUN_ID", "1845bffe10b44f6ebeca18fe10c71f2c")

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    ctx = browser.new_context(
        viewport={"width": 1600, "height": 1000},
        http_credentials={"username": DAG_USER, "password": DAG_TOK},
    )
    page = ctx.new_page()

    exp_url = f"{BASE}/#/experiments/{BEST_EXPERIMENT_ID}"
    print("loading", exp_url)
    page.goto(exp_url, wait_until="domcontentloaded", timeout=60000)
    try:
        page.wait_for_selector("text=optuna_sweep", timeout=45000)
    except Exception as e:
        print("runs table selector not found (continuing):", e)
    page.wait_for_timeout(6000)
    page.screenshot(path=str(DASHBOARD), full_page=True)
    print("saved", DASHBOARD)

    run_url = f"{BASE}/#/experiments/{BASELINE_EXPERIMENT_ID}/runs/{BASELINE_RUN_ID}/artifacts"
    print("loading", run_url)
    try:
        page.goto(run_url, wait_until="domcontentloaded", timeout=60000)
        try:
            page.wait_for_selector("text=confusion_matrix.png", timeout=45000)
            page.click("text=confusion_matrix.png")
            page.wait_for_timeout(3000)
        except Exception as e:
            print("artifacts tree selector not found (continuing):", e)
        page.wait_for_timeout(5000)
        page.screenshot(path=str(ARTIFAK), full_page=True)
        print("saved", ARTIFAK)
    except Exception as e:
        print("run page goto failed:", e)
    browser.close()
