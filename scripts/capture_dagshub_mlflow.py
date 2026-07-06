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
BASE = f"https://{DAG_USER}:{DAG_TOK}@dagshub.com/{DAG_USER}/{DAG_REPO}.mlflow"

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    ctx = browser.new_context(
        viewport={"width": 1600, "height": 1000},
        http_credentials={"username": DAG_USER, "password": DAG_TOK},
    )
    page = ctx.new_page()
    print("loading", BASE)
    try:
        page.goto(BASE, wait_until="domcontentloaded", timeout=60000)
    except Exception as e:
        print("home goto failed, falling back:", e)
        page.goto(f"https://dagshub.com/{DAG_USER}/{DAG_REPO}.mlflow/", wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(10000)
    page.screenshot(path=str(DASHBOARD), full_page=True)
    print("saved", DASHBOARD)
    # Experiments tab
    try:
        page.goto(BASE + "/#/experiments", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(8000)
        page.screenshot(path=str(ARTIFAK), full_page=True)
        print("saved", ARTIFAK)
    except Exception as e:
        print("experiments goto failed:", e)
    browser.close()
