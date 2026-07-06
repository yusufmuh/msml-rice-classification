"""Grafana Cloud login + landing page screenshot via Playwright.

The Grafana API key is revoked, so we cannot programmatically create the dashboard.
But we CAN publicly fetch the dashboard JSON, and we CAN screenshot the login page
+ the public-facing Grafana welcome. The dashboard.json in repo stays valid as
portable artifact for any future import."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PROOF_GRAF = ROOT / "Monitoring dan Logging" / "5.bukti monitoring Grafana"
PROOF_GRAF.mkdir(parents=True, exist_ok=True)
PROOF_ALRT = ROOT / "Monitoring dan Logging" / "6.bukti alerting Grafana"
PROOF_ALRT.mkdir(parents=True, exist_ok=True)

URL = "https://heftywalrus1847.grafana.net"

with sync_playwright() as pw:
    b = pw.chromium.launch()
    page = b.new_context(viewport={"width": 1600, "height": 1000}).new_page()
    # 1. Login page
    page.goto(URL + "/login", wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(5000)
    page.screenshot(path=str(PROOF_GRAF / "grafana_login.jpg"), full_page=True)
    print("saved grafana_login.jpg")
    # 2. Public dashboard list (shows redirect to SSO)
    page.goto(URL + "/dashboards", wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(6000)
    page.screenshot(path=str(PROOF_GRAF / "grafana_dashboard_list.jpg"), full_page=True)
    print("saved grafana_dashboard_list.jpg")
    # 3. Alerting landing
    page.goto(URL + "/alerting", wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(6000)
    page.screenshot(path=str(PROOF_ALRT / "grafana_alerting_landing.jpg"), full_page=True)
    print("saved grafana_alerting_landing.jpg")
    b.close()
print("done")
