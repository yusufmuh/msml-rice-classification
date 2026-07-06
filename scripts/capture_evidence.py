"""Capture all monitoring evidence via Playwright + REST API.

Targets (paths relative to repo root):
- Membangun_model/screenshoot_dashboard.jpg       (DagsHub MLflow dashboard)
- Membangun_model/screenshoot_artifak.jpg         (DagsHub MLflow run artifacts)
- Monitoring dan Logging/1.bukti_serving.png      (MLflow serve ping page / Inference output)
- Monitoring dan Logging/4.bukti monitoring Prometheus/targets.jpg
- Monitoring dan Logging/5.bukti monitoring Grafana/dashboard.jpg
- Monitoring dan Logging/6.bukti alerting Grafana/alerts.jpg
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
MON_DIR = ROOT / "Monitoring dan Logging"
PROOF_PROM = MON_DIR / "4.bukti monitoring Prometheus"
PROOF_GRAF = MON_DIR / "5.bukti monitoring Grafana"
PROOF_ALRT = MON_DIR / "6.bukti alerting Grafana"
for p in (PROOF_PROM, PROOF_GRAF, PROOF_ALRT, MON_DIR / "1.bukti_serving"):
    p.mkdir(parents=True, exist_ok=True)


def _auth_url_for_dagshub() -> str:
    u = os.getenv("DAGSHUB_USERNAME", "")
    t = os.getenv("DAGSHUB_TOKEN", "")
    repo = os.getenv("DAGSHUB_REPO", "")
    return f"https://{u}:{t}@dagshub.com/{u}/{repo}.mlflow"


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed. Run: pip install playwright && playwright install chromium", file=sys.stderr)
        return 1

    dagshub_url = _auth_url_for_dagshub()
    grafana_url = os.getenv("GRAFANA_URL", "").strip()
    grafana_token = os.getenv("GRAFANA_TOKEN", "").strip()
    prom_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1600, "height": 900})
        page = ctx.new_page()

        # 1. DagsHub MLflow dashboard
        try:
            page.goto(f"{dagshub_url}/#/experiments/1", wait_until="load")
            page.wait_for_timeout(8000)
            page.screenshot(path=str(ROOT / "Membangun_model" / "screenshoot_dashboard.jpg"),
                            full_page=True)
            print("saved screenshoot_dashboard.jpg")
        except Exception as e:
            print(f"dashboard shot failed: {e}", file=sys.stderr)

        # 2. DagsHub MLflow artifacts (latest run)
        try:
            page.goto(f"{dagshub_url}/#/experiments/2", wait_until="load")
            page.wait_for_timeout(5000)
            page.screenshot(path=str(ROOT / "Membangun_model" / "screenshoot_artifak.jpg"),
                            full_page=True)
            print("saved screenshoot_artifak.jpg")
        except Exception as e:
            print(f"artifact shot failed: {e}", file=sys.stderr)

        # 3. MLflow serving: ping + Invoke. Screenshot of curl via libreoffice? Use playwright UI proxy.
        try:
            page.goto("http://localhost:5005/", wait_until="load", timeout=5000)
            page.screenshot(path=str(MON_DIR / "1.bukti_serving" / "mlflow_serve_root.png"),
                            full_page=True)
        except Exception as e:
            print(f"MLflow serve UI shot skipped ({e}).")

        # 4. Prometheus targets
        try:
            page.goto(f"{prom_url}/targets", wait_until="load", timeout=15000)
            page.wait_for_timeout(2000)
            page.screenshot(path=str(PROOF_PROM / "prometheus_targets.jpg"),
                            full_page=True)
            page.goto(f"{prom_url}/graph?g0.expr=prediction_total&g0.tab=0",
                      wait_until="load", timeout=15000)
            page.wait_for_timeout(3000)
            page.screenshot(path=str(PROOF_PROM / "prometheus_predictions.jpg"),
                            full_page=True)
            print("saved prometheus screenshots")
        except Exception as e:
            print(f"prometheus shot skipped: {e}", file=sys.stderr)

        # 5. Grafana dashboards & alerts (cloud or local)
        if grafana_url and grafana_token:
            page.set_extra_http_headers({"Authorization": f"Bearer {grafana_token}"})
            try:
                page.goto(f"{grafana_url}/dashboards", wait_until="load", timeout=30000)
                page.wait_for_timeout(5000)
                page.screenshot(path=str(PROOF_GRAF / "grafana_dashboards.jpg"),
                                full_page=True)
                dash_uid = "msml-rice-dashboard"
                page.goto(f"{grafana_url}/d/{dash_uid}", wait_until="load", timeout=30000)
                page.wait_for_timeout(6000)
                page.screenshot(path=str(PROOF_GRAF / "grafana_dashboard.jpg"),
                                full_page=True)
                page.goto(f"{grafana_url}/alerting/list", wait_until="load", timeout=30000)
                page.wait_for_timeout(4000)
                page.screenshot(path=str(PROOF_ALRT / "grafana_alerts.jpg"),
                                full_page=True)
                print("saved grafana screenshots")
            except Exception as e:
                print(f"grafana shots skipped: {e}", file=sys.stderr)
        else:
            print("GRAFANA_URL/TOKEN not configured — skipping cloud grafana shots.")

        browser.close()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
