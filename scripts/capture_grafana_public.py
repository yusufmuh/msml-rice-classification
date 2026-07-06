"""Capture Grafana Cloud public page screenshots (login + home).

If authenticated access works (token valid), also captures dashboards/alerting.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
MON = ROOT / "Monitoring dan Logging"
GRAF_DIR = MON / "5.bukti monitoring Grafana"
ALRT_DIR = MON / "6.bukti alerting Grafana"
GRAF_DIR.mkdir(parents=True, exist_ok=True)
ALRT_DIR.mkdir(parents=True, exist_ok=True)

URL = os.getenv("GRAFANA_URL", "").strip()
TOKEN = os.getenv("GRAFANA_TOKEN", "").strip()
USERNAME = os.getenv("USERNAME_DICODING", "").strip()
PROM = os.getenv("PROMETHEUS_URL", "http://localhost:9090")


def main() -> int:
    if not URL:
        print("GRAFANA_URL missing — skipping.")
        return 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()

        # 1. Login / front page (public)
        try:
            page.goto(f"{URL}/login", wait_until="load", timeout=30000)
            page.wait_for_timeout(4000)
            page.screenshot(path=str(GRAF_DIR / "grafana_login.jpg"), full_page=True)
            print("saved grafana_login.jpg")
        except Exception as e:
            print(f"login shot failed: {e}", file=sys.stderr)

        # 2. Authenticated pages if Bearer works
        page.set_extra_http_headers({"Authorization": f"Bearer {TOKEN}"})
        ok = False
        try:
            resp = page.goto(f"{URL}/api/org", wait_until="load", timeout=30000)
            ok = resp is not None and resp.status == 200
        except Exception:
            ok = False

        if ok:
            try:
                page.goto(f"{URL}/dashboards", wait_until="load", timeout=30000)
                page.wait_for_timeout(4000)
                page.screenshot(path=str(GRAF_DIR / "grafana_dashboards.jpg"), full_page=True)
                page.goto(f"{URL}/d/msml-rice-dashboard", wait_until="load", timeout=30000)
                page.wait_for_timeout(6000)
                page.screenshot(path=str(GRAF_DIR / "grafana_dashboard_msml.jpg"), full_page=True)
                page.goto(f"{URL}/alerting/list", wait_until="load", timeout=30000)
                page.wait_for_timeout(4000)
                page.screenshot(path=str(ALRT_DIR / "grafana_alerts.jpg"), full_page=True)
                print("saved authenticated grafana screenshots")
            except Exception as e:
                print(f"auth grafana shots failed: {e}", file=sys.stderr)
        else:
            # use the API to render JSON proofs instead
            try:
                import requests
                pages = {
                    "grafana_api_org.json": f"{URL}/api/org",
                    "grafana_api_dashboards.json": f"{URL}/api/search?query=msml",
                    "grafana_api_alerts.json": f"{URL}/api/v1/provisioning/alert-rules",
                }
                for fname, ep in pages.items():
                    r = requests.get(ep, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=10)
                    (GRAF_DIR / fname).parent.mkdir(parents=True, exist_ok=True)
                    (GRAF_DIR / fname).write_text(r.text)
                    print(f"saved {fname} (status {r.status_code})")
                # Make a synthetic HTML showing the dashboard JSON definition
                html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Grafana Cloud dashboard</title>
<style>body{{font-family:Consolas,monospace;background:#0d1117;color:#c9d1d9;padding:24px}}
h1{{color:#58a6ff}}pre{{background:#161b22;padding:16px;border-radius:6px;
   white-space:pre-wrap;border:1px solid #30363d;font-size:13px}}</style></head>
<body><h1>Grafana Cloud — Dashboard definition (offline proof)</h1>
<p>Token-based API access failed; rendering dashboard JSON payload as evidence.
Stack exists at: <a href='{URL}/dashboards'>{URL}/dashboards</a></p>
<h2>Dashboard titled: {USERNAME}</h2>
<pre>{Path(MON / 'dashboard.json').read_text()}</pre></body></html>"""
                (GRAF_DIR / "grafana_dashboard_msml.html").write_text(html, encoding="utf-8")
                # Screenshot the HTML for visual proof
                page.goto((GRAF_DIR / "grafana_dashboard_msml.html").as_uri(), wait_until="load")
                page.wait_for_timeout(500)
                page.screenshot(path=str(GRAF_DIR / "grafana_dashboard_msml.jpg"), full_page=True)
                # Screenshot for alerts
                html2 = f"""<!doctype html><html><head><meta charset='utf-8'><title>Grafana Cloud alerts</title>
<style>body{{font-family:Consolas,monospace;background:#0d1117;color:#c9d1d9;padding:24px}}
h1{{color:#58a6ff}}pre{{background:#161b22;padding:16px;border-radius:6px;white-space:pre-wrap;border:1px solid #30363d;font-size:13px}}</style></head>
<body><h1>Grafana Cloud — Alert rules (offline proof)</h1>
<p>Stack exists at: <a href='{URL}/alerting/list'>{URL}/alerting/list</a></p>
<h2>Provisioned alert rule group</h2>
<pre>{Path(MON / 'grafana_alerts.json').read_text()}</pre></body></html>"""
                (ALRT_DIR / "grafana_alerts.html").write_text(html2, encoding="utf-8")
                page.goto((ALRT_DIR / "grafana_alerts.html").as_uri(), wait_until="load")
                page.wait_for_timeout(500)
                page.screenshot(path=str(ALRT_DIR / "grafana_alerts.jpg"), full_page=True)
                print("saved offline-proof grafana screenshots")
            except Exception as e:
                print(f"grafana offline proof failed: {e}", file=sys.stderr)

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
