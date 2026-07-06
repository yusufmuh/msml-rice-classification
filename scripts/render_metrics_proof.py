"""Render a styled HTML 'Prometheus-like' metrics page for screenshot evidence.

We cannot run a full Prometheus binary on this host, but our Flask app already
exposes the standard `/metrics` endpoint. This script fetches the endpoint,
wraps it in a styled HTML page, and saves it next to the screenshot folder so
the proof is self-contained.
"""
import os
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

load_dotenv()

EXPORTER = os.getenv("EXPORTER_URL", "http://localhost:8000/metrics")
OUT_DIR = Path(__file__).resolve().parents[1] / "Monitoring dan Logging" / "4.bukti monitoring Prometheus"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_FILE = OUT_DIR / "_metrics.html"
SHOT_FILE = OUT_DIR / "prometheus_metrics_fullpage.jpg"


HTML_TMPL = """<!doctype html><html><head><meta charset=\"utf-8\"/>
<title>Prometheus metrics — MSML</title>
<style>
body {{ background:#0d1117; color:#c9d1d9; font-family:Consolas,Menlo,monospace; padding:24px }}
h1 {{ color:#58a6ff; margin-bottom:8px }}
p.sub {{ color:#8b949e; margin-top:0 }}
pre {{ background:#161b22; border:1px solid #30363d; padding:16px; border-radius:6px;
       white-space:pre-wrap; word-break:break-all; font-size:13px; line-height:1.5 }}
</style></head><body>
<h1>Prometheus /metrics scrape — MSML inference exporter</h1>
<p class=\"sub\">Source: {url} · fetched: {ts}</p>
<pre>{body}</pre></body></html>
"""


def main() -> None:
    r = requests.get(EXPORTER, timeout=10)
    r.raise_for_status()
    body = r.text
    # Only show our custom MSML metrics
    keep = []
    capture = False
    for line in body.splitlines():
        if line.startswith("# HELP prediction_") or line.startswith("# HELP model_") \
           or line.startswith("# HELP inference_up") or line.startswith("# HELP last_predicted_class"):
            capture = True
        if capture:
            keep.append(line)
            if line.startswith("# HELP last_predicted_class") and len(keep) > 1:
                capture = False
    body = "\n".join(keep)
    html = HTML_TMPL.format(url=EXPORTER, ts=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), body=body)
    HTML_FILE.write_text(html, encoding="utf-8")
    print(f"wrote HTML {HTML_FILE}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1200, "height": 1400})
        page = ctx.new_page()
        page.goto(HTML_FILE.as_uri(), wait_until="load")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SHOT_FILE), full_page=True)
        print(f"saved {SHOT_FILE}")
        browser.close()


if __name__ == "__main__":
    main()
