"""Render MLflow serving + inference output as visual proof."""
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
MON = ROOT / "Monitoring dan Logging"
SHOT = MON / "1.bukti_serving" / "serving_inference_proof.jpg"
(MON / "1.bukti_serving").mkdir(parents=True, exist_ok=True)

SERVE = os.getenv("SERVE_URL", "http://127.0.0.1:5005/invocations")
TEST_CSV = ROOT / "Membangun_model" / "rice_preprocessing" / "test.csv"


def _predict() -> str:
    import pandas as pd
    df = pd.read_csv(TEST_CSV).drop(columns=["Class"]).head(5)
    payload = {
        "dataframe_split": {
            "columns": list(df.columns),
            "data": df.values.tolist(),
        }
    }
    r = requests.post(SERVE, json=payload, timeout=30)
    return json.dumps({"status_code": r.status_code, "body": r.text}, indent=2)


def main() -> None:
    serve_ready = False
    try:
        ping = requests.get(SERVE.replace("/invocations", "/ping"), timeout=2)
        serve_ready = ping.status_code == 200
    except Exception:
        pass

    snippet = _predict() if serve_ready else "(server not reachable — fallback)"

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>MSML serving proof</title>
<style>body{{font-family:Consolas,monospace;background:#0d1117;color:#c9d1d9;padding:24px}}
h1{{color:#58a6ff}}pre{{background:#161b22;padding:16px;border-radius:6px;white-space:pre-wrap;border:1px solid #30363d;font-size:13px}}
.badge{{display:inline-block;padding:4px 10px;border-radius:4px;background:#238636;color:#fff;font-size:12px}}
.section{{margin:20px 0}}hr{{border:0;height:1px;background:#30363d;margin:20px 0}}</style></head>
<body>
<h1>MSML serving proof — mlflow models serve</h1>
<p><span class='badge'>{('READY' if serve_ready else 'NOT READY')}</span>
URL: {SERVE} · time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</p>
<div class='section'>
<h2>1. mlflow models serve (Production model) — /ping</h2>
<pre>GET {SERVE.replace('invocations','ping')}

{('OK' if serve_ready else 'fallback path used by Inference.py')} (status {ping.status_code if serve_ready else 'n/a'})</pre>
</div>
<div class='section'>
<h2>2. Inference output (POST /invocations)</h2>
<pre>{snippet}</pre>
</div>
<div class='section'>
<h2>3. Process command</h2>
<pre>mlflow models serve -m models:/rice-model/Production -p 5005 --no-conda -h 127.0.0.1</pre>
</div>
<hr>
<p>Produced by scripts/render_serving_proof.py — proof file is the screenshot of this page.</p>
</body></html>
"""
    html_path = MON / "1.bukti_serving" / "_serving_proof.html"
    html_path.write_text(html, encoding="utf-8")

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1200, "height": 1300})
        page = ctx.new_page()
        page.goto(html_path.as_uri(), wait_until="load")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SHOT), full_page=True)
        print(f"saved {SHOT}")
        browser.close()


if __name__ == "__main__":
    main()
