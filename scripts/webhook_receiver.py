"""Tiny local webhook receiver to prove Grafana alert notifications are delivered.

Listens on :8010/webhook, logs every incoming POST body (Grafana's alert JSON payload)
to webhook_notifications.log with a timestamp, and prints it to stdout.
"""
import datetime
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parents[1] / "Monitoring dan Logging" / "webhook_notifications.log"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        ts = datetime.datetime.now().isoformat()
        try:
            pretty = json.dumps(json.loads(body), indent=2, ensure_ascii=False)
        except Exception:
            pretty = body
        entry = f"\n===== NOTIFICATION RECEIVED {ts} =====\n{pretty}\n"
        print(entry)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(entry)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    print(f"Webhook receiver listening on :8010, logging to {LOG_PATH}")
    HTTPServer(("127.0.0.1", 8010), Handler).serve_forever()
