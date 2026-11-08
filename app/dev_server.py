"""Run the app on this computer, with the same headers Vercel sends.

    python app/dev_server.py            # http://localhost:8000
    NEBIUS_API_KEY=... python app/dev_server.py   # with the agent

Reads .env from the repo root if it's there.
"""
from __future__ import annotations

import json
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP / "api"))

env = APP.parent / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.startswith("#") and value.strip():
            os.environ.setdefault(key.strip(), value.strip())

import ask  # noqa: E402

HEADERS = {h["key"]: h["value"] for h in json.loads((APP / "vercel.json").read_text())["headers"][0]["headers"]}
HEADERS.pop("Strict-Transport-Security")  # HTTPS only


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        for k, v in HEADERS.items():
            self.send_header(k, v)
        super().end_headers()

    def do_POST(self):
        if self.path != "/api/ask":
            return self.send_error(404)
        ask.handler.do_POST(self)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"http://localhost:{port}  (agent {'on' if os.environ.get('NEBIUS_API_KEY') else 'off: no NEBIUS_API_KEY'})")
    ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(APP))).serve_forever()
