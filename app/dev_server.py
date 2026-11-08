"""Run the app on this computer, with the same headers Vercel sends.

    python app/dev_server.py                         # http://localhost:8000
    NEBIUS_API_KEY=... python app/dev_server.py      # with the agent
"""
from __future__ import annotations

import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP / "api"))
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
    print(f"http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(APP))).serve_forever()
