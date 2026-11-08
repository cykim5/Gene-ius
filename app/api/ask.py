"""POST /api/ask: the agent explains one variant's prediction in plain language.

The browser sends only a gene, a variation_id, and a question. The evidence comes from the
server's own copy of predictions.json, never from the request, so nobody can feed the model
made-up evidence. The API key stays on the server (NEBIUS_API_KEY in Vercel's settings).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "genes"
BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.us-north1.nebius.com/v1/")
MODEL = os.environ.get("LLM_MODEL", "nvidia/Nemotron-3.5-Super-VL")
MAX_QUESTION = 400
RATE_LIMIT = 8  # requests per IP per minute (per server instance)

SYSTEM_PROMPT = """You explain a variant of uncertain significance (VUS) to a clinical genetics lab.
You get one JSON record with our model's prediction and its evidence. Use only that record.

Cover, in this order and in plain language:
A. Harmful or benign: the harmful score, the confidence, and the ACMG evidence strength.
B. How the protein breaks: the mechanism and FoldX energy if the record has them.
C. Which disease: the nearest known disease variant if the record has one.
D. Next step: the treatment strategy (always call it a hypothesis) and the assay that would settle it.

Rules:
- Never invent genes, diseases, numbers, or evidence that aren't in the record.
- If a field is empty, say it isn't available. Don't guess.
- If confidence is "unsure" or the evidence is indeterminate, say so first and lead with the assay.
- This is research and decision support, not a clinical classification.
- Under 180 words. No markdown headers."""

_GENE = re.compile(r"^[A-Z0-9]{2,10}$")
_ID = re.compile(r"^[0-9]{1,12}$")
_hits: dict[str, list[float]] = {}


def _rate_limited(ip: str) -> bool:
    now = time.time()
    recent = [t for t in _hits.get(ip, []) if now - t < 60]
    recent.append(now)
    _hits[ip] = recent
    return len(recent) > RATE_LIMIT


def _record(gene: str, variation_id: str) -> dict | None:
    path = DATA / f"{gene}.json"
    if not path.exists():
        return None
    record = next((r for r in json.loads(path.read_text()) if r.get("variation_id") == variation_id), None)
    if record is not None:
        index = json.loads((DATA.parent / "index.json").read_text())
        record = dict(record, mouse_evidence=index["genes"].get(gene, {}).get("mouse", ""))
    return record


def _call_llm(record: dict, question: str) -> str:
    body = {
        "model": MODEL,
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "2048")),
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Record:\n{json.dumps(record)}\n\nQuestion: {question}"},
        ],
    }
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['NEBIUS_API_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=50) as r:
        out = json.loads(r.read())
    text = out["choices"][0]["message"].get("content") or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def answer(payload: dict, ip: str) -> tuple[int, dict]:
    """Returns (HTTP status, JSON body). Shared by Vercel and the local dev server."""
    if _rate_limited(ip):
        return 429, {"error": "Too many questions. Try again in a minute."}
    gene = str(payload.get("gene", "")).upper()
    variation_id = str(payload.get("variation_id", ""))
    question = str(payload.get("question", "")).strip()[:MAX_QUESTION] or "Explain this variant."
    if not _GENE.match(gene) or not _ID.match(variation_id):
        return 400, {"error": "Unknown variant."}
    record = _record(gene, variation_id)
    if record is None:
        return 404, {"error": "Unknown variant."}
    if not os.environ.get("NEBIUS_API_KEY"):
        return 503, {"error": "The agent isn't connected yet. Add NEBIUS_API_KEY in the server settings."}
    try:
        return 200, {"answer": _call_llm(record, question), "model": MODEL}
    except Exception:
        return 502, {"error": "The agent didn't answer. Try again."}


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = min(int(self.headers.get("Content-Length", 0)), 4096)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError
        except ValueError:
            status, body = 400, {"error": "Bad request."}
        else:
            ip = (self.headers.get("x-forwarded-for") or self.client_address[0]).split(",")[0].strip()
            status, body = answer(payload, ip)
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send_response(405)
        self.send_header("Allow", "POST")
        self.end_headers()
