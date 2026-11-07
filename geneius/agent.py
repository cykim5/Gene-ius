"""Agent loop: the LLM decides which tools to call and writes the final answer.

Uses an OpenAI-compatible endpoint (Nebius). Set NEBIUS_API_KEY in .env.
If the main model errors, it retries once with LLM_FALLBACK_MODEL.
"""
from __future__ import annotations

import json
import os

from .tools import TOOL_FUNCTIONS

SYSTEM_PROMPT = """You are GENEIUS, a copilot that helps clinicians find the genetic cause of a rare disease.

Steps:
1. Call map_to_hpo on the clinician's description.
2. Call rank_candidates with the HPO IDs.
3. Call get_evidence_paths for the top 3 genes.
4. If the top candidates have close scores, call suggest_followup and, if useful, get_mechanism.
5. Answer with a ranked list. For each gene, give the disease and explain why using only the
   evidence paths you retrieved. Say plainly when the evidence is weak.

Never invent genes, diseases, or links that the tools did not return. This is decision
support, not a diagnosis."""

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "map_to_hpo",
            "description": "Map free-text symptoms to HPO terms.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_candidates",
            "description": "Rank candidate genes and diseases for a set of HPO terms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hpo_ids": {"type": "array", "items": {"type": "string"}},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["hpo_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_evidence_paths",
            "description": "Knowledge graph paths linking the patient's HPO terms to a gene.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hpo_ids": {"type": "array", "items": {"type": "string"}},
                    "gene": {"type": "string"},
                    "max_paths": {"type": "integer", "default": 5},
                },
                "required": ["hpo_ids", "gene"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mechanism",
            "description": "Disease mechanism summary.",
            "parameters": {
                "type": "object",
                "properties": {"disease": {"type": "string"}},
                "required": ["disease"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_followup",
            "description": "Suggest one symptom question that best separates the candidate diseases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"gene": {"type": "string"}, "disease": {"type": "string"}},
                        },
                    },
                    "known_hpo_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["candidates", "known_hpo_ids"],
            },
        },
    },
]


def _client():
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["NEBIUS_API_KEY"],
        base_url=os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"),
    )


def _complete(client, messages):
    models = [os.environ.get("LLM_MODEL"), os.environ.get("LLM_FALLBACK_MODEL")]
    models = [m for m in models if m]
    if not models:
        raise RuntimeError("Set LLM_MODEL in .env")
    for i, model in enumerate(models):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, tools=TOOL_SPECS, temperature=0.2
            )
        except Exception:
            if i == len(models) - 1:
                raise


def run_agent(text: str, max_steps: int = 10, verbose: bool = False) -> dict:
    """Returns {answer, trace}. trace lists every tool call, for the demo UI."""
    client = _client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    trace = []
    for _ in range(max_steps):
        msg = _complete(client, messages).choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return {"answer": msg.content, "trace": trace}
        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
                result = TOOL_FUNCTIONS[name](**args)
            except Exception as e:
                args, result = {}, {"error": str(e)}
            trace.append({"tool": name, "args": args, "result": result})
            if verbose:
                print(f"[tool] {name}({json.dumps(args)[:120]})")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    return {"answer": "Stopped: hit the step limit before a final answer.", "trace": trace}
