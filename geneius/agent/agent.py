"""Agent loop: plan, call tools, check the evidence, loop or stop, answer.

Uses an OpenAI-compatible endpoint (Nebius). Set NEBIUS_API_KEY in .env.
If the main model errors, it retries once with LLM_FALLBACK_MODEL.

TODO: GNN-guided search (score neighbors with gnn_score, follow only the top ones)
and the grounding checker (every claim must cite an edge a tool returned).
"""
from __future__ import annotations

import json
import os

from .tools import TOOL_FUNCTIONS

SYSTEM_PROMPT = """You are an agent that helps interpret variants of uncertain significance (VUS).

For a variant, find out:
A. Is it harmful or benign?
B. How does it break the protein (gain of function, loss of function, dominant-negative, misfolding)?
C. Which disease and symptoms follow? Eye-only or syndromic?
D. Which treatment strategy fits?

How to work:
1. Plan the steps.
2. Call tools. Start with gnn_predict, then structural_neighbors and find_paths for evidence.
3. Check whether the evidence is strong and consistent.
4. If it's weak or conflicting, gather more. If it's still unclear, say you're unsure and call suggest_assay.
5. Answer in plain language. Every claim must cite a path or neighbor a tool returned.

For a question about a gene, its diseases, or symptoms, use find_entity, expand, and find_paths.

Never invent genes, diseases, variants, or links the tools did not return.
A result with "stub": true is a placeholder, not evidence. Don't cite it; say that tool isn't built yet.
Treatments are hypotheses. Research use only, not a diagnosis."""

DEFAULT_BASE_URL = "https://api.tokenfactory.us-north1.nebius.com/v1/"
DEFAULT_MODEL = "nvidia/Nemotron-3.5-Super-VL"
# Reasoning models spend part of the budget thinking before they answer or call a tool.
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2048"))

S = {"type": "string"}
LIST_S = {"type": "array", "items": S}


def _spec(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


TOOL_SPECS = [
    _spec("map_symptoms", "Map free-text symptoms to HPO terms.", {"text": S}, ["text"]),
    _spec("find_entity", "Look up a gene, variant, disease, or symptom.", {"query": S}, ["query"]),
    _spec("expand", "Neighbors of a node, optionally filtered by node type.",
          {"node_id": S, "node_type": S}, ["node_id"]),
    _spec("find_paths", "Knowledge graph paths linking HPO terms to a gene.",
          {"hpo_ids": LIST_S, "gene": S, "max_paths": {"type": "integer"}}, ["hpo_ids", "gene"]),
    _spec("structural_neighbors", "Known variants nearby in 3D and at matching residues in related proteins.",
          {"variant": S}, ["variant"]),
    _spec("gnn_predict", "GNN predictions A to D (harmful, mechanism, disease, treatment) with confidence.",
          {"variant": S}, ["variant"]),
    _spec("gnn_score", "GNN score for a proposed link between two nodes.", {"source": S, "target": S},
          ["source", "target"]),
    _spec("evidence_by_source", "How much each data source contributed to the prediction.",
          {"variant": S}, ["variant"]),
    _spec("suggest_treatment", "Treatment strategy hypothesis from mechanism, gene, and disease.",
          {"mechanism": S, "gene": S, "disease": S}, ["mechanism"]),
    _spec("suggest_assay", "Experiment that would settle an uncertain case.", {"gene": S}, ["gene"]),
    _spec("evidence_report", "One-page clinician-style evidence report.", {"variant": S}, ["variant"]),
    _spec("show_structure", "Send the variant and its neighbors to the 3D viewer.", {"variant": S}, ["variant"]),
]


def _client():
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["NEBIUS_API_KEY"],
        base_url=os.environ.get("NEBIUS_BASE_URL") or DEFAULT_BASE_URL,
    )


def _complete(client, messages):
    models = [m for m in (os.environ.get("LLM_MODEL") or DEFAULT_MODEL, os.environ.get("LLM_FALLBACK_MODEL")) if m]
    for i, model in enumerate(models):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, tools=TOOL_SPECS, temperature=0.2, max_tokens=MAX_TOKENS
            )
        except Exception:
            if i == len(models) - 1:
                raise


def run_agent(question: str, max_steps: int = 12, verbose: bool = False) -> dict:
    """Returns {answer, trace}. trace lists every tool call, for the agent panel in the app."""
    client = _client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
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
