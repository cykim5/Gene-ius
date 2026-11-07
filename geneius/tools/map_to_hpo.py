"""map_to_hpo: free-text symptoms -> HPO terms.

Current version: offline dictionary matcher over HPO names + exact/layperson synonyms
(longest phrase wins, simple plural handling). Fast and needs no API key.

TODO (Sri): add the embedding matcher (Qwen3-Embedding-8B) for phrasings that aren't
in HPO verbatim, e.g. "can't walk straight" -> Ataxia. Keep this matcher as the fallback.
Known gap: negation is ignored ("no seizures" still maps to Seizure).
"""
from __future__ import annotations

import re
from functools import lru_cache

from .. import config
from ..data import get_hpo
from . import stubs

_WORD = re.compile(r"[A-Za-z0-9]+")
_MAX_PHRASE = 10


def _singular(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _key(phrase: str) -> str:
    return " ".join(_singular(w.lower()) for w in _WORD.findall(phrase))


@lru_cache(maxsize=1)
def _index() -> tuple[dict[str, str], int]:
    hpo = get_hpo()
    idx: dict[str, str] = {}
    for t, name in sorted(hpo.names.items()):
        if not hpo.is_phenotypic(t):
            continue
        for phrase in [name, *hpo.synonyms.get(t, [])]:
            k = _key(phrase)
            if k:
                idx.setdefault(k, t)
    longest = max((len(k.split()) for k in idx), default=1)
    return idx, min(longest, _MAX_PHRASE)


def map_to_hpo(text: str) -> list[dict]:
    """Returns [{hpo_id, label, matched_text}] in the order they appear in the text."""
    if config.use_stubs():
        return [dict(x) for x in stubs.MAP_TO_HPO]
    idx, maxlen = _index()
    hpo = get_hpo()
    spans = list(_WORD.finditer(text))
    toks = [_singular(m.group().lower()) for m in spans]
    out, seen, i = [], set(), 0
    while i < len(toks):
        for n in range(min(maxlen, len(toks) - i), 0, -1):
            key = " ".join(toks[i:i + n])
            if n == 1 and len(key) < 4:  # skip tiny single-word matches
                continue
            t = idx.get(key)
            if t:
                if t not in seen:
                    seen.add(t)
                    out.append({
                        "hpo_id": t,
                        "label": hpo.label(t),
                        "matched_text": text[spans[i].start():spans[i + n - 1].end()],
                    })
                i += n
                break
        else:
            i += 1
    return out
