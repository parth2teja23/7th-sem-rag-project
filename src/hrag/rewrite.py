from __future__ import annotations

import re

from hrag.llm import OllamaUnavailable, chat

_SYSTEM = (
    "You expand search queries. Reply with at most 20 comma-separated keywords or "
    "synonyms that would help retrieve relevant documents. No prose, no explanation."
)


def expand_query(q: str) -> str:
    try:
        reply = chat(q, system=_SYSTEM)
    except OllamaUnavailable:
        return q
    words = [w for w in re.split(r"[,\s]+", reply.strip()) if w][:20]
    return q + " " + " ".join(words) if words else q
