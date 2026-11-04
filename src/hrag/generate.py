from __future__ import annotations

import re
from dataclasses import dataclass

from hrag.llm import chat
from hrag.types import Doc

_SYSTEM = (
    "Answer the question in a short phrase or one sentence, using only the numbered passages. "
    "Cite the supporting passages like [1] or [2][3]. "
    "If the passages do not support an answer, reply exactly: not in context"
)


@dataclass
class Answer:
    text: str
    cited: list[str]


def generate_answer(question: str, passages: list[Doc]) -> Answer:
    block = "\n".join(f"[{i}] ({d.title}) {d.text}" for i, d in enumerate(passages, 1))
    prompt = f"Passages:\n{block}\n\nQuestion: {question}\nAnswer:"
    text = chat(prompt, system=_SYSTEM).strip()
    cited: list[str] = []
    for m in re.finditer(r"\[(\d+)\]", text):
        n = int(m.group(1))
        if 1 <= n <= len(passages) and passages[n - 1].id not in cited:
            cited.append(passages[n - 1].id)
    return Answer(text=text, cited=cited)
