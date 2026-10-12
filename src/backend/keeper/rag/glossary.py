"""Query term normalization: append the English names of game terms found in a Ukrainian query.

The knowledge base is English; players write Ukrainian, often with Ukrainian names for game concepts
("розсудливість" for Sanity). Before retrieval, annotate() appends the English terms of every glossary entry
matched in the query, e.g. "Що дає екстремальний успіх під час атаки?" ->
"Що дає екстремальний успіх під час атаки? (Extreme success, attack)". No LLM call; deterministic.
The glossary is a YAML file (see data/glossary/coc7e-uk-en.yaml).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml


class Glossary:
    def __init__(self, entries: list[dict]):
        self.entries = [(e["en"], [re.compile(r"(?<!\w)(?:" + p + r")", re.IGNORECASE) for p in e["uk"]])
                        for e in entries]

    @classmethod
    def load(cls, path: Path) -> "Glossary":
        return cls(yaml.safe_load(path.read_text(encoding="utf-8"))["terms"])

    def terms(self, text: str) -> list[str]:
        """English terms whose Ukrainian patterns occur in text, in glossary order, skipping ones already present."""
        found = []
        low = text.lower()
        for en, pats in self.entries:
            if en.lower() in low or en in found:
                continue
            if any(p.search(text) for p in pats):
                found.append(en)
        return found

    def annotate(self, text: str) -> str:
        terms = self.terms(text)
        return f"{text} ({', '.join(terms)})" if terms else text


@lru_cache(maxsize=4)
def load_glossary(path: str) -> Glossary:
    return Glossary.load(Path(path))
