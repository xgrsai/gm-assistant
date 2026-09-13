"""Versioned Jinja prompt templates; the sha256 of each template goes into every run record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPTS_DIR = Path(__file__).parent / "prompts"
SUFFIX = ".md.j2"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)  # jinja's tojson escapes Cyrillic into \u sequences


class PromptSet:
    def __init__(self, directory: Path = PROMPTS_DIR):
        self.dir = directory
        self.env = Environment(loader=FileSystemLoader(str(directory), encoding="utf-8"),
                               undefined=StrictUndefined, keep_trailing_newline=True, autoescape=False,
                               trim_blocks=True, lstrip_blocks=True)
        self.env.filters["json"] = _json

    def render(self, name: str, **kw) -> str:
        return self.env.get_template(name + SUFFIX).render(**kw).strip()

    def hash(self, name: str) -> str:
        # Hash LF-normalised source so Windows and Linux checkouts produce the same value.
        src = (self.dir / (name + SUFFIX)).read_text(encoding="utf-8").replace("\r\n", "\n")
        return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
