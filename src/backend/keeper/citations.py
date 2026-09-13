"""[R#] citation markers in Keeper text, validated against the chunks that were actually retrieved."""

from __future__ import annotations

import re

BRACKET = re.compile(r"\[(\s*R\d+(?:\s*[,;]\s*R?\d+)*\s*)\]")
NUMBER = re.compile(r"\d+")


def cited_markers(text: str) -> list[int]:
    """Markers in order of first appearance: '[R1]', '[R1, R3]', '[R2; R4]' -> [1, 3, 2, 4]."""
    seen: list[int] = []
    for group in BRACKET.findall(text or ""):
        for n in NUMBER.findall(group):
            if int(n) not in seen:
                seen.append(int(n))
    return seen


def validate(text: str, hits: list[dict]) -> tuple[list[dict], list[int]]:
    by_marker = {h["marker"]: h for h in hits}
    valid, invalid = [], []
    for marker in cited_markers(text):
        hit = by_marker.get(marker)
        if hit is None:
            invalid.append(marker)
            continue
        valid.append({k: hit.get(k) for k in ("marker", "chunk_id", "source", "section", "source_type", "score",
                                              "page", "excerpt")})
    return valid, invalid
