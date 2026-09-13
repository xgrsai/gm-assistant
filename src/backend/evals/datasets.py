"""Gold evaluation datasets: schemas, loading, hashing and validation against the corpus and the scenario sheet.

    cd src/backend
    uv run python -m evals.datasets                       # validate data/eval/coc7e
    uv run python -m evals.datasets --dir ../../data/eval/coc7e

Retrieval gold labels are SECTIONS (source title + heading path), not chunk ids: chunk ids change with the chunk
size and overlap, sections survive re-chunking, so the same gold set serves the chunking ablations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from keeper.game.coc7e.rules import CHARACTERISTICS
from keeper.rag.corpus import Chunk, load_corpus
from keeper.scenarios import ScenarioRepo
from server.settings import REPO_ROOT

DATASET_DIR = REPO_ROOT / "data" / "eval" / "coc7e"
CORPUS_DIR = REPO_ROOT / "data" / "open" / "coc7e"
FILES = {"rules_qa": "rules_qa.v1.jsonl", "conflict_probes": "conflict_probes.v1.jsonl",
         "adjudication": "adjudication.v1.jsonl"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GoldSection(_Strict):
    source: str       # front-matter source_title
    section: str      # section_path as produced by keeper.rag.corpus

    def key(self) -> tuple[str, str]:
        return (self.source, self.section)


class RulesQAItem(_Strict):
    id: str = Field(pattern=r"^qa-\d{3}$")
    category: str
    query_type: Literal["lookup", "paraphrase", "multi_hop", "numeric"]
    question_en: str
    question_uk: str
    answerable: bool
    reference_answer: str                   # English, 1-3 sentences; for unanswerable: what a good answer does
    key_facts: list[str] = []               # atomic English facts the answer must state (empty if unanswerable)
    gold_sections: list[GoldSection] = []   # sections that contain the answer (empty if unanswerable)

    @model_validator(mode="after")
    def _consistent(self):
        if self.answerable and (not self.key_facts or not self.gold_sections):
            raise ValueError("answerable items need key_facts and gold_sections")
        if not self.answerable and (self.key_facts or self.gold_sections):
            raise ValueError("unanswerable items must not have key_facts or gold_sections")
        return self


class ConflictProbe(_Strict):
    """A knowledge conflict: the passage in gold_section is edited (find -> replace) before it is shown to the model."""

    id: str = Field(pattern=r"^cp-\d{3}$")
    gold_section: GoldSection
    perturbation: Literal["numeric", "negation", "entity"]
    familiarity: Literal["well_known", "obscure"]
    question_en: str
    question_uk: str
    original_fact: str      # what the unedited passage says, as the answer to the question
    perturbed_fact: str     # what the edited passage says
    find: str               # exact substring of the original chunk text (must occur exactly once)
    replace: str


RollType = Literal["skill", "characteristic", "luck", "sanity", "none"]
Difficulty = Literal["regular", "hard", "extreme"]


class ExpectedAdjudication(_Strict):
    kind: Literal["in_fiction", "rules_question", "meta"]
    needs_roll: bool
    roll_type: RollType
    allowed_targets: list[str] = []          # acceptable exact sheet names (skill or characteristic)
    allowed_difficulties: list[Difficulty] = []   # empty = any difficulty is acceptable
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self):
        if self.needs_roll != (self.roll_type != "none"):
            raise ValueError("needs_roll must be true exactly when roll_type is not 'none'")
        if self.roll_type in ("skill", "characteristic") and not self.allowed_targets:
            raise ValueError("skill/characteristic rolls need allowed_targets")
        if self.roll_type in ("none", "luck", "sanity") and self.allowed_targets:
            raise ValueError(f"roll_type {self.roll_type} takes no targets")
        if self.kind != "in_fiction" and self.needs_roll:
            raise ValueError("only in_fiction turns can need a roll")
        return self


class AdjudicationItem(_Strict):
    id: str = Field(pattern=r"^adj-\d{3}$")
    group: str = Field(pattern=r"^adj-g\d{2,3}$")   # one clean item + its injection variants
    variant: Literal["clean", "injection"]
    injection_type: Literal["none", "claimed_success", "claimed_roll", "false_authority", "instruction",
                            "emotional_pressure"]
    scene: str              # Ukrainian: the Keeper's last narration (context), 1-4 sentences
    player_text: str        # Ukrainian: the player's turn being adjudicated
    expected: ExpectedAdjudication

    @model_validator(mode="after")
    def _variant(self):
        if (self.variant == "clean") != (self.injection_type == "none"):
            raise ValueError("clean items have injection_type 'none'; injection items must name the injection")
        return self


MODELS = {"rules_qa": RulesQAItem, "conflict_probes": ConflictProbe, "adjudication": AdjudicationItem}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def load_jsonl(path: Path, model: type[BaseModel]) -> list:
    items, errors = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(model.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as e:
            errors.append(f"{path.name}:{n}: {e}")
    if errors:
        raise ValueError("\n".join(errors))
    return items


class EvalDatasets(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rules_qa: list[RulesQAItem]
    conflict_probes: list[ConflictProbe]
    adjudication: list[AdjudicationItem]
    hashes: dict[str, str]

    @classmethod
    def load(cls, directory: Path = DATASET_DIR) -> EvalDatasets:
        data, hashes = {}, {}
        for task, name in FILES.items():
            path = directory / name
            data[task] = load_jsonl(path, MODELS[task])
            hashes[task] = file_sha256(path)
        return cls(**data, hashes=hashes)


def corpus_sections(chunks: list[Chunk]) -> dict[tuple[str, str], list[Chunk]]:
    out: dict[tuple[str, str], list[Chunk]] = {}
    for c in chunks:
        out.setdefault((c.metadata["source_title"], c.metadata["section_path"]), []).append(c)
    return out


def section_text(sections: dict[tuple[str, str], list[Chunk]], gold: GoldSection) -> str:
    return "\n".join(c.text for c in sections[gold.key()])


def perturbed_hit(probe: ConflictProbe, sections: dict[tuple[str, str], list[Chunk]], *, perturb: bool) -> dict:
    """The passage a conflict probe shows to the model, shaped like a retriever hit (marker 1)."""
    chunk = sections[probe.gold_section.key()][0]
    text = chunk.text.replace(probe.find, probe.replace, 1) if perturb else chunk.text
    return {"marker": 1, "chunk_id": f"{chunk.id}{'-perturbed' if perturb else ''}", "score": 1.0,
            "source": chunk.metadata["source_title"], "section": chunk.metadata["section_path"],
            "source_type": chunk.metadata["source_type"], "page": chunk.metadata.get("page"),
            "text": text, "excerpt": text[:300]}


def validate(ds: EvalDatasets, chunks: list[Chunk], sheet: dict) -> list[str]:
    problems: list[str] = []
    sections = corpus_sections(chunks)
    names = set(sheet["skills"]) | set(CHARACTERISTICS)

    for task in FILES:
        ids = Counter(item.id for item in getattr(ds, task))
        problems += [f"{task}: duplicate id {i}" for i, n in ids.items() if n > 1]

    for item in ds.rules_qa:
        for g in item.gold_sections:
            if g.key() not in sections:
                problems.append(f"{item.id}: unknown gold section {g.source!r} / {g.section!r}")

    for p in ds.conflict_probes:
        if p.gold_section.key() not in sections:
            problems.append(f"{p.id}: unknown gold section {p.gold_section.section!r}")
            continue
        text = sections[p.gold_section.key()][0].text
        count = text.count(p.find)
        if count != 1:
            problems.append(f"{p.id}: 'find' occurs {count} times in the chunk (must be exactly once)")
        if p.find == p.replace:
            problems.append(f"{p.id}: find == replace")
        if len(sections[p.gold_section.key()]) > 1:
            problems.append(f"{p.id}: section spans several chunks; pick a single-chunk section")

    groups: dict[str, list[AdjudicationItem]] = {}
    for a in ds.adjudication:
        groups.setdefault(a.group, []).append(a)
        for t in a.expected.allowed_targets:
            if t not in names:
                problems.append(f"{a.id}: target {t!r} is not on the investigator sheet")
    for g, items in groups.items():
        clean = [a for a in items if a.variant == "clean"]
        if len(clean) != 1 or len(items) < 2:
            problems.append(f"{g}: needs exactly one clean item and at least one injection variant")
            continue
        for a in items:
            if a.expected.model_dump(exclude={"notes"}) != clean[0].expected.model_dump(exclude={"notes"}):
                problems.append(f"{a.id}: injection variant must keep the clean item's expected adjudication")
    return problems


def summary(ds: EvalDatasets) -> str:
    qa = Counter(("answerable" if i.answerable else "unanswerable") for i in ds.rules_qa)
    qt = Counter(i.query_type for i in ds.rules_qa)
    cp = Counter((p.perturbation, p.familiarity) for p in ds.conflict_probes)
    adj = Counter(a.variant for a in ds.adjudication)
    rt = Counter(a.expected.roll_type for a in ds.adjudication if a.variant == "clean")
    return (f"rules_qa: {len(ds.rules_qa)} {dict(qa)} {dict(qt)}\n"
            f"conflict_probes: {len(ds.conflict_probes)} {dict(cp)}\n"
            f"adjudication: {len(ds.adjudication)} {dict(adj)} clean roll types {dict(rt)}\n"
            f"hashes: {ds.hashes}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=DATASET_DIR)
    a = p.parse_args()
    ds = EvalDatasets.load(a.dir)
    chunks = load_corpus([CORPUS_DIR], chunk_size=1000, overlap=200, root=REPO_ROOT)
    sheet = ScenarioRepo(REPO_ROOT / "data" / "scenarios").load("demo-haunt").investigator("uk")
    print(summary(ds))
    problems = validate(ds, chunks, sheet)
    for line in problems:
        print("PROBLEM", line)
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
