"""Schemas of the Phase 4 SFT data: seeds (what the author writes), frozen inputs (what the builder renders),
targets and reviews (what S5 adds).

Flow (every file is JSONL, one object per line, Ukrainian text kept unescaped):

    seeds.v1.jsonl   SftSeed     one narrator turn to teach: scene so far, player turn, adjudication (the output of
                                 the adjudicator is an INPUT here), knowledge-base mode. Written by hand / by Claude.
    inputs.v1.jsonl  SftInput    `python -m sft.build inputs`: the seed plus the dice rolled by code, the exact KB
                                 chunks and the rendered chat messages. Committed and frozen, so a target can never
                                 disagree with the rolls it narrates, even if the rules code changes later.
    targets.v1.jsonl SftTarget   the Keeper reply for one input (authored later, reviewed by the author).
    reviews.v1.jsonl SftReview   the author's verdict on one target.

A seed describes the scene only up to the player's turn; `history` excludes the scenario opening (the builder
prepends it, exactly as `seed_campaign` does). Held-out and unknown corpus sections, unknown skill names and
unsupported adjudications are rejected by the builder with the seed id in the message.
"""

from __future__ import annotations

import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from keeper.nodes import AdjudicationV3

ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{2,63}$"
Kind = Literal["in_fiction", "rules_question", "meta"]
KbMode = Literal["oracle", "distractor_only", "none"]
MAX_PLAYER_TEXT = 4000   # the longest player message the app accepts


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SectionRef(_Strict):
    """A corpus section as in the gold datasets: front-matter source title + heading path."""

    source: str
    section: str

    def key(self) -> tuple[str, str]:
        return (self.source, self.section)


def text_problem(text: str) -> str | None:
    """Why a turn text cannot go into training data as it is (it is rejected, never silently fixed).

    The training gates (`sft.gates.check_messages`) refuse non-NFC content and assistant replies with outer
    whitespace, and the app strips player text, so an unstripped player turn would differ from production.
    """
    if not text.strip():
        return "text is blank"
    if unicodedata.normalize("NFC", text) != text:
        return "text is not NFC-normalised (decomposed characters, e.g. a letter plus a combining mark)"
    if text != text.strip():
        return "text has leading or trailing whitespace"
    return None


def _clean_text(text: str) -> str:
    if problem := text_problem(text):
        raise ValueError(problem)
    return text


class HistoryTurn(_Strict):
    role: Literal["player", "keeper"]
    text: str

    @field_validator("text")
    @classmethod
    def _check_text(cls, v: str) -> str:
        return _clean_text(v)


class KbSpec(_Strict):
    """What the KNOWLEDGE BASE block of the prompt contains.

    oracle           the chunks of `oracle_sections` (the answer is there) plus `n_distractors` unrelated train chunks
    distractor_only  only `n_distractors` chunks that do NOT answer the question: the target must be a "Keeper ruling"
    none             nothing was retrieved ("no relevant passages were found"); also the only mode when use_rag is off
    """

    mode: KbMode = "none"
    oracle_sections: list[SectionRef] = []
    n_distractors: int = Field(0, ge=0, le=4)
    # Sections that may answer the question but must NOT be shown as distractors (REQUIRED for distractor_only;
    # held-out or train sections, they only have to exist in the corpus). Their sibling sections, i.e. the ones under
    # the same top-level heading, are excluded too. The oracle sections and their siblings are always excluded.
    exclude_sections: list[SectionRef] = []

    @model_validator(mode="after")
    def _consistent(self):
        if self.mode == "oracle" and not self.oracle_sections:
            raise ValueError("kb.mode 'oracle' needs oracle_sections")
        if self.mode != "oracle" and self.oracle_sections:
            raise ValueError(f"kb.mode {self.mode!r} takes no oracle_sections")
        if self.mode == "distractor_only" and self.n_distractors < 1:
            raise ValueError("kb.mode 'distractor_only' needs n_distractors >= 1")
        if self.mode == "none" and (self.n_distractors or self.exclude_sections):
            raise ValueError("kb.mode 'none' takes no distractors and no exclude_sections")
        if self.mode == "distractor_only" and not self.exclude_sections:
            raise ValueError("kb.mode 'distractor_only' needs exclude_sections: the sections that would answer the "
                             "question, so they are never drawn as distractors")
        for name in ("oracle_sections", "exclude_sections"):
            keys = [s.key() for s in getattr(self, name)]
            if len(set(keys)) != len(keys):
                raise ValueError(f"duplicate {name}")
        if {s.key() for s in self.oracle_sections} & {s.key() for s in self.exclude_sections}:
            raise ValueError("a section cannot be both an oracle and excluded")
        return self


class SftSeed(_Strict):
    """One narrator turn to teach.

    The investigator is always the pack's fresh pregen sheet: `history` is context only and must not describe
    mechanical consequences (Sanity or HP lost, wounds, spent Luck) that the sheet in the prompt does not show.
    Texts must be NFC and stripped (see `text_problem`).
    """

    id: str = Field(pattern=ID_PATTERN)
    scenario_id: str
    language: Literal["uk", "en"]
    kind: Kind
    use_rag: bool
    investigator_id: str | None = None      # None = the pack's first pregen, as in a new campaign
    history: list[HistoryTurn] = []         # player, keeper, player, keeper ... (the opening is prepended)
    player_text: str = Field(max_length=MAX_PLAYER_TEXT)
    adjudication: AdjudicationV3
    kb: KbSpec = Field(default_factory=KbSpec)

    @field_validator("player_text")
    @classmethod
    def _check_player_text(cls, v: str) -> str:
        return _clean_text(v)

    @model_validator(mode="after")
    def _consistent(self):
        problems = []
        if len(self.history) % 2 or any(t.role != ("player", "keeper")[i % 2] for i, t in enumerate(self.history)):
            problems.append("history must alternate player, keeper, ... and end with a keeper turn")
        a = self.adjudication
        if a.kind != self.kind:
            problems.append(f"adjudication.kind {a.kind!r} differs from kind {self.kind!r}")
        if a.needs_roll != (a.roll_type != "none"):
            problems.append("adjudication.needs_roll must be true exactly when roll_type is not 'none'")
        if self.kind != "in_fiction" and (a.needs_roll or a.new_day):
            problems.append(f"a {self.kind} turn takes no roll and does not move the clock")
        if a.roll_type in ("skill", "characteristic") and not a.target:
            problems.append(f"roll_type {a.roll_type!r} needs a target")
        if a.roll_type in ("none", "luck", "sanity") and a.target:
            problems.append(f"roll_type {a.roll_type!r} takes no target")
        if a.roll_type == "sanity" and not (a.sanity_event or a.sanity_loss):
            problems.append("a sanity roll needs sanity_event or sanity_loss")
        if a.roll_type != "sanity" and (a.sanity_event or a.sanity_loss):
            problems.append("sanity_event / sanity_loss only go with roll_type 'sanity'")
        if a.is_push:
            problems.append("is_push is not supported in SFT seeds (the previous roll is not part of a seed)")
        if not (0 <= a.bonus_dice <= 2 and 0 <= a.penalty_dice <= 2):
            problems.append("bonus_dice and penalty_dice must be 0..2")
        if not self.use_rag and self.kb.mode != "none":
            problems.append("use_rag is false, so kb.mode must be 'none'")
        if problems:
            raise ValueError("; ".join(problems))
        return self


class KbHit(_Strict):
    """One retrieved chunk as rendered into the prompt (the text itself is in the messages / the corpus)."""

    marker: int
    chunk_id: str
    source: str
    section: str
    source_type: str
    role: Literal["oracle", "distractor"]


class RenderInfo(_Strict):
    """What the prompt was rendered with, so a build is auditable."""

    spec: dict
    spec_hash: str
    generator_profile: dict
    prompt_version: str
    prompt_hash: str
    corpus_chunk_ids_hash: str
    split_seed: int
    dice_base_seed: int                  # per-example dice seed = f(dice_base_seed, id), see sft.build.dice_seed


class Message(_Strict):
    role: Literal["system", "user", "assistant"]
    content: str


class SftInput(_Strict):
    """One frozen input line: everything a target must agree with, and the exact chat the model will see."""

    id: str
    seed_hash: str                       # sha256/16 of the seed as read; a changed seed changes this
    scenario_id: str
    language: Literal["uk", "en"]
    kind: Kind
    use_rag: bool
    investigator_id: str | None
    turn_no: int                         # number of player turns including this one (the dice salt)
    dice_seed: int
    history: list[HistoryTurn]
    player_text: str
    adjudication: dict                   # as the adjudicator would hand it to the narrator
    rolls: list[dict]                    # records from rules.resolve, rolled once and frozen here
    errors: list[str]                    # engine notes from rules.resolve (the build refuses non-empty ones)
    kb_mode: KbMode
    oracle_sections: list[SectionRef]
    context_sections: list[SectionRef]   # every section shown in the prompt, oracle and distractors
    hits: list[KbHit]
    render: RenderInfo
    messages: list[Message]              # system + trimmed history, ending with the player turn
    input_hash: str                      # sha256/16 of `messages`; targets reference it


class Provenance(_Strict):
    author: Literal["claude", "human"]
    model: str = ""
    created: str = ""                    # ISO date
    note: str = ""


class SftTarget(_Strict):
    id: str                              # the input id
    text: str = Field(min_length=1)
    provenance: Provenance
    input_hash: str | None = None        # SftInput.input_hash the text was written against

    @model_validator(mode="after")
    def _not_blank(self):
        if not self.text.strip():
            raise ValueError("target text is blank")
        return self


class SftReview(_Strict):
    id: str                              # the target / input id
    verdict: Literal["accept", "edit", "reject"]
    edited_text: str | None = None       # required for 'edit'
    note: str = ""

    @model_validator(mode="after")
    def _edit_has_text(self):
        if self.verdict == "edit" and not (self.edited_text or "").strip():
            raise ValueError("verdict 'edit' needs edited_text")
        if self.verdict != "edit" and self.edited_text is not None:
            raise ValueError(f"verdict {self.verdict!r} takes no edited_text")
        return self

