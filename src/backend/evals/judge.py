"""LLM-as-judge for rules answers and knowledge-conflict probes (structured output, versioned prompts).

The prompt of an item and the post-processing of a judgment are pure functions (`*_prompt`, `*_result`): the async
`judge_*_answer` functions call a model between them, and `evals/external_judge.py` exports the same prompts to a judge
that runs outside the harness and imports its judgments through the same `*_result` functions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from keeper.prompts import PromptSet

from .datasets import ConflictProbe, RulesQAItem

PROMPTS = PromptSet(Path(__file__).parent / "prompts")


class QAJudgment(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect"]
    key_facts_present: list[int] = Field(default_factory=list, description="Indices of key facts stated correctly")
    contradicts_reference: bool
    abstained: bool
    reason: str = ""


class ConflictJudgment(BaseModel):
    follows: Literal["a", "b", "both", "neither"]
    reason: str = ""


VERDICT_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def judge_signature(judge: str, version: str) -> str:
    """What a judgment is keyed by next to its record: the judge profile and the prompt version with its hash."""
    return f"{judge}|{version}|{PROMPTS.hash(version)}"


def rules_prompt(item: RulesQAItem, lang: str, answer: str, version: str) -> str:
    question = item.question_uk if lang == "uk" else item.question_en
    return PROMPTS.render(version, question=question, answerable=item.answerable,
                          reference_answer=item.reference_answer, key_facts=item.key_facts, answer=answer)


def rules_result(item: RulesQAItem, j: QAJudgment) -> dict:
    present = sorted({i for i in j.key_facts_present if 0 <= i < len(item.key_facts)})
    return {**j.model_dump(), "key_facts_present": present, "score": VERDICT_SCORE[j.verdict],
            "key_fact_coverage": (len(present) / len(item.key_facts)) if item.key_facts else None}


async def judge_rules_answer(llms, profile: str, prompt_version: str, item: RulesQAItem, lang: str,
                             answer: str) -> dict:
    prompt = rules_prompt(item, lang, answer, prompt_version)
    j: QAJudgment = await llms.structured(profile, QAJudgment, seed=0).ainvoke([HumanMessage(prompt)])
    return rules_result(item, j)


def swap_order(item_id: str) -> bool:
    """Deterministic 50/50 order of the two facts, so a position bias of the judge cannot favour one condition."""
    return hashlib.sha256(item_id.encode()).digest()[0] % 2 == 1


def conflict_prompt(probe: ConflictProbe, lang: str, answer: str, version: str) -> tuple[str, bool]:
    """(prompt, whether the two facts were shown swapped: A = the perturbed fact)."""
    question = probe.question_uk if lang == "uk" else probe.question_en
    swapped = swap_order(probe.id)
    a, b = (probe.original_fact, probe.perturbed_fact) if not swapped else (probe.perturbed_fact, probe.original_fact)
    return PROMPTS.render(version, fact_a=a, fact_b=b, question=question, answer=answer), swapped


def conflict_result(probe: ConflictProbe, j: ConflictJudgment) -> dict:
    """Maps the judge's A/B back to the original/perturbed fact (independent of the condition)."""
    swapped = swap_order(probe.id)
    mapping = {"a": "original", "b": "perturbed"} if not swapped else {"a": "perturbed", "b": "original"}
    follows = mapping.get(j.follows, j.follows)  # both / neither pass through
    return {"follows": follows, "reason": j.reason, "order_swapped": swapped}


async def judge_conflict_answer(llms, profile: str, prompt_version: str, probe: ConflictProbe, lang: str,
                                answer: str) -> dict:
    """Classifies the answer as following the ORIGINAL or the PERTURBED fact (independent of the condition)."""
    prompt, _ = conflict_prompt(probe, lang, answer, prompt_version)
    j: ConflictJudgment = await llms.structured(profile, ConflictJudgment, seed=0).ainvoke([HumanMessage(prompt)])
    return conflict_result(probe, j)
