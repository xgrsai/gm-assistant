"""LLM-as-judge for rules answers and knowledge-conflict probes (structured output, versioned prompts)."""

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


async def judge_rules_answer(llms, profile: str, prompt_version: str, item: RulesQAItem, lang: str,
                             answer: str) -> dict:
    question = item.question_uk if lang == "uk" else item.question_en
    prompt = PROMPTS.render(prompt_version, question=question, answerable=item.answerable,
                            reference_answer=item.reference_answer, key_facts=item.key_facts, answer=answer)
    j: QAJudgment = await llms.structured(profile, QAJudgment, seed=0).ainvoke([HumanMessage(prompt)])
    present = sorted({i for i in j.key_facts_present if 0 <= i < len(item.key_facts)})
    return {**j.model_dump(), "key_facts_present": present, "score": VERDICT_SCORE[j.verdict],
            "key_fact_coverage": (len(present) / len(item.key_facts)) if item.key_facts else None}


def swap_order(item_id: str) -> bool:
    """Deterministic 50/50 order of the two facts, so a position bias of the judge cannot favour one condition."""
    return hashlib.sha256(item_id.encode()).digest()[0] % 2 == 1


async def judge_conflict_answer(llms, profile: str, prompt_version: str, probe: ConflictProbe, lang: str,
                                answer: str) -> dict:
    """Classifies the answer as following the ORIGINAL or the PERTURBED fact (independent of the condition)."""
    question = probe.question_uk if lang == "uk" else probe.question_en
    swapped = swap_order(probe.id)
    a, b = (probe.original_fact, probe.perturbed_fact) if not swapped else (probe.perturbed_fact, probe.original_fact)
    prompt = PROMPTS.render(prompt_version, fact_a=a, fact_b=b, question=question, answer=answer)
    j: ConflictJudgment = await llms.structured(profile, ConflictJudgment, seed=0).ainvoke([HumanMessage(prompt)])
    mapping = {"a": "original", "b": "perturbed"} if not swapped else {"a": "perturbed", "b": "original"}
    follows = mapping.get(j.follows, j.follows)  # both / neither pass through
    return {"follows": follows, "reason": j.reason, "order_swapped": swapped}
