"""Experiment specification: the variant is data, frozen per campaign and hashed into every run record."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: str
    generator: str                                  # model profile that writes the narration (the FT factor)
    aux: str = "gemini-aux"                         # model profile for adjudication, fixed across arms
    adjudicator: Literal["aux", "generator"] = "aux"
    use_rag: bool = False                           # the RAG factor (Phase 2)
    prompt_version: str = "keeper-v1"
    adjudicate_prompt_version: str = "adjudicate-v1"
    seed: int | None = None                         # None in play (OS entropy); fixed in evals
    history_max_tokens: int = 6000

    def spec_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class KeeperContext:
    """Run-scoped context passed as `context=`; never checkpointed."""

    spec: ExperimentSpec
    campaign_id: str
