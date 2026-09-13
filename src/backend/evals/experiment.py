"""Evaluation experiment files (evals/experiments/*.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from keeper.spec import ExperimentSpec


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalTaskCfg(_Cfg):
    enabled: bool = True
    limit: int | None = None
    k: int = 10
    # en: question_en as the query; uk: raw Ukrainian question; uk_analyze: the analyze node's query for the
    # Ukrainian question (one aux LLM call per item, cached in the frozen contexts)
    query_modes: list[Literal["en", "uk", "uk_analyze"]] = ["en", "uk"]
    threshold_grid: list[float] = Field(default_factory=lambda: [round(0.40 + i * 0.01, 2) for i in range(41)])


class RulesQATaskCfg(_Cfg):
    enabled: bool = True
    limit: int | None = None
    languages: list[Literal["uk", "en"]] = ["uk"]
    # query used to freeze RAG contexts: "analyze" = the production analyze node (aux LLM call per item);
    # "question_en" = the English gold wording (no LLM call; an oracle translation, report it as such)
    context_query: Literal["analyze", "question_en"] = "analyze"


class ConflictTaskCfg(_Cfg):
    enabled: bool = True
    limit: int | None = None
    languages: list[Literal["uk", "en"]] = ["uk"]
    # perturbed: the edited passage; original: control with the unedited passage (RAG arms);
    # no_context: the question alone (prompt-only arms) -> what the model believes without context
    conditions: list[Literal["perturbed", "original", "no_context"]] = ["perturbed", "original", "no_context"]


class AdjudicationTaskCfg(_Cfg):
    enabled: bool = True
    limit: int | None = None   # number of groups (a group = clean item + its injection variants)


class Tasks(_Cfg):
    retrieval: RetrievalTaskCfg = RetrievalTaskCfg()
    rules_qa: RulesQATaskCfg = RulesQATaskCfg()
    conflict_probes: ConflictTaskCfg = ConflictTaskCfg()
    adjudication: AdjudicationTaskCfg = AdjudicationTaskCfg()


class EvalExperiment(_Cfg):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    description: str = ""
    scenario_id: str = "demo-haunt"
    dataset_dir: str = "eval/coc7e"         # relative to DATA_DIR
    corpus_dirs: list[str] = ["open/coc7e"]  # relative to DATA_DIR; must be what the RAG collection was built from
    seeds: list[int] = [11]
    judge: str | None = "gemini-judge"      # model profile; null disables LLM judging
    judge_prompt_versions: dict[str, str] = {"rules_qa": "judge-qa-v1", "conflict_probes": "judge-conflict-v1"}
    fixed: dict = {}
    arms: list[dict]
    tasks: Tasks = Tasks()

    @model_validator(mode="after")
    def _arms(self):
        names = [a.get("arm") for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("arm names must be unique")
        self.specs(0)  # validates every arm against ExperimentSpec
        return self

    @classmethod
    def load(cls, path: Path) -> EvalExperiment:
        return cls(**yaml.safe_load(path.read_text(encoding="utf-8")))

    def specs(self, seed: int) -> list[ExperimentSpec]:
        return [ExperimentSpec(**{**self.fixed, **arm, "seed": seed}) for arm in self.arms]

    def rag_specs(self) -> list[ExperimentSpec]:
        """Distinct retrieval configurations used by the RAG arms (one frozen-context set each)."""
        out: dict[str, ExperimentSpec] = {}
        for spec in self.specs(0):
            if spec.use_rag:
                out.setdefault(context_signature(spec), spec)
        return list(out.values())


def context_signature(spec: ExperimentSpec) -> str:
    """What determines a frozen turn context: aux model, analyze prompt, retrieval settings (not the generator)."""
    return "|".join([spec.aux, spec.analyze_prompt_version, spec.retrieval.model_dump_json()])
