from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from .experiments import ExperimentRegistry
from .game.coc7e.dice import DEFAULT_RULES, RulesConfig
from .llm import ModelProfile
from .prompts import PromptSet
from .runlog import RunLog
from .scenarios import ScenarioRepo


class LLMs(Protocol):
    def profile(self, name: str) -> ModelProfile: ...
    def chat(self, name: str, *, seed: int | None = None) -> BaseChatModel: ...
    def structured(self, name: str, schema: type[BaseModel], *, seed: int | None = None) -> Runnable: ...


@dataclass
class Deps:
    llms: LLMs
    prompts: PromptSet
    scenarios: ScenarioRepo
    experiments: ExperimentRegistry
    runlog: RunLog
    rules: RulesConfig = field(default_factory=lambda: DEFAULT_RULES)
