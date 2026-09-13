"""Scenario packs: data/scenarios/<id>/scenario.yaml (self-written or openly licensed content only)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from .game.coc7e.rules import build_investigator

Localized = dict[str, str]  # {"uk": ..., "en": ...}


class Pregen(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: Localized
    occupation: Localized
    age: int
    characteristics: dict[str, int]
    skills: dict[str, int]
    luck: int


class ScenarioPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    game_system: str = "coc7e"
    title: Localized
    era: Localized
    blurb: Localized
    opening: Localized
    keeper_notes: str
    sanity_events: dict[str, str] = {}
    investigators: list[Pregen]

    def localized(self, field: Localized, language: str) -> str:
        return field.get(language) or field.get("en") or next(iter(field.values()))

    def scenario_state(self, language: str) -> dict:
        return {
            "id": self.id,
            "game_system": self.game_system,
            "title": self.localized(self.title, language),
            "era": self.localized(self.era, language),
            "language": language,
            "opening": self.localized(self.opening, language).strip(),
            "keeper_notes": self.keeper_notes.strip(),
            "sanity_events": self.sanity_events,
        }

    def investigator(self, language: str, pregen_id: str | None = None) -> dict:
        pregen = next((p for p in self.investigators if p.id == pregen_id), self.investigators[0])
        raw = pregen.model_dump()
        raw["name"] = self.localized(pregen.name, language)
        raw["occupation"] = self.localized(pregen.occupation, language)
        return build_investigator(raw)

    def public(self) -> dict:
        return {"id": self.id, "title": self.title, "era": self.era, "blurb": self.blurb}


class ScenarioRepo:
    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, ScenarioPack] = {}

    def ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.parent.name for p in self.root.glob("*/scenario.yaml"))

    def load(self, scenario_id: str) -> ScenarioPack:
        if scenario_id not in self._cache:
            path = self.root / scenario_id / "scenario.yaml"
            if not path.is_file():
                raise KeyError(scenario_id)
            self._cache[scenario_id] = ScenarioPack(**yaml.safe_load(path.read_text(encoding="utf-8")))
        return self._cache[scenario_id]

    def all(self) -> list[ScenarioPack]:
        return [self.load(i) for i in self.ids()]
