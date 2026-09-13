"""Named experiment arms (experiments/*.yaml) resolved into frozen ExperimentSpecs."""

from __future__ import annotations

from pathlib import Path

import yaml

from .spec import ExperimentSpec


class ExperimentRegistry:
    def __init__(self, arms: dict[str, ExperimentSpec], default_arm: str):
        if default_arm not in arms:
            raise ValueError(f"default arm {default_arm!r} is not defined (known: {sorted(arms)})")
        self.arms = arms
        self.default_arm = default_arm

    @classmethod
    def from_dir(cls, directory: Path, default_arm: str) -> ExperimentRegistry:
        arms: dict[str, ExperimentSpec] = {}
        for path in sorted(directory.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            fixed = raw.get("fixed", {})
            for arm in raw.get("arms", []):
                spec = ExperimentSpec(**{**fixed, **arm})
                if spec.arm in arms:
                    raise ValueError(f"arm {spec.arm!r} defined twice ({path.name})")
                arms[spec.arm] = spec
        return cls(arms, default_arm)

    def resolve(self, arm: str | None = None) -> ExperimentSpec:
        return self.arms[arm or self.default_arm]
