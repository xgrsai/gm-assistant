"""Campaign registry (var/app.db). Conversation and game state live in LangGraph checkpoints, not here."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from keeper.spec import ExperimentSpec

SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    scenario_title TEXT NOT NULL,
    language TEXT NOT NULL,
    arm TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0
)
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class Campaign:
    id: str
    title: str
    scenario_id: str
    scenario_title: str
    language: str
    spec: ExperimentSpec
    created_at: str
    updated_at: str

    def public(self) -> dict:
        return {"id": self.id, "title": self.title, "scenario_id": self.scenario_id,
                "scenario_title": self.scenario_title, "language": self.language, "arm": self.spec.arm,
                "created_at": self.created_at, "updated_at": self.updated_at}


class CampaignStore:
    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    @classmethod
    @asynccontextmanager
    async def open(cls, path: Path) -> AsyncIterator[CampaignStore]:
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(SCHEMA)
            await db.commit()
            yield cls(db)

    @staticmethod
    def _row(row: aiosqlite.Row) -> Campaign:
        return Campaign(id=row["id"], title=row["title"], scenario_id=row["scenario_id"],
                        scenario_title=row["scenario_title"], language=row["language"],
                        spec=ExperimentSpec.model_validate_json(row["spec_json"]),
                        created_at=row["created_at"], updated_at=row["updated_at"])

    async def create(self, *, title: str, scenario_id: str, scenario_title: str, language: str,
                     spec: ExperimentSpec) -> Campaign:
        now = _now()
        c = Campaign(id=uuid.uuid4().hex, title=title, scenario_id=scenario_id, scenario_title=scenario_title,
                     language=language, spec=spec, created_at=now, updated_at=now)
        await self.db.execute(
            "INSERT INTO campaigns (id, title, scenario_id, scenario_title, language, arm, spec_json, spec_hash,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (c.id, c.title, c.scenario_id, c.scenario_title, c.language, spec.arm, spec.model_dump_json(),
             spec.spec_hash(), now, now))
        await self.db.commit()
        return c

    async def get(self, campaign_id: str) -> Campaign | None:
        async with self.db.execute("SELECT * FROM campaigns WHERE id = ? AND archived = 0", (campaign_id,)) as cur:
            row = await cur.fetchone()
        return self._row(row) if row else None

    async def list(self) -> list[Campaign]:
        async with self.db.execute("SELECT * FROM campaigns WHERE archived = 0 ORDER BY updated_at DESC") as cur:
            return [self._row(r) for r in await cur.fetchall()]

    async def touch(self, campaign_id: str) -> None:
        await self.db.execute("UPDATE campaigns SET updated_at = ? WHERE id = ?", (_now(), campaign_id))
        await self.db.commit()

    async def delete(self, campaign_id: str) -> None:
        await self.db.execute("DELETE FROM campaigns WHERE id = ?", (campaign_id,))
        await self.db.commit()
