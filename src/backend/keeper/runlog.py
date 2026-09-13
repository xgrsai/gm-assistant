"""One JSONL research record per turn (artifacts/runs/<date>.jsonl, git-ignored)."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path


class RunLog:
    def __init__(self, directory: Path, *, git_sha: str | None = None, enabled: bool = True):
        self.dir = directory
        self.git_sha = git_sha
        self.enabled = enabled
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        if not self.enabled:
            return
        now = datetime.now(UTC)
        record = {"ts": now.isoformat(timespec="seconds"), "git_sha": self.git_sha, **record}
        self.dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, open(self.dir / f"{now:%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")
