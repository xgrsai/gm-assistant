"""A real process kill during narration must not lose the dice: Retry resumes after `mechanics` (durability="sync")."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

BACKEND = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(tmp: Path, port: int, delay: float) -> subprocess.Popen:
    env = {**os.environ, "CRASH_TMP": str(tmp), "CRASH_NARRATE_DELAY": str(delay), "PYTHONUTF8": "1"}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "tests.crash_app:app", "--factory",
                             "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
                            cwd=BACKEND, env=env)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("server did not start")


def sse_events(response: httpx.Response):
    event = "message"
    for line in response.iter_lines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            yield event, json.loads(line[5:])


def test_kill_during_narrate_then_retry(tmp_path):
    port = free_port()
    base = f"http://127.0.0.1:{port}/api"
    proc = start_server(tmp_path, port, delay=60)
    try:
        cid = httpx.post(f"{base}/campaigns", json={"title": "crash", "language": "uk"}).json()["id"]
        roll = None
        with httpx.stream("POST", f"{base}/campaigns/{cid}/turns", timeout=30,
                          json={"client_turn_id": str(uuid.uuid4()), "text": "Оглядаю підлогу"}) as r:
            for event, data in sse_events(r):
                if event == "progress" and data["node"] == "narrate" and roll is not None:
                    proc.kill()  # hard kill while the narrator is "thinking"
                    break
                if event == "roll":
                    roll = data
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert roll is not None

    proc = start_server(tmp_path, port, delay=0)
    try:
        msgs = httpx.get(f"{base}/campaigns/{cid}/messages").json()
        assert msgs["incomplete"] is True
        with httpx.stream("POST", f"{base}/campaigns/{cid}/turns/retry", timeout=30) as r:
            events = list(sse_events(r))
        names = [e for e, _ in events]
        assert "roll" not in names and ("progress", {"node": "adjudicate"}) not in events
        assert events[-1][0] == "done"
        assert events[-1][1]["message"]["rolls"] == [roll]  # the same dice, not a re-roll
    finally:
        proc.kill()
        proc.wait(timeout=10)
