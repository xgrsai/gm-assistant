"""App factory for the crash-resume test: fake models, narration optionally slowed down so the process can be killed."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from keeper.deps import Deps
from keeper.experiments import ExperimentRegistry
from keeper.prompts import PromptSet
from keeper.runlog import RunLog
from keeper.scenarios import ScenarioRepo
from keeper.spec import ExperimentSpec
from server.main import create_app
from server.settings import REPO_ROOT, Settings

from .fakes import FakeFactory

SPOT = {"kind": "in_fiction", "needs_roll": True, "roll_type": "skill", "target": "Spot Hidden"}


class SlowFake(GenericFakeChatModel):
    delay: float = 0.0

    async def _astream(self, *args, **kwargs):
        await asyncio.sleep(self.delay)
        async for chunk in super()._astream(*args, **kwargs):
            yield chunk

    async def _agenerate(self, *args, **kwargs):
        await asyncio.sleep(self.delay)
        return await super()._agenerate(*args, **kwargs)


class SlowFactory(FakeFactory):
    def __init__(self, delay: float, **kw):
        super().__init__(**kw)
        self.delay = delay

    def chat(self, name, *, seed=None):
        return SlowFake(messages=self.narrations, delay=self.delay)


def app():
    tmp = Path(os.environ["CRASH_TMP"])
    settings = Settings(_env_file=None, checkpoint_path=tmp / "cp.db", app_db_path=tmp / "app.db",
                        artifacts_dir=tmp / "artifacts", data_dir=REPO_ROOT / "data")
    fake = SlowFactory(float(os.environ.get("CRASH_NARRATE_DELAY", "0")),
                       narrations=["Після падіння хід завершено."], adjudications=[SPOT])
    arms = {"test": ExperimentSpec(arm="test", generator="gen", aux="aux", seed=None)}  # OS entropy, like play
    deps = Deps(llms=fake, prompts=PromptSet(), scenarios=ScenarioRepo(settings.data_dir / "scenarios"),
                experiments=ExperimentRegistry(arms, "test"), runlog=RunLog(settings.artifacts_dir / "runs"))
    return create_app(settings, deps)
