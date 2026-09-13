from __future__ import annotations

import asyncio
import logging
import subprocess
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from keeper.deps import Deps
from keeper.experiments import ExperimentRegistry
from keeper.graph import build_graph
from keeper.llm import ModelFactory
from keeper.prompts import PromptSet
from keeper.runlog import RunLog
from keeper.scenarios import ScenarioRepo

from .routes import router
from .settings import REPO_ROOT, Settings, get_settings
from .store import CampaignStore

VERSION = "0.2.0"
log = logging.getLogger("gm")


def detect_git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_deps(s: Settings) -> Deps:
    llms = ModelFactory.from_yaml(s.config_dir / "models.yaml", google_api_key=s.google_api_key,
                                  ollama_url=s.ollama_base_url, vllm_url=s.vllm_base_url,
                                  timeout=s.model_timeout_s, google_rpm=s.google_rpm)
    experiments = ExperimentRegistry.from_dir(s.experiments_dir, s.default_arm)
    spec = experiments.resolve()
    for name in {spec.generator, spec.aux}:  # fail fast instead of on the first turn
        if llms.profile(name).provider == "google_genai" and not s.google_api_key:
            raise RuntimeError(f"GOOGLE_API_KEY is required by model profile {name!r} of arm {spec.arm!r}")
    return Deps(llms=llms, prompts=PromptSet(), scenarios=ScenarioRepo(s.data_dir / "scenarios"),
                experiments=experiments,
                runlog=RunLog(s.artifacts_dir / "runs", git_sha=s.git_sha or detect_git_sha(),
                              enabled=s.runlog_enabled))


@asynccontextmanager
async def open_checkpointer(path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    path = path.resolve()
    if any(part.lower().startswith("onedrive") for part in path.parts):
        raise RuntimeError(f"{path}: SQLite WAL on a OneDrive-synced folder is unsafe; set CHECKPOINT_PATH elsewhere")
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


def create_app(settings: Settings | None = None, deps: Deps | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        s = settings or get_settings()
        logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        d = deps or build_deps(s)
        async with AsyncExitStack() as stack:
            saver = await stack.enter_async_context(open_checkpointer(s.checkpoint_path))
            app.state.store = await stack.enter_async_context(CampaignStore.open(s.app_db_path))
            app.state.checkpointer = saver
            app.state.deps = d
            app.state.graph = build_graph(d).compile(checkpointer=saver)
            app.state.locks = defaultdict(asyncio.Lock)  # one uvicorn worker by design
            log.info("gm-assistant ready: arm=%s", d.experiments.default_arm)
            yield

    app = FastAPI(title="gm-assistant", version=VERSION, lifespan=lifespan)
    app.include_router(router, prefix="/api")  # no CORS middleware: same origin via the Vite proxy
    return app


app = create_app()
