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
from qdrant_client import AsyncQdrantClient

from keeper.deps import Deps
from keeper.experiments import ExperimentRegistry
from keeper.graph import build_graph
from keeper.llm import ModelFactory
from keeper.prompts import PromptSet
from keeper.rag.embeddings import build_embeddings, load_profiles
from keeper.rag.retriever import Retriever
from keeper.runlog import RunLog
from keeper.scenarios import ScenarioRepo

from .routes import router
from .settings import REPO_ROOT, Settings, get_settings
from .store import CampaignStore

VERSION = "0.3.0"
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
    retriever = None
    if any(a.use_rag for a in experiments.arms.values()):
        profiles = load_profiles(s.config_dir / "embedders.yaml")
        retriever = Retriever(AsyncQdrantClient(url=s.qdrant_url),
                              {n: build_embeddings(p, ollama_url=s.ollama_base_url) for n, p in profiles.items()},
                              {n: p.dim for n, p in profiles.items()})
    return Deps(llms=llms, prompts=PromptSet(), scenarios=ScenarioRepo(s.data_dir / "scenarios"),
                experiments=experiments, retriever=retriever,
                runlog=RunLog(s.artifacts_dir / "runs", git_sha=s.git_sha or detect_git_sha(),
                              enabled=s.runlog_enabled))


async def verify_retrieval(d: Deps) -> None:
    """Every RAG arm needs an existing collection and a reachable embedder: fail at startup, not mid-game."""
    # dedupe by value: RetrievalSpec holds a dict (quotas), so it is not hashable
    rag_arms = [a for a in d.experiments.arms.values() if a.use_rag]
    specs = list({a.retrieval.model_dump_json(): a.retrieval for a in rag_arms}.values())
    if specs and d.retriever is None:
        raise RuntimeError("RAG arms are defined but no retriever is configured")
    for spec in specs:
        try:
            await d.retriever.verify(spec)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"knowledge base unavailable for collection {spec.collection!r} "
                               f"(is Qdrant running and is Ollama serving the embedder?): {e!r}") from e


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
            if d.retriever is not None:
                stack.push_async_callback(d.retriever.client.close)
                await verify_retrieval(d)
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
