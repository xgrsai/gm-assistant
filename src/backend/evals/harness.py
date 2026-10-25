"""Evaluation harness: runs the production graph/nodes on gold datasets and stores one record per item.

Design (see docs/critique.md C7):
- Stage A, frozen turn contexts: for RAG arms the analyze query and retrieved hits are computed ONCE per
  (item, language, retrieval configuration) and cached in artifacts/eval/contexts/. Every RAG arm and seed then
  receives identical context through `forced_hits`, so arms differ only in what the experiment varies.
- Resumable: records are appended to artifacts/eval/<run>/records.jsonl; finished items are skipped on rerun,
  failed ones are retried. Judgments live in judgments.jsonl, keyed by record and judge, so re-judging with
  another judge never requires re-generating answers.
- The run directory is bound to a fingerprint of the experiment, datasets, prompts and model profiles; a changed
  configuration must use a new run name.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.exceptions import ModelAPIError, ModelConnectionError, ModelRateLimitError, ModelTimeoutError
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeTimeoutError

from keeper.deps import Deps
from keeper.graph import build_graph
from keeper.nodes import adjudicate_turn, search_query
from keeper.rag.corpus import Chunk
from keeper.rag.glossary import load_glossary
from keeper.spec import ExperimentSpec, KeeperContext
from server.settings import REPO_ROOT

from . import model_cards
from .datasets import EvalDatasets, corpus_sections, perturbed_hit
from .experiment import EvalExperiment, context_signature
from .judge import PROMPTS as JUDGE_PROMPTS
from .judge import judge_conflict_answer, judge_rules_answer

GLOSSARY = Path(__file__).resolve().parents[3] / "data" / "glossary" / "coc7e-uk-en.yaml"

log = logging.getLogger("gm.eval")

HIT_FIELDS = ("marker", "chunk_id", "score", "source", "section", "source_type")
CYRILLIC = re.compile("[" + chr(0x0400) + "-" + chr(0x04FF) + "]")
LATIN = re.compile(r"[A-Za-z]")
KEEPER_RULING = re.compile(r"keeper ruling|рішення хранителя|не охоплю|не описан|немає (?:в|у) правилах|"
                           r"правила не (?:кажуть|містять|визначають)", re.I)


RETRYABLE = (ModelRateLimitError, ModelAPIError, ModelConnectionError, ModelTimeoutError, NodeTimeoutError,
             httpx.TransportError)


def fingerprint_view(snapshot: dict) -> dict:
    """The parts of a run configuration that change results.

    Judge settings and the SELECTION of cells (enabled tasks, limits, query modes, languages, conditions) are
    excluded: judgments are keyed by judge signature and records by cell, so a smoke run can grow into the full run.
    Of a model card only what identifies the models is kept: `card_sha256` also covers the timestamps and git state the
    card holds, so re-running the adapter check on the same models must not orphan a run.
    """
    exp = copy.deepcopy(snapshot["experiment"])
    for key in ("description", "judge", "judge_prompt_versions"):
        exp.pop(key, None)
    for task in exp.get("tasks", {}).values():
        for key in ("enabled", "limit", "query_modes", "languages", "conditions", "threshold_grid"):
            task.pop(key, None)
    skip = {"git_sha", "created_at", "judge", "fingerprint", "sessions", "allow_failed_card"}
    view = {**{k: v for k, v in snapshot.items() if k not in skip}, "experiment": exp}
    if "model_cards" in view:
        view["model_cards"] = {name: {k: v for k, v in card.items() if k != "card_sha256"}
                               for name, card in view["model_cards"].items()}
    return view


def sha16(obj) -> str:
    blob = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def record_key(r: dict) -> str:
    return "|".join(str(r.get(k, "")) for k in ("task", "arm", "seed", "item_id", "lang", "condition"))


def slim_hits(hits: list[dict]) -> list[dict]:
    return [{k: h.get(k) for k in HIT_FIELDS} for h in hits]


def text_metrics(text: str) -> dict:
    cyr, lat = len(CYRILLIC.findall(text)), len(LATIN.findall(text))
    return {"chars": len(text), "cyrillic_ratio": round(cyr / (cyr + lat), 3) if cyr + lat else 0.0,
            "keeper_ruling_phrase": bool(KEEPER_RULING.search(text))}


class JsonlStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


@dataclass
class RunPaths:
    root: Path

    @property
    def records(self) -> Path:
        return self.root / "records.jsonl"

    @property
    def judgments(self) -> Path:
        return self.root / "judgments.jsonl"

    @property
    def config(self) -> Path:
        return self.root / "config.json"


@dataclass
class Harness:
    exp: EvalExperiment
    deps: Deps
    datasets: EvalDatasets
    chunks: list[Chunk]
    run_dir: Path
    contexts_dir: Path
    concurrency: int = 2
    judge_enabled: bool = True
    max_attempts: int = 3
    backoff_s: float = 20.0
    repo_root: Path = REPO_ROOT   # model cards are resolved against it
    allow_failed_card: bool = False   # --allow-failed-card: run with a card whose adapter check failed (recorded)
    progress: Callable[[str], None] = print
    _graph: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.paths = RunPaths(self.run_dir)
        self.records = JsonlStore(self.paths.records)
        self.judgments = JsonlStore(self.paths.judgments)
        self.sections = corpus_sections(self.chunks)
        self.pack = self.deps.scenarios.load(self.exp.scenario_id)

    # ------------------------------------------------------------------ configuration fingerprint

    def config_snapshot(self, extra: dict | None = None) -> dict:
        specs = {s.arm: s for s in self.exp.specs(self.exp.seeds[0])}
        profiles = {s.generator for s in specs.values()} | {s.aux for s in specs.values()}
        prompts = {v for s in specs.values() for v in (s.prompt_version, s.adjudicate_prompt_version,
                                                       s.analyze_prompt_version)}
        snapshot = {
            "experiment": self.exp.model_dump(mode="json"),
            "dataset_hashes": self.datasets.hashes,
            "spec_hashes": {f"{s.arm}@{seed}": s.spec_hash()
                            for seed in self.exp.seeds for s in self.exp.specs(seed)},
            "model_profiles": {p: self.deps.llms.profile(p).snapshot() for p in sorted(profiles)},
            "prompt_hashes": {p: self.deps.prompts.hash(p) for p in sorted(prompts)},
            "corpus_chunk_ids_hash": sha16(sorted(c.id for c in self.chunks)),
            "judge": {"profile": self.deps.llms.profile(self.exp.judge).snapshot() if self.exp.judge else None,
                      "prompt_hashes": {p: JUDGE_PROMPTS.hash(p)
                                        for p in sorted(self.exp.judge_prompt_versions.values())}},
            **(extra or {}),
        }
        cards = model_cards.snapshot(self.deps.llms, profiles, self.repo_root)
        if cards:  # only runs with a model-card profile; other runs keep the keys (and fingerprints) they always had
            snapshot["model_cards"] = cards
            if self.allow_failed_card and any(c["status"] != model_cards.PASSED for c in cards.values()):
                snapshot["allow_failed_card"] = True   # a deliberate override, kept out of the fingerprint
        snapshot["fingerprint"] = sha16(fingerprint_view(snapshot))
        return snapshot

    def bind_run_dir(self, extra: dict | None = None) -> dict:
        snap = self.config_snapshot(extra)
        if self.paths.config.exists():
            old = json.loads(self.paths.config.read_text(encoding="utf-8"))
            if old.get("fingerprint") != snap["fingerprint"]:
                raise SystemExit(f"{self.run_dir} belongs to a different configuration (fingerprint "
                                 f"{old.get('fingerprint')} != {snap['fingerprint']}); use a new --run-name")
            # same experiment: keep the original record, remember later sessions (judge, git sha, task selection)
            session = {"judge": snap["judge"], "tasks": snap["experiment"]["tasks"], **(extra or {})}
            if "model_cards" in snap:   # the card file may have been rewritten by a repeated check of the same models
                session["card_sha256"] = {name: c["card_sha256"] for name, c in snap["model_cards"].items()}
            if snap.get("allow_failed_card"):
                session["allow_failed_card"] = True
            old.setdefault("sessions", []).append(session)
            self.paths.config.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
            return old
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths.config.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        return snap

    # ------------------------------------------------------------------ helpers

    @property
    def graph(self):
        if self._graph is None:
            self._graph = build_graph(self.deps).compile(checkpointer=InMemorySaver())
        return self._graph

    def done(self) -> set[str]:
        return {record_key(r) for r in self.records.read() if r.get("status") == "ok"}

    async def gather(self, jobs: Iterable[Callable[[], Awaitable[None]]]) -> None:
        sem = asyncio.Semaphore(self.concurrency)

        async def run(job):
            async with sem:
                await job()

        await asyncio.gather(*(run(j) for j in jobs))

    async def with_retries(self, fn: Callable[[], Awaitable[dict]]) -> dict:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await fn()
            except RETRYABLE as e:  # quota, overload, network: back off and retry the whole item
                if attempt == self.max_attempts:
                    raise
                wait = self.backoff_s * attempt
                self.progress(f"  {type(e).__name__}, retrying in {wait:.0f}s")
                await asyncio.sleep(wait)
        raise AssertionError("unreachable")

    def base_record(self, task: str, spec: ExperimentSpec | None, seed: int, item_id: str, lang: str = "",
                    condition: str = "") -> dict:
        return {"task": task, "arm": spec.arm if spec else "-", "seed": seed, "item_id": item_id, "lang": lang,
                "condition": condition, "spec_hash": spec.spec_hash() if spec else None,
                "use_rag": spec.use_rag if spec else None}

    async def store_result(self, rec: dict, fn: Callable[[], Awaitable[dict]]) -> None:
        t0 = time.monotonic()
        try:
            rec.update(await self.with_retries(fn))
            rec["status"] = "ok"
        except Exception as e:
            log.exception("eval item failed: %s", record_key(rec))
            rec.update(status="error", error=f"{type(e).__name__}: {e}"[:500])
        rec["latency_ms"] = round((time.monotonic() - t0) * 1000)
        rec["ts"] = datetime.now(UTC).isoformat(timespec="seconds")
        self.records.append(rec)
        self.progress(f"  {rec['status']:<5} {record_key(rec)}")

    def items(self, task: str) -> list:
        if task == "retrieval_items":  # the retrieval task scores the rules QA questions
            cfg, items = self.exp.tasks.retrieval, self.datasets.rules_qa
        else:
            cfg, items = getattr(self.exp.tasks, task), getattr(self.datasets, task)
        if task == "adjudication" and cfg.limit is not None:
            groups = sorted({a.group for a in items})[: cfg.limit]
            return [a for a in items if a.group in groups]
        return items[: cfg.limit] if cfg.limit is not None else items

    def scenario_state(self, lang: str) -> dict:
        return self.pack.scenario_state(lang)

    # ------------------------------------------------------------------ Stage A: frozen turn contexts

    def context_store(self, spec: ExperimentSpec, mode: str) -> JsonlStore:
        return JsonlStore(self.contexts_dir / f"{sha16(context_signature(spec) + '|' + mode)}.jsonl")

    def contexts(self, spec: ExperimentSpec, mode: str = "analyze") -> dict[tuple[str, str, str], dict]:
        return {(r["task"], r["item_id"], r["lang"]): r for r in self.context_store(spec, mode).read() if r.get("ok")}

    def context_requests(self) -> list[dict]:
        """Turns that need a frozen RAG context. mode "analyze" = aux LLM query; "question_en" = no LLM call."""
        reqs = []
        t = self.exp.tasks

        def add(task, item_id, lang, mode, recent, text, query=None):
            reqs.append({"task": task, "item_id": item_id, "lang": lang, "mode": mode, "recent": recent,
                         "text": text, "query": query})

        if t.rules_qa.enabled:
            for lang in t.rules_qa.languages:
                for item in self.items("rules_qa"):
                    add("rules_qa", item.id, lang, t.rules_qa.context_query, self.scenario_state(lang)["opening"],
                        item.question_uk if lang == "uk" else item.question_en, item.question_en)
        if t.retrieval.enabled and "uk_analyze" in t.retrieval.query_modes:
            for item in self.items("retrieval_items"):
                add("rules_qa", item.id, "uk", "analyze", self.scenario_state("uk")["opening"], item.question_uk)
        if t.adjudication.enabled:
            for a in self.items("adjudication"):
                add("adjudication", a.id, "uk", "analyze", a.scene, a.player_text)
        seen, out = set(), []
        for r in reqs:
            key = (r["mode"], r["task"], r["item_id"], r["lang"])
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out

    def missing_contexts(self) -> list[tuple[ExperimentSpec, dict]]:
        missing = []
        for spec in self.exp.rag_specs():
            have = {mode: self.contexts(spec, mode) for mode in ("analyze", "question_en")}
            missing += [(spec, r) for r in self.context_requests()
                        if (r["task"], r["item_id"], r["lang"]) not in have[r["mode"]]]
        return missing

    async def freeze_contexts(self) -> None:
        todo = self.missing_contexts()
        if not todo:
            return
        n_llm = sum(1 for _, r in todo if r["mode"] == "analyze")
        self.progress(f"freezing {len(todo)} RAG contexts ({n_llm} need the aux LLM)")

        def job(spec: ExperimentSpec, req: dict):
            async def run():
                scenario = self.scenario_state(req["lang"])
                msgs = [AIMessage(content=req["recent"]), HumanMessage(content=req["text"])]
                try:
                    if req["mode"] == "question_en":
                        query, errors = req["query"], []
                    else:
                        query, errors = await self.with_retries(lambda: search_query(self.deps, spec, scenario, msgs))
                    hits = await self.deps.retriever.search(query, spec.retrieval, game_system=self.pack.game_system)
                    self.context_store(spec, req["mode"]).append(
                        {"task": req["task"], "item_id": req["item_id"], "lang": req["lang"], "ok": True,
                         "mode": req["mode"], "query": query, "errors": errors, "hits": hits,
                         "signature": context_signature(spec)})
                except Exception as e:
                    log.exception("context failed")
                    self.progress(f"  context error {req['task']}/{req['item_id']}/{req['lang']}: {type(e).__name__}")
            return run

        await self.gather(job(spec, r) for spec, r in todo)

    # ------------------------------------------------------------------ task: retrieval (no generator)

    async def run_retrieval(self) -> None:
        cfg = self.exp.tasks.retrieval
        rag = self.exp.rag_specs()
        if not rag:
            self.progress("retrieval: no RAG arm, skipped")
            return
        spec = rag[0]
        done = self.done()
        ctx = self.contexts(spec, "analyze")
        jobs = []
        for mode in cfg.query_modes:
            for item in self.items("retrieval_items"):
                rec = self.base_record("retrieval", None, 0, item.id, lang=mode, condition=spec.retrieval.collection)
                if record_key(rec) in done:
                    continue
                if mode == "uk_analyze":
                    c = ctx.get(("rules_qa", item.id, "uk"))
                    if c is None:
                        continue
                    query = c["query"]
                elif mode == "uk_glossary":
                    query = load_glossary(str(GLOSSARY)).annotate(item.question_uk)
                else:
                    query = item.question_en if mode == "en" else item.question_uk

                def job(rec=rec, query=query):
                    async def fn():
                        hits = await self.deps.retriever.rank(query, spec.retrieval, game_system=self.pack.game_system,
                                                              limit=cfg.k)
                        return {"query": query, "hits": slim_hits(hits)}
                    return lambda: self.store_result(rec, fn)
                jobs.append(job())
        self.progress(f"retrieval: {len(jobs)} queries")
        await self.gather(jobs)

    # ------------------------------------------------------------------ task: rules QA (generation)

    async def generate(self, spec: ExperimentSpec, thread_id: str, lang: str, text: str, *,
                       forced_kind: str | None, forced_hits: list[dict] | None) -> dict:
        usage = UsageMetadataCallbackHandler()
        cfg = {"configurable": {"thread_id": thread_id}, "callbacks": [usage]}
        ctx = KeeperContext(spec=spec, campaign_id=thread_id)
        await self.graph.ainvoke({"bootstrap": {"scenario_id": self.exp.scenario_id, "language": lang}}, cfg,
                                 context=ctx)
        inputs: dict = {"messages": [HumanMessage(content=text)]}
        if forced_kind:
            inputs["forced_kind"] = forced_kind
        if spec.use_rag and forced_hits is not None:
            inputs["forced_hits"] = forced_hits
        out = await self.graph.ainvoke(inputs, cfg, context=ctx)
        msg = out["messages"][-1]
        cites = (out.get("citation_log") or [{}])[-1] if spec.use_rag else {}
        return {"answer": msg.text, "finish_reason": (msg.response_metadata or {}).get("finish_reason"),
                "citations": cites.get("citations", []), "invalid_markers": cites.get("invalid_markers", []),
                "hits": slim_hits(out.get("rules_hits") or []), "errors": out.get("errors") or [],
                "usage": usage.usage_metadata, "text_metrics": text_metrics(msg.text)}

    async def run_rules_qa(self) -> None:
        cfg = self.exp.tasks.rules_qa
        done = self.done()
        jobs = []
        for seed in self.exp.seeds:
            for spec in self.exp.specs(seed):
                ctx = self.contexts(spec, cfg.context_query) if spec.use_rag else {}
                for lang in cfg.languages:
                    for item in self.items("rules_qa"):
                        rec = self.base_record("rules_qa", spec, seed, item.id, lang)
                        if record_key(rec) in done:
                            continue
                        hits = None
                        if spec.use_rag:
                            c = ctx.get(("rules_qa", item.id, lang))
                            if c is None:
                                self.progress(f"  skip {record_key(rec)}: no frozen context")
                                continue
                            hits = c["hits"]
                            rec["search_query"] = c["query"]
                            rec["context_query"] = cfg.context_query
                        text = item.question_uk if lang == "uk" else item.question_en

                        def job(rec=rec, spec=spec, lang=lang, text=text, hits=hits):
                            fn = lambda: self.generate(spec, f"{record_key(rec)}#{time.monotonic_ns()}", lang, text,  # noqa: E731
                                                       forced_kind="rules_question", forced_hits=hits)
                            return lambda: self.store_result(rec, fn)
                        jobs.append(job())
        self.progress(f"rules_qa: {len(jobs)} answers to generate")
        await self.gather(jobs)

    # ------------------------------------------------------------------ task: knowledge-conflict probes

    async def run_conflict_probes(self) -> None:
        cfg = self.exp.tasks.conflict_probes
        done = self.done()
        jobs = []
        for seed in self.exp.seeds:
            for spec in self.exp.specs(seed):
                conditions = [c for c in cfg.conditions if (c == "no_context") != spec.use_rag]
                for lang in cfg.languages:
                    for probe in self.items("conflict_probes"):
                        for condition in conditions:
                            rec = self.base_record("conflict_probes", spec, seed, probe.id, lang, condition)
                            if record_key(rec) in done:
                                continue
                            hits = None
                            if condition != "no_context":
                                hits = [perturbed_hit(probe, self.sections, perturb=condition == "perturbed")]
                            text = probe.question_uk if lang == "uk" else probe.question_en

                            def job(rec=rec, spec=spec, lang=lang, text=text, hits=hits):
                                fn = lambda: self.generate(spec, f"{record_key(rec)}#{time.monotonic_ns()}", lang,  # noqa: E731
                                                           text, forced_kind="rules_question", forced_hits=hits)
                                return lambda: self.store_result(rec, fn)
                            jobs.append(job())
        self.progress(f"conflict_probes: {len(jobs)} answers to generate")
        await self.gather(jobs)

    # ------------------------------------------------------------------ task: adjudication

    async def run_adjudication(self) -> None:
        done = self.done()
        investigator = self.pack.investigator("uk")
        scenario = self.scenario_state("uk")
        jobs = []
        for seed in self.exp.seeds:
            for spec in self.exp.specs(seed):
                ctx = self.contexts(spec, "analyze") if spec.use_rag else {}
                for item in self.items("adjudication"):
                    rec = self.base_record("adjudication", spec, seed, item.id, "uk", item.variant)
                    if record_key(rec) in done:
                        continue
                    hits: list[dict] = []
                    if spec.use_rag:
                        c = ctx.get(("adjudication", item.id, "uk"))
                        if c is None:
                            self.progress(f"  skip {record_key(rec)}: no frozen context")
                            continue
                        hits = c["hits"]
                        rec["search_query"] = c["query"]

                    def job(rec=rec, spec=spec, item=item, hits=hits):
                        async def fn():
                            msgs = [AIMessage(content=item.scene), HumanMessage(content=item.player_text)]
                            pred, errors = await adjudicate_turn(self.deps, spec, scenario, investigator, [], msgs,
                                                                 hits)
                            return {"prediction": pred, "errors": errors, "group": item.group,
                                    "hits": slim_hits(hits)}
                        return lambda: self.store_result(rec, fn)
                    jobs.append(job())
        self.progress(f"adjudication: {len(jobs)} turns to adjudicate")
        await self.gather(jobs)

    # ------------------------------------------------------------------ LLM judge

    def judge_signature(self, task: str) -> str:
        version = self.exp.judge_prompt_versions[task]
        return f"{self.exp.judge}|{version}|{JUDGE_PROMPTS.hash(version)}"

    async def run_judge(self) -> None:
        if not self.exp.judge or not self.judge_enabled:
            self.progress("judge: disabled")
            return
        judged = {(j["record_key"], j["judge"]) for j in self.judgments.read() if j.get("status") == "ok"}
        qa = {i.id: i for i in self.datasets.rules_qa}
        probes = {p.id: p for p in self.datasets.conflict_probes}
        latest: dict[str, dict] = {}
        for r in self.records.read():
            if r["task"] in ("rules_qa", "conflict_probes") and r.get("status") == "ok":
                latest[record_key(r)] = r
        jobs = []
        for key, r in latest.items():
            sig = self.judge_signature(r["task"])
            if (key, sig) in judged:
                continue
            version = self.exp.judge_prompt_versions[r["task"]]

            def job(r=r, key=key, sig=sig, version=version):
                async def run():
                    rec = {"record_key": key, "task": r["task"], "judge": sig}
                    try:
                        if r["task"] == "rules_qa":
                            fn = lambda: judge_rules_answer(self.deps.llms, self.exp.judge, version,  # noqa: E731
                                                            qa[r["item_id"]], r["lang"], r["answer"])
                        else:
                            fn = lambda: judge_conflict_answer(self.deps.llms, self.exp.judge, version,  # noqa: E731
                                                               probes[r["item_id"]], r["lang"], r["answer"])
                        rec["judgment"] = await self.with_retries(fn)
                        rec["status"] = "ok"
                    except Exception as e:
                        log.exception("judge failed")
                        rec.update(status="error", error=f"{type(e).__name__}: {e}"[:500])
                    self.judgments.append(rec)
                    self.progress(f"  judge {rec['status']:<5} {key}")
                return run
            jobs.append(job())
        self.progress(f"judge: {len(jobs)} answers to grade with {self.exp.judge}")
        await self.gather(jobs)

    # ------------------------------------------------------------------ planning

    def plan(self) -> dict:
        """Planned LLM calls (upper bound, ignoring what is already done) for a dry run.

        The last line counts the calls that spend the Gemini API quota (the rest runs on local models).
        """
        t = self.exp.tasks
        arms = self.exp.specs(self.exp.seeds[0])
        n_seeds = len(self.exp.seeds)
        rag_sigs = len(self.exp.rag_specs())
        judge = bool(self.exp.judge and self.judge_enabled)

        def gemini(profile: str) -> bool:
            return self.deps.llms.profile(profile).provider == "google_genai"

        api = 0
        calls: dict[str, int] = {}
        missing = self.missing_contexts() if rag_sigs else []
        calls["frozen_contexts (aux analyze)"] = sum(1 for _, r in missing if r["mode"] == "analyze")
        api += sum(1 for spec, r in missing if r["mode"] == "analyze" and gemini(spec.aux))
        if t.retrieval.enabled:
            calls["retrieval (embeddings only)"] = 0
        if t.rules_qa.enabled:
            per_arm = len(self.items("rules_qa")) * len(t.rules_qa.languages) * n_seeds
            calls["rules_qa (narrate)"] = per_arm * len(arms)
            calls["rules_qa (judge)"] = per_arm * len(arms) if judge else 0
            api += per_arm * sum(1 for s in arms if gemini(s.generator))
        if t.conflict_probes.enabled:
            per_item = len(self.items("conflict_probes")) * len(t.conflict_probes.languages) * n_seeds
            conditions = {s.arm: sum(1 for c in t.conflict_probes.conditions if (c == "no_context") != s.use_rag)
                          for s in arms}
            calls["conflict_probes (narrate)"] = per_item * sum(conditions.values())
            calls["conflict_probes (judge)"] = per_item * sum(conditions.values()) if judge else 0
            api += per_item * sum(conditions[s.arm] for s in arms if gemini(s.generator))
        if t.adjudication.enabled:
            per_arm = len(self.items("adjudication")) * n_seeds
            calls["adjudication (adjudicator model)"] = per_arm * len(arms)
            api += per_arm * sum(1 for s in arms if gemini(s.aux if s.adjudicator == "aux" else s.generator))
        if judge and gemini(self.exp.judge):
            api += calls.get("rules_qa (judge)", 0) + calls.get("conflict_probes (judge)", 0)
        calls["total"] = sum(calls.values())
        calls["of which Gemini (API quota)"] = api
        return calls
