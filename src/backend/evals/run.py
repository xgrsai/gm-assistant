"""Run an evaluation experiment.

    cd src/backend
    uv run python -m evals.run evals/experiments/phase3-gemini.yaml --dry-run        # planned LLM calls
    uv run python -m evals.run evals/experiments/phase3-gemini.yaml --tasks retrieval  # embeddings only, free
    uv run python -m evals.run evals/experiments/phase3-gemini.yaml --limit 5          # smoke run
    uv run python -m evals.run evals/experiments/phase3-gemini.yaml                    # full run (resumable)
    uv run python -m evals.run evals/experiments/phase3-gemini.yaml --summarize-only
    uv run python -m evals.run evals/experiments/phase3-gemini.yaml --judge-only      # grade existing answers

A judge whose profile has provider "external" (claude-sonnet-agent) is never called: --judge-only and the last step
of a full run only say so, the report shows what is not graded yet, and the judgments come from
`python -m evals.external_judge`.

A profile with a model card (the Phase 4 models) needs a card whose adapter check passed (--allow-failed-card to
override, recorded) and must be served by Ollama under the digest in the card, or the run stops before generating;
--dry-run only warns.

Outputs go to artifacts/eval/runs/<run-name>/ (config.json, records.jsonl, judgments.jsonl, metrics.json,
report.md). Rerunning the same command resumes: finished items are skipped and failed ones retried.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from qdrant_client import AsyncQdrantClient

from keeper.deps import Deps, LLMs
from keeper.experiments import ExperimentRegistry
from keeper.llm import ModelFactory
from keeper.prompts import PromptSet
from keeper.rag.corpus import load_corpus
from keeper.rag.embeddings import build_embeddings, load_profiles
from keeper.rag.retriever import Retriever
from keeper.runlog import RunLog
from keeper.scenarios import ScenarioRepo
from server.main import detect_git_sha
from server.settings import BACKEND_DIR, REPO_ROOT, Settings, get_settings

from . import model_cards
from .datasets import EvalDatasets, validate
from .experiment import EvalExperiment
from .harness import Harness
from .metrics import summarize

TASKS = ("retrieval", "rules_qa", "conflict_probes", "adjudication")


def needs_retriever(exp: EvalExperiment) -> bool:
    t = exp.tasks
    return bool(exp.rag_specs()) and (t.retrieval.enabled or t.rules_qa.enabled or t.adjudication.enabled)


def build_eval_deps(s: Settings, exp: EvalExperiment) -> Deps:
    llms = ModelFactory.from_yaml(s.config_dir / "models.yaml", google_api_key=s.google_api_key,
                                  ollama_url=s.ollama_base_url, vllm_url=s.vllm_base_url,
                                  timeout=s.model_timeout_s, google_rpm=s.google_rpm)
    specs = exp.specs(exp.seeds[0])
    used = {x for spec in specs for x in (spec.generator, spec.aux)} | ({exp.judge} if exp.judge else set())
    for name in used:
        if llms.profile(name).provider == "google_genai" and not s.google_api_key:
            raise SystemExit(f"GOOGLE_API_KEY is required by model profile {name!r}")
    retriever = None
    if needs_retriever(exp):
        profiles = load_profiles(s.config_dir / "embedders.yaml")
        retriever = Retriever(AsyncQdrantClient(url=s.qdrant_url),
                              {n: build_embeddings(p, ollama_url=s.ollama_base_url) for n, p in profiles.items()},
                              {n: p.dim for n, p in profiles.items()})
    return Deps(llms=llms, prompts=PromptSet(), scenarios=ScenarioRepo(s.data_dir / "scenarios"),
                experiments=ExperimentRegistry({x.arm: x for x in specs}, specs[0].arm),
                runlog=RunLog(s.artifacts_dir / "runs", enabled=False), retriever=retriever)


class CardCheck(NamedTuple):
    """What the model card check does in a given mode of the CLI."""

    enabled: bool     # run the check at all
    ollama: bool      # also compare the digests with the Ollama server
    warn_only: bool   # print the problems instead of stopping


def card_check_mode(*, summarize_only: bool, dry_run: bool, judge_only: bool) -> CardCheck:
    """--summarize-only touches no model: no check. --dry-run only warns. --judge-only generates nothing, so it needs
    the cards (the run fingerprint is built from them) but not Ollama. A real run checks everything and stops."""
    if summarize_only:
        return CardCheck(enabled=False, ollama=False, warn_only=False)
    return CardCheck(enabled=True, ollama=dry_run or not judge_only, warn_only=dry_run)


def check_model_cards(llms: LLMs, exp: EvalExperiment, *, root: Path = REPO_ROOT,
                      listing: Callable[[], Mapping[str, str]] | None = None, warn_only: bool = False,
                      allow_failed_card: bool = False) -> None:
    """Fail fast when a profile with a model card (the fine-tuned model or its base) is not the model on record.

    The card must exist and say its adapter check passed (`allow_failed_card` turns a failed one into a warning),
    and with `listing` (Ollama's tag -> digest) every such Ollama model must be served under the digest in the card,
    so a re-created or re-pulled tag cannot silently change the results. A dry run (`warn_only`) prints the problems
    instead of stopping.
    """
    names = {x for spec in exp.specs(exp.seeds[0]) for x in (spec.generator, spec.aux)}
    errors, warnings = model_cards.check_cards(llms, names, root=root, listing=listing, allow_failed=allow_failed_card)
    if errors and not warn_only:
        raise SystemExit("model card check failed:\n" + "\n".join(f"- {p}" for p in errors))
    for p in [*errors, *warnings]:
        print(f"WARNING: {p}")


def apply_overrides(exp: EvalExperiment, tasks: list[str] | None, limit: int | None,
                    query_modes: list[str] | None = None) -> EvalExperiment:
    if query_modes is not None:
        exp.tasks.retrieval.query_modes = query_modes
    for name in TASKS:
        cfg = getattr(exp.tasks, name)
        if tasks is not None:
            cfg.enabled = name in tasks
        if limit is not None:
            cfg.limit = limit if cfg.limit is None else min(cfg.limit, limit)
    return exp


async def run(args: argparse.Namespace) -> None:
    s = get_settings()
    exp = apply_overrides(EvalExperiment.load(args.experiment), args.tasks, args.limit, args.query_modes)
    run_name = args.run_name or exp.name
    datasets = EvalDatasets.load(s.data_dir / exp.dataset_dir)
    chunks = load_corpus([s.data_dir / d for d in exp.corpus_dirs], chunk_size=1000, overlap=200, root=REPO_ROOT)
    deps = build_eval_deps(s, exp)
    mode = card_check_mode(summarize_only=args.summarize_only, dry_run=args.dry_run, judge_only=args.judge_only)
    if mode.enabled:
        check_model_cards(deps.llms, exp, warn_only=mode.warn_only, allow_failed_card=args.allow_failed_card,
                          listing=(lambda: model_cards.ollama_listing(s.ollama_base_url)) if mode.ollama else None)
    problems = validate(datasets, chunks, deps.scenarios.load(exp.scenario_id).investigator("uk"))
    if problems and not args.allow_invalid_datasets:
        raise SystemExit("dataset validation failed:\n" + "\n".join(problems))

    harness = Harness(exp=exp, deps=deps, datasets=datasets, chunks=chunks,
                      run_dir=s.artifacts_dir / "eval" / "runs" / run_name,
                      contexts_dir=s.artifacts_dir / "eval" / "contexts",
                      concurrency=args.concurrency, judge_enabled=not args.no_judge,
                      allow_failed_card=args.allow_failed_card)
    judge_sigs = {t: harness.judge_signature(t) for t in exp.judge_prompt_versions} if exp.judge else {}

    if args.summarize_only:
        summarize(harness.run_dir, datasets, judge_sigs, external_judge=harness.judge_is_external)
        print(f"report: {harness.run_dir / 'report.md'}")
        return
    if args.dry_run:
        for k, v in harness.plan().items():
            print(f"{k:<34} {v}")
        return

    harness.bind_run_dir(extra={"git_sha": detect_git_sha(), "created_at": datetime.now(UTC).isoformat()})
    try:
        if deps.retriever is not None and not args.judge_only:
            for spec in exp.rag_specs():
                await deps.retriever.verify(spec.retrieval)
            await harness.freeze_contexts()
        t = exp.tasks
        if args.judge_only:
            await harness.run_judge()
            summarize(harness.run_dir, datasets, judge_sigs, external_judge=harness.judge_is_external)
            print(f"report: {harness.run_dir / 'report.md'}")
            return
        if t.retrieval.enabled:
            await harness.run_retrieval()
        if t.rules_qa.enabled:
            await harness.run_rules_qa()
        if t.conflict_probes.enabled:
            await harness.run_conflict_probes()
        if t.adjudication.enabled:
            await harness.run_adjudication()
        await harness.run_judge()
    finally:
        if deps.retriever is not None:
            await deps.retriever.client.close()
    summarize(harness.run_dir, datasets, judge_sigs, external_judge=harness.judge_is_external)
    print(f"report: {harness.run_dir / 'report.md'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", type=Path, help="e.g. evals/experiments/phase3-gemini.yaml")
    p.add_argument("--tasks", type=lambda v: [x.strip() for x in v.split(",") if x.strip()], default=None,
                   help=f"comma-separated subset of {','.join(TASKS)}")
    p.add_argument("--limit", type=int, default=None, help="max items per task (groups for adjudication)")
    p.add_argument("--query-modes", type=lambda v: [x.strip() for x in v.split(",") if x.strip()], default=None,
                   help="retrieval query modes to run, e.g. en,uk (uk_analyze needs the aux LLM; uk_glossary does not)")
    p.add_argument("--run-name", default=None, help="defaults to the experiment name")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--judge-only", action="store_true", help="only grade answers already in the run, then report")
    p.add_argument("--allow-invalid-datasets", action="store_true")
    p.add_argument("--allow-failed-card", action="store_true",
                   help="run even though a model card says its adapter check failed (recorded in config.json)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    p = build_parser()
    a = p.parse_args()
    if a.query_modes and set(a.query_modes) - {"en", "uk", "uk_analyze", "uk_glossary"}:
        p.error("query modes must be en, uk, uk_analyze or uk_glossary")
    if a.tasks and set(a.tasks) - set(TASKS):
        p.error(f"unknown task(s): {sorted(set(a.tasks) - set(TASKS))}")
    if not a.experiment.is_absolute() and not a.experiment.exists():
        a.experiment = BACKEND_DIR / a.experiment
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
