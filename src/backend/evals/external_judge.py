"""Grade a run with a judge that runs outside the harness (a model profile with provider "external").

Such a judge cannot be called from Python (for example Claude Code subagents), so the exact judge prompts are exported
to files, answered elsewhere and the judgments imported back into the run's judgments.jsonl, where metrics.py reads them
like any other. The judge sees only an opaque id, the schema to answer with and the prompt (the same prompt text and
schema as the harness's own judge, from evals/judge.py), never the arm, the condition or the record key; those stay in
map.json.

    cd src/backend
    uv run python -m evals.external_judge export evals/experiments/phase4-qwen-ft-claude-judge.yaml --batches 10
    # a judge answers every batch-XX.jsonl with batch-XX.out.jsonl: {"id": ..., "judgment": {...}} per line, the
    # judgment being an object of the schema named in the input line (schemas.json)
    uv run python -m evals.external_judge import evals/experiments/phase4-qwen-ft-claude-judge.yaml
    uv run python -m evals.run evals/experiments/phase4-qwen-ft-claude-judge.yaml --summarize-only

Files go to <run>/external_judge/<judge profile>/: batch-XX.jsonl (the judge's input), schemas.json, and the private
map.json (id -> record key, task, item, ...; the judge must not read it). All answers of one (task, item, language)
land in one batch, so the arms and conditions of an item are graded by the same judge, and the items of a batch are
shuffled.

`import` validates every judgment with the pydantic model of the item, applies the harness's own post-processing and
appends records shaped like the harness's (plus an "external" field naming the batch and export). It is idempotent,
imports the valid lines even when others are bad, and exits non-zero while problems or missing ids remain. For a second
pass, run `export --force` after importing: it exports only the answers still not graded (the previous export is
archived).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from keeper.deps import Deps
from keeper.experiments import ExperimentRegistry
from keeper.llm import ModelFactory
from keeper.prompts import PromptSet
from keeper.rag.corpus import load_corpus
from keeper.runlog import RunLog
from keeper.scenarios import ScenarioRepo
from server.settings import BACKEND_DIR, REPO_ROOT, get_settings

from .datasets import EvalDatasets
from .experiment import EvalExperiment
from .harness import Harness, record_key, sha16
from .judge import (
    PROMPTS,
    ConflictJudgment,
    QAJudgment,
    conflict_prompt,
    conflict_result,
    rules_prompt,
    rules_result,
)

EXPORT_DIRNAME = "external_judge"
SCHEMAS: dict[str, type[BaseModel]] = {"QAJudgment": QAJudgment, "ConflictJudgment": ConflictJudgment}
SCHEMA_OF_TASK = {"rules_qa": "QAJudgment", "conflict_probes": "ConflictJudgment"}
MISSING_SHOWN = 25   # ids listed per batch in the report of missing ids


def export_dir(run_dir: Path, judge: str) -> Path:
    return run_dir / EXPORT_DIRNAME / judge


def batch_name(index: int, count: int) -> str:
    return f"batch-{index:0{max(2, len(str(count - 1)))}d}"


def write_lines(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def require_external(h: Harness) -> str:
    """The judge profile of the experiment, which must be an external one."""
    if not h.exp.judge:
        raise SystemExit("the experiment has no judge")
    if not h.judge_is_external:
        raise SystemExit(f"judge profile {h.exp.judge!r} is not external (provider "
                         f"{h.deps.llms.profile(h.exp.judge).provider!r}): the harness grades it itself, use "
                         "python -m evals.run --judge-only")
    return h.exp.judge


# ------------------------------------------------------------------------------------------ export


def plan_batches(records: list[dict], n_batches: int, seed: int) -> list[list[dict]]:
    """Records in `n_batches` batches (fewer when there are fewer groups), reproducible for a seed.

    A group is every answer to one (task, item, language): the arms and conditions of an item stay together, so a paired
    contrast is graded by the same judge. Groups go largest first to the batch with the fewest items (balanced), and the
    items of each batch are shuffled, so the position in a batch says nothing about the arm.
    """
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in sorted(records, key=record_key):   # canonical: the order of the records file does not matter
        groups[(r["task"], r["item_id"], r["lang"])].append(r)
    rng = random.Random(seed)
    order = sorted(groups)
    rng.shuffle(order)
    order.sort(key=lambda g: -len(groups[g]))   # stable: ties stay in the shuffled order
    batches: list[list[dict]] = [[] for _ in range(max(1, min(n_batches, len(order))))]
    for g in order:
        min(batches, key=len).extend(groups[g])
    for batch in batches:
        rng.shuffle(batch)
    return batches


def export_files(directory: Path) -> list[Path]:
    """The files of an export: the batches and their outputs, the schemas and the map."""
    return [p for p in [*sorted(directory.glob("batch-*.jsonl")), directory / "schemas.json", directory / "map.json"]
            if p.is_file()]


def archive_export(directory: Path) -> Path:
    """Moves the files of an earlier export into archive/<its tag>/ (never deletes), returns the new place."""
    files = export_files(directory)
    try:
        tag = json.loads((directory / "map.json").read_text(encoding="utf-8"))["export"]
    except (OSError, ValueError, KeyError):
        tag = "unknown"
    target = directory / "archive" / tag
    n = 1
    while target.exists():
        n += 1
        target = directory / "archive" / f"{tag}-{n}"
    target.mkdir(parents=True)
    for p in files:
        p.replace(target / p.name)
    return target


def export_batches(h: Harness, n_batches: int, seed: int = 7, force: bool = False) -> list[Path]:
    """Writes the judge prompts of every answer without a judgment under the current signature; returns the batches."""
    judge = require_external(h)
    if n_batches < 1:
        raise SystemExit("--batches must be at least 1")
    pending = h.ungraded()
    if not pending:
        h.progress("nothing to export: every answer of the run already has a judgment with this signature")
        return []
    directory = export_dir(h.run_dir, judge)
    existing = bool(export_files(directory))
    if existing and not force:
        raise SystemExit(f"{directory} already holds an export: run `import` first, then `export --force` (the old "
                         "export is archived, never deleted)")
    if existing:
        archived = archive_export(directory)
        h.progress(f"previous export moved to {archived} (outputs not imported yet are not read there)")
    signatures = {t: h.judge_signature(t) for t in SCHEMA_OF_TASK}
    qa = {i.id: i for i in h.datasets.rules_qa}
    probes = {p.id: p for p in h.datasets.conflict_probes}
    by_key = {key: r for key, r, _ in pending}
    batches = plan_batches(list(by_key.values()), n_batches, seed)
    # The tag is part of every id: the ids of two exports never coincide unless their content does too (same answers,
    # seed and batch count, hence the same mapping), so a late output of an old export cannot be mapped to a new item.
    tag = sha16({"judge": judge, "signatures": signatures, "seed": seed, "batches": len(batches),
                 "keys": sorted(by_key)})[:8]
    items: dict[str, dict] = {}
    paths, number = [], 0
    directory.mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(batches):
        name, lines = batch_name(index, len(batches)), []
        for r in batch:
            version = h.exp.judge_prompt_versions[r["task"]]
            info = {"record_key": record_key(r), "task": r["task"], "item_id": r["item_id"], "lang": r["lang"],
                    "prompt_version": version, "schema": SCHEMA_OF_TASK[r["task"]], "batch": name}
            if r["task"] == "rules_qa":
                prompt = rules_prompt(qa[r["item_id"]], r["lang"], r["answer"], version)
            else:
                prompt, info["order_swapped"] = conflict_prompt(probes[r["item_id"]], r["lang"], r["answer"], version)
            number += 1
            item_id = f"{tag}-{number:04d}"
            items[item_id] = info
            lines.append({"id": item_id, "schema": info["schema"], "prompt": prompt})
        path = directory / f"{name}.jsonl"
        write_lines(path, lines)
        paths.append(path)
        h.progress(f"{path}  {len(lines)} items  {sum(len(x['prompt']) for x in lines)} chars")
    write_json(directory / "schemas.json", {name: model.model_json_schema() for name, model in SCHEMAS.items()})
    write_json(directory / "map.json", {
        "export": tag, "created_at": datetime.now(UTC).isoformat(timespec="seconds"), "judge": judge, "seed": seed,
        "batches": len(batches), "signatures": signatures, "dataset_hashes": h.datasets.hashes, "items": items})
    h.progress(f"export {tag}: {number} items in {len(batches)} batches (seed {seed}). Give each judge ONE batch file "
               f"and schemas.json; map.json stays private. Answers: batch-XX.out.jsonl, then `import`.")
    return paths


# ------------------------------------------------------------------------------------------ import


@dataclass
class ImportResult:
    imported: int = 0
    already_graded: int = 0
    problems: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)          # exported ids still without a judgment in the run
    batch_of: dict[str, str] = field(default_factory=dict)    # id -> batch file name, for every exported id

    @property
    def ok(self) -> bool:
        return not self.problems and not self.missing


def load_map(h: Harness) -> tuple[Path, dict]:
    directory = export_dir(h.run_dir, require_external(h))
    path = directory / "map.json"
    if not path.is_file():
        raise SystemExit(f"{path} not found: run `export` first")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("judge") != h.exp.judge:
        raise SystemExit(f"{path} is an export for judge {meta.get('judge')!r}, not {h.exp.judge!r}")
    signatures = {t: h.judge_signature(t) for t in SCHEMA_OF_TASK}
    if meta.get("signatures") != signatures:
        raise SystemExit("the judge prompts changed since the export (signatures differ): the judgments would not "
                         f"belong to the prompts in use. Exported {meta.get('signatures')}, now {signatures}")
    if meta.get("dataset_hashes") != h.datasets.hashes:
        raise SystemExit("the gold datasets changed since the export (hashes differ)")
    return directory, meta


def reason(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in err['loc']) or 'judgment'}: {err['msg']}" for err in e.errors())


def read_outputs(directory: Path, items: dict[str, dict]) -> tuple[dict[str, BaseModel], list[str]]:
    """id -> validated judgment of every usable line of batch-*.out.jsonl, and a description of every other line.

    Not usable: invalid JSON, a line that is not {"id", "judgment"}, an unknown id, a judgment that does not fit the
    schema of its item. An id that occurs more than once is not imported at all (which of the copies is meant?).
    """
    problems: list[str] = []
    seen: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for path in sorted(directory.glob("batch-*.out.jsonl")):
        for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{path.name}:{n}"
            try:
                row = json.loads(line)
            except ValueError as e:
                problems.append(f"{where}: invalid JSON ({e})")
                continue
            if not isinstance(row, dict) or not isinstance(row.get("judgment"), dict) or "id" not in row:
                problems.append(f'{where}: a line must be an object with "id" and a "judgment" object')
            elif not isinstance(row["id"], str) or row["id"] not in items:
                problems.append(f"{where}: unknown id {row['id']!r}")
            else:
                seen[row["id"]].append((where, row["judgment"]))
    valid: dict[str, BaseModel] = {}
    for item_id, copies in seen.items():
        if len(copies) > 1:
            problems.append(f"duplicate id {item_id} ({', '.join(w for w, _ in copies)}): none of the copies imported")
            continue
        where, judgment = copies[0]
        try:
            valid[item_id] = SCHEMAS[items[item_id]["schema"]].model_validate(judgment)
        except ValidationError as e:
            problems.append(f"{where}: id {item_id} is not a {items[item_id]['schema']}: {reason(e)}")
    return valid, problems


def record_session(h: Harness, meta: dict, imported: int) -> None:
    """Notes in config.json which judge graded the run later (its `experiment` block keeps the first judge)."""
    path = h.paths.config
    if not imported or not path.is_file():
        return
    config = json.loads(path.read_text(encoding="utf-8"))
    config.setdefault("sessions", []).append({
        "judge": {"profile": h.deps.llms.profile(h.exp.judge).snapshot(),
                  "prompt_hashes": {p: PROMPTS.hash(p) for p in sorted(h.exp.judge_prompt_versions.values())}},
        "external_judge": {"export": meta["export"], "seed": meta["seed"], "batches": meta["batches"],
                           "imported": imported, "ts": datetime.now(UTC).isoformat(timespec="seconds")}})
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def import_judgments(h: Harness) -> ImportResult:
    directory, meta = load_map(h)
    items = meta["items"]
    valid, problems = read_outputs(directory, items)
    result = ImportResult(problems=problems, batch_of={i: info["batch"] for i, info in items.items()})
    qa = {i.id: i for i in h.datasets.rules_qa}
    probes = {p.id: p for p in h.datasets.conflict_probes}
    judged = {(j["record_key"], j["judge"]) for j in h.judgments.read() if j.get("status") == "ok"}
    for item_id, info in items.items():
        if item_id not in valid:
            continue
        sig = meta["signatures"][info["task"]]
        if (info["record_key"], sig) in judged:
            result.already_graded += 1
            continue
        if info["task"] == "rules_qa":
            judgment = rules_result(qa[info["item_id"]], valid[item_id])
        else:
            judgment = conflict_result(probes[info["item_id"]], valid[item_id])
        h.judgments.append({"record_key": info["record_key"], "task": info["task"], "judge": sig,
                            "judgment": judgment, "status": "ok",
                            "external": {"batch": info["batch"], "export": meta["export"]}})
        judged.add((info["record_key"], sig))
        result.imported += 1
    result.missing = [i for i, info in items.items() if (info["record_key"], meta["signatures"][info["task"]])
                      not in judged]
    record_session(h, meta, result.imported)
    return result


def report_import(h: Harness, result: ImportResult) -> None:
    h.progress(f"imported {result.imported} judgments ({result.already_graded} were already graded)")
    for problem in result.problems:
        h.progress(f"ERROR {problem}")
    if result.missing:
        by_batch: dict[str, list[str]] = defaultdict(list)
        for item_id in result.missing:
            by_batch[result.batch_of[item_id]].append(item_id)
        h.progress(f"{len(result.missing)} ids still without a valid judgment:")
        for name, ids in sorted(by_batch.items()):
            more = f" (+{len(ids) - MISSING_SHOWN} more)" if len(ids) > MISSING_SHOWN else ""
            h.progress(f"  {name}: {len(ids)} missing: {', '.join(ids[:MISSING_SHOWN])}{more}")
        h.progress("for a second pass: `export --force` now exports only the answers still not graded")
    elif not result.problems:
        h.progress("every exported answer is graded; rebuild the report with `evals.run ... --summarize-only`")


# ------------------------------------------------------------------------------------------ CLI


def open_harness(experiment: Path, run_name: str | None) -> Harness:
    """The experiment's harness on an existing run, after checking that the run belongs to the experiment."""
    s = get_settings()
    exp = EvalExperiment.load(experiment)
    run_dir = s.artifacts_dir / "eval" / "runs" / (run_name or exp.name)
    if not (run_dir / "config.json").is_file():
        raise SystemExit(f"{run_dir} is not a run (no config.json); pass --run-name")
    datasets = EvalDatasets.load(s.data_dir / exp.dataset_dir)
    chunks = load_corpus([s.data_dir / d for d in exp.corpus_dirs], chunk_size=1000, overlap=200, root=REPO_ROOT)
    # No Qdrant client and no API key: nothing here calls a model, the deps only serve the configuration snapshot
    specs = exp.specs(exp.seeds[0])
    deps = Deps(llms=ModelFactory.from_yaml(s.config_dir / "models.yaml"), prompts=PromptSet(),
                scenarios=ScenarioRepo(s.data_dir / "scenarios"),
                experiments=ExperimentRegistry({x.arm: x for x in specs}, specs[0].arm),
                runlog=RunLog(s.artifacts_dir / "runs", enabled=False), retriever=None)
    h = Harness(exp=exp, deps=deps, datasets=datasets, chunks=chunks, run_dir=run_dir,
                contexts_dir=s.artifacts_dir / "eval" / "contexts")
    old = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    fingerprint = h.config_snapshot()["fingerprint"]
    if old.get("fingerprint") != fingerprint:
        raise SystemExit(f"{run_dir} belongs to a different configuration (fingerprint {old.get('fingerprint')} != "
                         f"{fingerprint}); use the experiment file of the run or another --run-name")
    return h


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("experiment", type=Path, help="e.g. evals/experiments/phase4-qwen-ft-claude-judge.yaml")
    common.add_argument("--run-name", default=None, help="defaults to the experiment name")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", parents=[common], help="write the judge prompts of the ungraded answers")
    e.add_argument("--batches", type=int, required=True, help="number of batch files (one judge each)")
    e.add_argument("--seed", type=int, default=7, help="seed of the batch assignment and shuffles (default 7)")
    e.add_argument("--force", action="store_true", help="replace an existing export (the old one is archived)")
    sub.add_parser("import", parents=[common], help="validate batch-XX.out.jsonl and append the judgments to the run")
    return p


def main() -> None:
    a = build_parser().parse_args()
    if not a.experiment.is_absolute() and not a.experiment.exists():
        a.experiment = BACKEND_DIR / a.experiment
    h = open_harness(a.experiment, a.run_name)
    if a.cmd == "export":
        export_batches(h, a.batches, a.seed, a.force)
        return
    result = import_judgments(h)
    report_import(h, result)
    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
