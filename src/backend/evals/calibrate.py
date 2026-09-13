"""Validate the LLM judge against human labels (Cohen's kappa) before reporting judge-based numbers.

    cd src/backend
    uv run python -m evals.calibrate export phase3-gemini --n 40        # blinded CSV for a human rater
    # fill the human_verdict column: correct | partially_correct | incorrect (open in Excel/LibreOffice, UTF-8)
    uv run python -m evals.calibrate score phase3-gemini

The CSV hides arm names, strips [R#] citation markers and shuffles rows (deterministically), so the rater cannot
tell prompt-only from RAG answers by their form.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from keeper.citations import BRACKET
from server.settings import get_settings

from .datasets import EvalDatasets
from .experiment import EvalExperiment
from .harness import JsonlStore, record_key
from .judge import PROMPTS
from .stats import cohen_kappa

VERDICTS = ("correct", "partially_correct", "incorrect")


def run_dir(name: str) -> Path:
    return get_settings().artifacts_dir / "eval" / "runs" / name


def judge_sig(config: dict) -> str:
    exp = EvalExperiment(**config["experiment"])
    version = exp.judge_prompt_versions["rules_qa"]
    return f"{exp.judge}|{version}|{PROMPTS.hash(version)}"


def export(name: str, n: int, seed: int) -> Path:
    d = run_dir(name)
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    s = get_settings()
    ds = EvalDatasets.load(s.data_dir / config["experiment"]["dataset_dir"])
    items = {i.id: i for i in ds.rules_qa}
    sig = judge_sig(config)
    judged = {j["record_key"] for j in JsonlStore(d / "judgments.jsonl").read()
              if j.get("status") == "ok" and j["judge"] == sig}
    records = {record_key(r): r for r in JsonlStore(d / "records.jsonl").read()
               if r.get("status") == "ok" and r["task"] == "rules_qa"}
    keys = sorted(k for k in records if k in judged)
    random.Random(seed).shuffle(keys)
    keys = keys[:n]
    mapping = {}
    out = d / "human_labels.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "question", "answerable", "reference_answer", "key_facts", "answer",
                    "human_verdict", "notes"])
        for k in keys:
            r = records[k]
            item = items[r["item_id"]]
            sid = hashlib.sha256(k.encode()).hexdigest()[:10]
            mapping[sid] = k
            question = item.question_uk if r["lang"] == "uk" else item.question_en
            w.writerow([sid, question, item.answerable, item.reference_answer, " | ".join(item.key_facts),
                        BRACKET.sub("", r["answer"]), "", ""])
    (d / "human_labels.map.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"wrote {len(keys)} rows to {out}")
    return out


def score(name: str) -> dict:
    d = run_dir(name)
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    sig = judge_sig(config)
    mapping = json.loads((d / "human_labels.map.json").read_text(encoding="utf-8"))
    judge = {j["record_key"]: j["judgment"]["verdict"] for j in JsonlStore(d / "judgments.jsonl").read()
             if j.get("status") == "ok" and j["judge"] == sig}
    human, model = [], []
    with open(d / "human_labels.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = row["human_verdict"].strip().lower().replace(" ", "_")
            if not v:
                continue
            if v not in VERDICTS:
                raise SystemExit(f"{row['sample_id']}: human_verdict must be one of {VERDICTS}, got {v!r}")
            human.append(v)
            model.append(judge[mapping[row["sample_id"]]])
    if not human:
        raise SystemExit("no human labels yet")
    binary_h = ["correct" if v == "correct" else "not" for v in human]
    binary_m = ["correct" if v == "correct" else "not" for v in model]
    result = {
        "n": len(human),
        "kappa_3way": round(cohen_kappa(human, model), 3),
        "kappa_binary": round(cohen_kappa(binary_h, binary_m), 3),
        "agreement": round(sum(a == b for a, b in zip(human, model, strict=True)) / len(human), 3),
        "confusion": {f"human={h} judge={m}": c
                      for (h, m), c in sorted(Counter(zip(human, model, strict=True)).items())},
    }
    (d / "judge_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("run")
    e.add_argument("--n", type=int, default=40)
    e.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score")
    s.add_argument("run")
    a = p.parse_args()
    if a.cmd == "export":
        export(a.run, a.n, a.seed)
    else:
        score(a.run)


if __name__ == "__main__":
    main()
