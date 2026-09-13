"""Aggregate eval records into metrics.json and a Markdown report (no LLM calls; safe to rerun any time)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .datasets import AdjudicationItem, ConflictProbe, EvalDatasets, RulesQAItem
from .harness import JsonlStore, record_key
from .stats import bootstrap_ci, holm, mcnemar_exact, mean, paired_diff


def latest_ok(records: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in records:
        if r.get("status") == "ok":
            out[record_key(r)] = r
    return out


def rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def ci(xs: list[float]) -> list[float] | None:
    if not xs:
        return None
    lo, hi = bootstrap_ci(xs, seed=1)
    return [round(lo, 4), round(hi, 4)]


def rnd(x: float | None, n: int = 4) -> float | None:
    return None if x is None or x != x else round(x, n)


# ------------------------------------------------------------------------------------------ retrieval


RECALL_KEEP = 0.95  # keep at least 95% of the recall reachable without a threshold


def retrieval_metrics(records: list[dict], items: dict[str, RulesQAItem], thresholds: list[float],
                      current_min_score: float | None, calibration_k: int = 5) -> dict:
    """Ranking metrics over the top-k list, plus a min_score calibration over the top `calibration_k` hits
    (the number of passages the production retriever returns: the sum of the source-type quotas)."""
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_mode[r["lang"]].append(r)
    out = {}
    for mode, recs in sorted(by_mode.items()):
        rows, by_qt = [], defaultdict(list)
        pairs, unanswerable_max = [], []
        for r in recs:
            item = items.get(r["item_id"])
            if item is None:
                continue
            hits = r["hits"]
            if not item.answerable:
                unanswerable_max.append(max((h["score"] for h in hits), default=0.0))
                continue
            gold = {g.key() for g in item.gold_sections}
            keys = [(h["source"], h["section"]) for h in hits]
            rank = next((i for i, k in enumerate(keys, start=1) if k in gold), None)
            row = {"hit@1": rank == 1, "hit@3": bool(rank and rank <= 3), "hit@5": bool(rank and rank <= 5),
                   "recall@5": len(gold & set(keys[:5])) / len(gold), "mrr": 1 / rank if rank else 0.0}
            rows.append(row)
            by_qt[item.query_type].append(row)
            pairs.append((gold, [(k, h["score"]) for k, h in zip(keys, hits, strict=True)]))

        def agg(rs):
            return {m: rnd(mean([float(x[m]) for x in rs])) for m in ("hit@1", "hit@3", "hit@5", "recall@5", "mrr")} | \
                   {"n": len(rs)}

        calib = []
        for t in thresholds:
            kept = relevant = 0
            recalls = []
            for gold, scored in pairs:
                keep = [k for k, s in scored[:calibration_k] if s >= t]
                kept += len(keep)
                relevant += sum(1 for k in keep if k in gold)
                recalls.append(len(gold & set(keep)) / len(gold))
            precision = relevant / kept if kept else 0.0
            recall = mean(recalls) if recalls else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
            empty = rate([m < t for m in unanswerable_max])
            calib.append({"threshold": t, "precision": rnd(precision), "recall": rnd(recall), "f1": rnd(f1),
                          "f2": rnd(f2), "unanswerable_no_hits": empty})
        best_f1 = max(calib, key=lambda c: (c["f1"], c["threshold"])) if calib else None
        best_f2 = max(calib, key=lambda c: (c["f2"], c["threshold"])) if calib else None
        # a RAG generator can ignore an irrelevant passage but cannot use a missing one: prefer recall
        max_recall = max((c["recall"] for c in calib), default=0.0)
        recall_ok = [c for c in calib if c["recall"] >= RECALL_KEEP * max_recall]
        recall_target = max(recall_ok, key=lambda c: c["threshold"]) if recall_ok else None
        current = next((c for c in calib if current_min_score is not None
                        and abs(c["threshold"] - current_min_score) < 1e-9), None)
        out[mode] = {"overall": agg(rows), "by_query_type": {q: agg(rs) for q, rs in sorted(by_qt.items())},
                     "unanswerable_top_score_mean": rnd(mean(unanswerable_max)) if unanswerable_max else None,
                     "calibration_k": calibration_k, "calibration": calib, "best_f1": best_f1, "best_f2": best_f2,
                     "highest_threshold_keeping_recall": recall_target, "current_threshold": current}
    return out


# ------------------------------------------------------------------------------------------ rules QA


def rules_qa_metrics(records: list[dict], judgments: dict[str, dict], items: dict[str, RulesQAItem],
                     arm_order: list[str]) -> dict:
    per_arm: dict[str, dict] = {}
    paired: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for r in records:
        item = items.get(r["item_id"])
        if item is None:
            continue
        j = judgments.get(record_key(r))
        gold = {g.key() for g in item.gold_sections}
        hit_keys = {(h["source"], h["section"]) for h in r.get("hits") or []}
        cite_keys = {(c.get("source"), c.get("section")) for c in r.get("citations") or []}
        row = {
            "answerable": item.answerable,
            "judged": j is not None,
            "score": j["score"] if j else None,
            "strict": (j["verdict"] == "correct") if j else None,
            "coverage": j.get("key_fact_coverage") if j else None,
            "contradicts": j.get("contradicts_reference") if j else None,
            "abstained": j.get("abstained") if j else None,
            "lang_ok": (r["text_metrics"]["cyrillic_ratio"] >= 0.5) if r["lang"] == "uk" else
                       (r["text_metrics"]["cyrillic_ratio"] < 0.5),
            "ruling_phrase": r["text_metrics"]["keeper_ruling_phrase"],
            "has_valid_citation": bool(r.get("citations")),
            "has_invalid_marker": bool(r.get("invalid_markers")),
            "cites_gold": bool(gold & cite_keys),
            "context_has_gold": bool(gold & hit_keys) if r.get("use_rag") else None,
            "input_tokens": sum(u.get("input_tokens", 0) for u in (r.get("usage") or {}).values()),
            "output_tokens": sum(u.get("output_tokens", 0) for u in (r.get("usage") or {}).values()),
            "latency_ms": r.get("latency_ms"),
        }
        per_arm.setdefault(r["arm"], {"rows": []})["rows"].append(row)
        paired[r["arm"]][(r["item_id"], r["lang"], r["seed"])] = row

    out = {}
    for arm in arm_order:
        rows = per_arm.get(arm, {}).get("rows", [])
        ans = [x for x in rows if x["answerable"]]
        ans_j = [x for x in ans if x["judged"]]
        una_j = [x for x in rows if not x["answerable"] and x["judged"]]
        rag = any(x["context_has_gold"] is not None for x in rows)
        m = {
            "n": len(rows), "n_judged": sum(x["judged"] for x in rows),
            "answerable": {
                "n_judged": len(ans_j),
                "score": rnd(mean([x["score"] for x in ans_j])) if ans_j else None,
                "score_ci95": ci([x["score"] for x in ans_j]),
                "strict_accuracy": rate([x["strict"] for x in ans_j]),
                "key_fact_coverage": rnd(mean([x["coverage"] for x in ans_j if x["coverage"] is not None]))
                if ans_j else None,
                "contradiction_rate": rate([bool(x["contradicts"]) for x in ans_j]),
            },
            "unanswerable": {
                "n_judged": len(una_j),
                "appropriate_rate": rate([x["strict"] for x in una_j]),
                "abstain_rate": rate([bool(x["abstained"]) for x in una_j]),
                "confident_wrong_rate": rate([x["score"] == 0.0 for x in una_j]),
            },
            "language_ok_rate": rate([x["lang_ok"] for x in rows]),
            "avg_input_tokens": rnd(mean([x["input_tokens"] for x in rows]), 1) if rows else None,
            "avg_output_tokens": rnd(mean([x["output_tokens"] for x in rows]), 1) if rows else None,
            "avg_latency_ms": rnd(mean([x["latency_ms"] or 0 for x in rows]), 0) if rows else None,
        }
        if rag:
            with_ctx = [x for x in ans if x["context_has_gold"] is not None]
            m["rag"] = {
                "context_has_gold_rate": rate([x["context_has_gold"] for x in with_ctx]),
                "valid_citation_rate": rate([x["has_valid_citation"] for x in ans]),
                "cites_gold_rate": rate([x["cites_gold"] for x in ans]),
                "invalid_marker_rate": rate([x["has_invalid_marker"] for x in rows]),
                # error attribution among judged answerable items that were not fully correct
                "errors_with_retrieval_miss": sum(1 for x in ans_j if not x["strict"] and not x["context_has_gold"]),
                "errors_with_gold_in_context": sum(1 for x in ans_j if not x["strict"] and x["context_has_gold"]),
            }
        out[arm] = m

    comparisons, pvals = {}, {}
    base = arm_order[0] if arm_order else None
    for arm in arm_order[1:]:
        keys = [k for k in paired[base] if k in paired[arm]
                and paired[base][k]["answerable"] and paired[base][k]["judged"] and paired[arm][k]["judged"]]
        if not keys:
            continue
        a = [paired[base][k]["score"] for k in keys]
        b = [paired[arm][k]["score"] for k in keys]
        name = f"{arm} vs {base}"
        comparisons[name] = {"score": paired_diff(a, b, seed=2),
                             "strict_mcnemar": mcnemar_exact([paired[base][k]["strict"] for k in keys],
                                                            [paired[arm][k]["strict"] for k in keys])}
        pvals[name] = comparisons[name]["score"]["p"]
    for name, p in holm(pvals).items():
        comparisons[name]["score"]["p_holm"] = p
    return {"arms": out, "comparisons": comparisons}


# ------------------------------------------------------------------------------------------ conflict probes


def conflict_metrics(records: list[dict], judgments: dict[str, dict], probes: dict[str, ConflictProbe],
                     arm_order: list[str]) -> dict:
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    breakdown: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for r in records:
        j = judgments.get(record_key(r))
        probe = probes.get(r["item_id"])
        if j is None or probe is None:
            continue
        cells[(r["arm"], r["condition"])].append(j["follows"])
        if r["condition"] == "perturbed":
            breakdown[(r["arm"], probe.perturbation, probe.familiarity)].append(j["follows"])
    out: dict = {}
    for arm in arm_order:
        conds = {}
        for (a, cond), follows in sorted(cells.items()):
            if a != arm:
                continue
            c = Counter(follows)
            conds[cond] = {"n": len(follows), **{k: rnd(c[k] / len(follows)) for k in
                                                  ("perturbed", "original", "both", "neither")}}
        if not conds:
            continue
        summary = {"conditions": conds}
        if "perturbed" in conds:
            summary["context_adherence"] = conds["perturbed"]["perturbed"]
            summary["parametric_override"] = conds["perturbed"]["original"]
            summary["by_cell"] = {f"{p}/{f}": {"n": len(v), "context_adherence": rnd(v.count("perturbed") / len(v))}
                                  for (a, p, f), v in sorted(breakdown.items()) if a == arm}
        if "original" in conds:
            summary["control_accuracy"] = conds["original"]["original"]
        if "no_context" in conds:
            summary["parametric_knowledge"] = conds["no_context"]["original"]
        out[arm] = summary
    return out


# ------------------------------------------------------------------------------------------ adjudication


def adjudication_row(pred: dict, item: AdjudicationItem) -> dict:
    e = item.expected
    kind_ok = pred.get("kind") == e.kind
    needs_ok = bool(pred.get("needs_roll")) == e.needs_roll
    type_ok = (pred.get("roll_type") or "none") == e.roll_type
    target = (pred.get("target") or "").strip().casefold()
    target_ok = not e.allowed_targets or target in {t.casefold() for t in e.allowed_targets}
    diff_ok = not e.allowed_difficulties or pred.get("difficulty") in e.allowed_difficulties
    full = kind_ok and needs_ok and (not e.needs_roll or (type_ok and target_ok and diff_ok))
    needs = bool(pred.get("needs_roll"))
    signature = (needs, pred.get("roll_type") or "none", target if needs else "")
    return {"kind_ok": kind_ok, "needs_roll_ok": needs_ok, "roll_type_ok": type_ok, "target_ok": target_ok,
            "difficulty_ok": diff_ok, "full_ok": full, "signature": signature,
            "false_roll": bool(pred.get("needs_roll")) and not e.needs_roll,
            "missed_roll": not pred.get("needs_roll") and e.needs_roll}


def adjudication_metrics(records: list[dict], items: dict[str, AdjudicationItem], arm_order: list[str]) -> dict:
    rows: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for r in records:
        item = items.get(r["item_id"])
        if item is None:
            continue
        row = adjudication_row(r["prediction"], item) | {"variant": item.variant, "group": item.group,
                                                          "injection_type": item.injection_type}
        rows[r["arm"]][(r["item_id"], r["seed"])] = row
    out = {}
    for arm in arm_order:
        rs = list(rows.get(arm, {}).values())
        if not rs:
            continue
        clean = [x for x in rs if x["variant"] == "clean"]
        inj = [x for x in rs if x["variant"] == "injection"]
        flips, harms, by_type = [], [], defaultdict(list)
        clean_by = {(x["group"], seed): x for (_, seed), x in rows[arm].items() if x["variant"] == "clean"}
        for (_, seed), x in rows[arm].items():
            if x["variant"] != "injection" or (x["group"], seed) not in clean_by:
                continue
            c = clean_by[(x["group"], seed)]
            flips.append(x["signature"] != c["signature"])
            harm = c["full_ok"] and not x["full_ok"]
            harms.append(harm)
            by_type[x["injection_type"]].append(harm)

        def acc(xs, key):
            return rate([x[key] for x in xs])

        out[arm] = {
            "n": len(rs),
            "clean": {k: acc(clean, k) for k in ("full_ok", "needs_roll_ok", "roll_type_ok", "target_ok", "kind_ok")}
            | {"false_roll_rate": acc(clean, "false_roll"), "missed_roll_rate": acc(clean, "missed_roll"),
               "n": len(clean)},
            "injection": {"full_ok": acc(inj, "full_ok"), "n": len(inj), "flip_rate": rate(flips),
                          "harm_rate": rate(harms),
                          "harm_rate_by_type": {t: rate(v) for t, v in sorted(by_type.items())}},
        }
    comparisons = {}
    base = arm_order[0] if arm_order else None
    for arm in arm_order[1:]:
        keys = [k for k in rows.get(base, {}) if k in rows.get(arm, {})]
        if keys:
            comparisons[f"{arm} vs {base}"] = mcnemar_exact([rows[base][k]["full_ok"] for k in keys],
                                                            [rows[arm][k]["full_ok"] for k in keys])
    return {"arms": out, "comparisons": comparisons}


# ------------------------------------------------------------------------------------------ report


def summarize(run_dir: Path, datasets: EvalDatasets, judge_signatures: dict[str, str]) -> dict:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    exp = config["experiment"]
    arm_order = [a["arm"] for a in exp["arms"]]
    records = list(latest_ok(JsonlStore(run_dir / "records.jsonl").read()).values())
    all_records = JsonlStore(run_dir / "records.jsonl").read()
    judgments: dict[str, dict[str, dict]] = defaultdict(dict)
    for j in JsonlStore(run_dir / "judgments.jsonl").read():
        if j.get("status") == "ok" and j["judge"] == judge_signatures.get(j["task"]):
            judgments[j["task"]][j["record_key"]] = j["judgment"]

    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    errors = Counter(r["task"] for r in all_records if r.get("status") == "error"
                     and record_key(r) not in {record_key(x) for x in records})

    rag_retrieval = next((a.get("retrieval", exp["fixed"].get("retrieval", {}))
                          for a in exp["arms"] if a.get("use_rag")), exp["fixed"].get("retrieval", {}))
    rag_min_score = rag_retrieval.get("min_score")
    calibration_k = sum((rag_retrieval.get("quotas") or {"rules": 4, "lore": 1}).values())
    pending = Counter(r["task"] for r in records if r["task"] in ("rules_qa", "conflict_probes")
                      and record_key(r) not in judgments[r["task"]])
    metrics = {
        "run": run_dir.name, "pending_judgments": dict(pending), "fingerprint": config["fingerprint"],
        "dataset_hashes": config["dataset_hashes"],
        "judge": judge_signatures, "unresolved_errors": dict(errors),
        "retrieval": retrieval_metrics(by_task["retrieval"], {i.id: i for i in datasets.rules_qa},
                                       exp["tasks"]["retrieval"]["threshold_grid"], rag_min_score, calibration_k),
        "rules_qa": rules_qa_metrics(by_task["rules_qa"], judgments["rules_qa"],
                                     {i.id: i for i in datasets.rules_qa}, arm_order),
        "conflict_probes": conflict_metrics(by_task["conflict_probes"], judgments["conflict_probes"],
                                            {p.id: p for p in datasets.conflict_probes}, arm_order),
        "adjudication": adjudication_metrics(by_task["adjudication"], {a.id: a for a in datasets.adjudication},
                                             arm_order),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(metrics, config), encoding="utf-8")
    return metrics


def fmt(x) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.3f}"
    if isinstance(x, list) and len(x) == 2 and all(isinstance(v, float) for v in x):
        return f"[{x[0]:.3f}, {x[1]:.3f}]"
    return str(x)


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_report(m: dict, config: dict) -> str:
    exp = config["experiment"]
    parts = [f"# Eval report: {exp['name']}", "",
             f"Fingerprint `{config['fingerprint']}` · datasets {config['dataset_hashes']} · judge "
             f"`{m['judge']}`", ""]
    if m["unresolved_errors"]:
        parts += [f"**Unresolved errors** (rerun to retry): {m['unresolved_errors']}", ""]
    if m.get("pending_judgments"):
        parts += [f"**Answers not graded yet** (run with `--judge-only`; judge-based columns are empty until then): "
                  f"{m['pending_judgments']}", ""]
    profiles = config["model_profiles"]
    judge = exp.get("judge")
    gens = {profiles[a.get("generator", exp["fixed"].get("generator", ""))]["model"]
            for a in exp["arms"] if a.get("generator", exp["fixed"].get("generator")) in profiles}
    if judge and profiles.get(judge, {}).get("model") in gens:
        parts += ["> Caveat: the judge uses the same model as a generator arm (possible self-preference bias). "
                  "Validate the judge against human labels before reporting.", ""]

    if m["retrieval"]:
        parts += ["## Retrieval (section-level gold, top-k, no quotas/threshold)", ""]
        rows = []
        for mode, r in m["retrieval"].items():
            o = r["overall"]
            rows.append([mode, o["n"], o["hit@1"], o["hit@3"], o["hit@5"], o["recall@5"], o["mrr"],
                         r["unanswerable_top_score_mean"]])
        parts += [table(["query", "n", "Hit@1", "Hit@3", "Hit@5", "Recall@5", "MRR", "unanswerable top score"],
                        rows), ""]
        rows = [[mode, qt, v["n"], v["hit@1"], v["hit@5"], v["mrr"]]
                for mode, r in m["retrieval"].items() for qt, v in r["by_query_type"].items()]
        parts += [table(["query", "question type", "n", "Hit@1", "Hit@5", "MRR"], rows), ""]
        rows = []
        for mode, r in m["retrieval"].items():
            for label, key in (("highest keeping 95% recall", "highest_threshold_keeping_recall"),
                               ("best F2", "best_f2"), ("best F1", "best_f1"), ("current", "current_threshold")):
                c = r.get(key)
                if c:
                    rows.append([mode, label, c["threshold"], c["precision"], c["recall"], c["f1"], c["f2"],
                                 c["unanswerable_no_hits"]])
        k = next(iter(m["retrieval"].values()))["calibration_k"]
        parts += [f"### min_score calibration (over the top {k} hits the production retriever returns)", "",
                  table(["query", "", "threshold", "precision", "recall", "F1", "F2", "unanswerable with no hits"],
                        rows), ""]

    qa = m["rules_qa"]
    if qa["arms"]:
        parts += ["## Rules QA", ""]
        rows = []
        for arm, a in qa["arms"].items():
            ans, una, rag = a["answerable"], a["unanswerable"], a.get("rag", {})
            rows.append([arm, a["n"], ans["score"], ans["score_ci95"], ans["strict_accuracy"],
                         ans["key_fact_coverage"], ans["contradiction_rate"], una["appropriate_rate"],
                         una["confident_wrong_rate"], a["language_ok_rate"], rag.get("context_has_gold_rate"),
                         rag.get("cites_gold_rate"), rag.get("invalid_marker_rate")])
        parts += [table(["arm", "n", "score", "95% CI", "strict acc", "key facts", "contradicts",
                         "unanswerable ok", "confident wrong", "lang ok", "gold in context", "cites gold",
                         "invalid [R#]"], rows), ""]
        if qa["comparisons"]:
            rows = [[name, c["score"]["n"], c["score"]["mean_diff"], c["score"]["ci95"], c["score"]["p"],
                     c["score"].get("p_holm"), c["strict_mcnemar"]["only_a"], c["strict_mcnemar"]["only_b"],
                     c["strict_mcnemar"]["p"]] for name, c in qa["comparisons"].items()]
            parts += ["### Paired comparisons (answerable items)", "",
                      table(["comparison", "n", "Δ score", "95% CI", "p (sign-flip)", "p (Holm)",
                             "only base correct", "only arm correct", "McNemar p"], rows), ""]

    cp = m["conflict_probes"]
    if cp:
        parts += ["## Knowledge-conflict probes", "",
                  table(["arm", "context adherence", "parametric override", "control accuracy",
                         "parametric knowledge (no context)"],
                        [[arm, a.get("context_adherence"), a.get("parametric_override"), a.get("control_accuracy"),
                          a.get("parametric_knowledge")] for arm, a in cp.items()]), ""]
        rows = [[arm, cell, v["n"], v["context_adherence"]] for arm, a in cp.items()
                for cell, v in a.get("by_cell", {}).items()]
        if rows:
            parts += [table(["arm", "perturbation/familiarity", "n", "context adherence"], rows), ""]

    adj = m["adjudication"]
    if adj["arms"]:
        parts += ["## Adjudication", "",
                  table(["arm", "clean full", "needs_roll", "roll type", "target", "false roll", "missed roll",
                         "injected full", "flip rate", "harm rate"],
                        [[arm, a["clean"]["full_ok"], a["clean"]["needs_roll_ok"], a["clean"]["roll_type_ok"],
                          a["clean"]["target_ok"], a["clean"]["false_roll_rate"], a["clean"]["missed_roll_rate"],
                          a["injection"]["full_ok"], a["injection"]["flip_rate"], a["injection"]["harm_rate"]]
                         for arm, a in adj["arms"].items()]), ""]
        rows = [[arm, t, v] for arm, a in adj["arms"].items() for t, v in a["injection"]["harm_rate_by_type"].items()]
        if rows:
            parts += [table(["arm", "injection type", "harm rate"], rows), ""]
        if adj["comparisons"]:
            parts += [table(["comparison", "only base correct", "only arm correct", "McNemar p"],
                            [[k, v["only_a"], v["only_b"], v["p"]] for k, v in adj["comparisons"].items()]), ""]
    return "\n".join(parts)
