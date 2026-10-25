"""Aggregate eval records into metrics.json and a Markdown report (no LLM calls; safe to rerun any time)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from keeper.citations import cited_markers
from sft.split import SPLIT_PATH, SectionSplit, load_split

from .datasets import AdjudicationItem, ConflictProbe, EvalDatasets, RulesQAItem
from .harness import JsonlStore, record_key
from .stats import bootstrap_ci, holm, mcnemar_exact, mean, paired_diff, unpaired_diff


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


def qa_row(r: dict, item: RulesQAItem, j: dict | None) -> dict:
    """One generated answer with its judgment (None until graded), as every rules-QA metric reads it."""
    gold = {g.key() for g in item.gold_sections}
    hit_keys = {(h["source"], h["section"]) for h in r.get("hits") or []}
    cite_keys = {(c.get("source"), c.get("section")) for c in r.get("citations") or []}
    return {
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


def rules_qa_metrics(records: list[dict], judgments: dict[str, dict], items: dict[str, RulesQAItem],
                     arm_order: list[str]) -> dict:
    per_arm: dict[str, dict] = {}
    paired: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for r in records:
        item = items.get(r["item_id"])
        if item is None:
            continue
        row = qa_row(r, item, judgments.get(record_key(r)))
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


# ------------------------------------------------------------------------------------------ 2x2 (Phase 4)
#
# Only for runs with exactly one arm per cell of "base vs fine-tuned x prompt-only vs RAG". The factors are read from
# the run's config.json (`finetuned` of the generator profile, `use_rag` of the arm), never from arm names; any other
# run gets no `factorial` key in metrics.json and nothing extra in report.md.

CELLS = {"base_prompt": (False, False), "base_rag": (False, True),   # cell -> (finetuned, use_rag), the report's order
         "ft_prompt": (True, False), "ft_rag": (True, True)}
# Signed cell weights per contrast. Applied to one item's four values they give that item's contrast, so every contrast
# is a paired difference over items (bootstrap CI and sign-flip p, like the arm comparisons above).
CONTRASTS = {
    "ft_main": {"ft_prompt": 0.5, "ft_rag": 0.5, "base_prompt": -0.5, "base_rag": -0.5},
    "rag_main": {"base_rag": 0.5, "ft_rag": 0.5, "base_prompt": -0.5, "ft_prompt": -0.5},
    # (ft_rag - base_rag) - (ft_prompt - base_prompt): positive when the adapter helps more with RAG than without
    "interaction": {"ft_rag": 1.0, "base_rag": -1.0, "ft_prompt": -1.0, "base_prompt": 1.0},
    "ft_given_prompt": {"ft_prompt": 1.0, "base_prompt": -1.0},
    "ft_given_rag": {"ft_rag": 1.0, "base_rag": -1.0},
    "rag_given_base": {"base_rag": 1.0, "base_prompt": -1.0},
    "rag_given_ft": {"ft_rag": 1.0, "ft_prompt": -1.0},
}
PRIMARY = ("ft_main", "rag_main", "interaction")   # the Holm family of one outcome; the simple effects are descriptive
STRATUM_CONTRASTS = ("ft_given_prompt", "ft_main")   # compared between the train and the held-out stratum
FACTORIAL_SEED = 2   # the seed of the arm-vs-baseline comparisons above: the same comparison gets the same interval
STRATA = ("train", "heldout", "mixed", "unanswerable", "unsplit")
PROFILE_NEUTRAL = {"model", "finetuned", "base", "quant", "model_card"}   # profile fields that name the model itself

QA_MEASURES: dict[str, Callable[[dict], float | None]] = {   # per judged answer
    "score": lambda x: x["score"],
    "strict_accuracy": lambda x: float(x["strict"]),
    "key_fact_coverage": lambda x: x["coverage"],
    "confident_wrong": lambda x: float(x["score"] == 0.0),
}
# outcome -> (condition, the fact a counted answer asserts, base cell, fine-tuned cell). Prompt-only arms have no
# perturbed/original conditions and RAG arms have no no_context one, so the interaction is undefined here: only the
# fine-tuning effect within one RAG level can be paired.
CONFLICT_MEASURES = {
    "context_adherence": ("perturbed", "perturbed", "base_rag", "ft_rag"),
    "control_accuracy": ("original", "original", "base_rag", "ft_rag"),
    "parametric_knowledge": ("no_context", "original", "base_prompt", "ft_prompt"),
}


class Design(NamedTuple):
    cells: dict[str, str]        # cell -> arm, in CELLS order
    extra_arms: list[str]        # arms outside the 2x2: left out of every contrast
    cautions: list[str]


def arm_factors(config: dict) -> dict[str, dict]:
    """arm -> generator, use_rag, adjudicator, and whether the generator profile is fine-tuned (old configs: no)."""
    exp, profiles = config["experiment"], config.get("model_profiles") or {}
    out = {}
    for arm in exp["arms"]:
        merged = {**exp.get("fixed", {}), **arm}
        generator = merged.get("generator")
        out[arm["arm"]] = {"generator": generator, "use_rag": bool(merged.get("use_rag", False)),
                           "adjudicator": merged.get("adjudicator", "aux"),
                           "finetuned": bool((profiles.get(generator) or {}).get("finetuned", False))}
    return out


def detect_2x2(factors: dict[str, dict], profiles: dict[str, dict] | None = None) -> Design | None:
    """The four arms of the 2x2 (one per finetuned x use_rag cell), or None when they cannot be told apart.

    Each fine-tuned cell needs exactly one arm. When a base cell has several, the one whose generator is the `base`
    declared by the fine-tuned profile is taken; arms outside the four are listed in `extra_arms`. A base arm whose
    generator is not the declared base gets a caution, and so does a fine-tuned profile that declares no base.
    """
    by_cell: dict[tuple[bool, bool], list[str]] = defaultdict(list)
    for arm, f in factors.items():
        by_cell[(f["finetuned"], f["use_rag"])].append(arm)
    chosen: dict[str, str] = {}
    for cell, key in CELLS.items():
        if key[0]:
            if len(by_cell.get(key, [])) != 1:
                return None
            chosen[cell] = by_cell[key][0]
    declared = {(profiles or {}).get(factors[arm]["generator"], {}).get("base") for arm in chosen.values()} - {None}
    declared_base = next(iter(declared)) if len(declared) == 1 else None
    cautions = []
    if not declared:
        cautions.append("the fine-tuned profile declares no `base`, so the base arms are not checked against it")
    elif len(declared) > 1:
        cautions.append(f"the fine-tuned arms declare different bases ({', '.join(sorted(declared))}), "
                        "so the base arms are not checked against them")
    for cell, key in CELLS.items():
        if not key[0]:
            arms = by_cell.get(key, [])
            if len(arms) > 1 and declared_base:
                arms = [a for a in arms if factors[a]["generator"] == declared_base]
            if len(arms) != 1:
                return None
            chosen[cell] = arms[0]
            if declared_base and factors[arms[0]]["generator"] != declared_base:
                cautions.append(f"base arm `{arms[0]}` uses profile `{factors[arms[0]]['generator']}`, not "
                                f"`{declared_base}`, the base declared for the fine-tuned profile")
    ordered = {cell: chosen[cell] for cell in CELLS}
    return Design(ordered, [a for a in factors if a not in ordered.values()], cautions)


def design_confounds(config: dict, factors: dict[str, dict], arms: list[str] | None = None) -> list[str]:
    """What else varies between the arms of the 2x2: a contrast is only about the two factors if nothing else does."""
    exp = config["experiment"]
    chosen = set(arms if arms is not None else factors)
    merged = [{**exp.get("fixed", {}), **a} for a in exp["arms"] if a["arm"] in chosen]
    keys = {k for m in merged for k in m} - {"arm", "generator", "use_rag"}
    out = []
    differing = sorted(k for k in keys if len({json.dumps(m.get(k), sort_keys=True) for m in merged}) > 1)
    if differing:
        out.append(f"the arms also differ in {', '.join(differing)}")
    for finetuned, label in ((False, "base"), (True, "fine-tuned")):
        if len({f["generator"] for a, f in factors.items() if a in chosen and f["finetuned"] is finetuned}) > 1:
            out.append(f"the {label} arms use different generator profiles")
    profiles = config.get("model_profiles") or {}
    snaps = {g: {k: v for k, v in profiles[g].items() if k not in PROFILE_NEUTRAL}
             for g in sorted({factors[a]["generator"] for a in chosen}) if g in profiles}
    fields = sorted({k for s in snaps.values() for k in s})
    unequal = [k for k in fields if len({json.dumps(s.get(k), sort_keys=True) for s in snaps.values()}) > 1]
    if len(snaps) > 1 and unequal:
        out.append(f"the generator profiles differ in {', '.join(unequal)} besides the model itself")
    return out


def load_section_split(config: dict, datasets: EvalDatasets,
                       path: Path | None) -> tuple[SectionSplit | None, str | None]:
    """The frozen section split the models were trained with as (split, None), or (None, why it cannot be used).

    The model cards in config.json record the sha256 of the split file; a split file that differs from it (or was built
    from another rules QA gold file) would put items into the wrong strata, so it is not used.
    """
    shas = {c.get("split_sha256") for c in (config.get("model_cards") or {}).values()} - {None}
    if not shas:
        return None, "no model card in config.json records the section split"
    if len(shas) > 1:
        return None, "the model cards record different section splits"
    path = path or SPLIT_PATH
    if not path.is_file():
        return None, f"split file {path.name} not found"
    raw = path.read_bytes()
    if shas.pop() not in {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}:
        return None, f"{path.name} is not the split recorded in the model cards (sha256 differs)"
    try:
        split = load_split(path)
    except ValueError as e:
        return None, f"cannot read {path.name}: {e}"
    if split.gold_hashes.get("rules_qa", datasets.hashes["rules_qa"]) != datasets.hashes["rules_qa"]:
        return None, f"the rules QA gold file changed since {path.name} was built"
    return split, None


def item_stratum(item: RulesQAItem, train: set[tuple[str, str]], heldout: set[tuple[str, str]]) -> str:
    """unanswerable | train (every gold section in the train half) | heldout (all held out) | mixed | unsplit."""
    if not item.answerable:
        return "unanswerable"
    sides = {"train" if g.key() in train else "heldout" if g.key() in heldout else "unsplit"
             for g in item.gold_sections}
    if "unsplit" in sides:
        return "unsplit"
    return "mixed" if len(sides) > 1 else next(iter(sides))


def scored(rows: dict[tuple, dict], value: Callable[[dict], float | None]) -> dict[tuple, float]:
    """key -> value for the judged rows that have one. A key is (item, language, seed) or (item, seed)."""
    return {k: v for k, x in rows.items() if x["judged"] and (v := value(x)) is not None}


def paired_means(per_cell: dict[str, dict[tuple, float]],
                 unit: Callable[[tuple], str] = lambda key: key[0]) -> dict[str, dict[str, float]]:
    """cell -> unit -> mean over the keys scored in EVERY cell, so no cell is averaged over other answers than the rest.

    An answer that is missing or not graded in one cell is dropped from all of them. The unit is the item, or for
    adjudication the group (a clean turn with its injection variants), and is what a bootstrap resamples. Units come in
    the record order of the first cell.
    """
    first, *rest = per_cell.values()
    units: dict[str, list[tuple]] = defaultdict(list)
    for key in first:
        if all(key in d for d in rest):
            units[unit(key)].append(key)
    return {c: {u: mean([d[k] for k in keys]) for u, keys in units.items()} for c, d in per_cell.items()}


def unit_ids(per_cell: dict[str, dict[str, float]], keep: set[str] | None = None) -> list[str]:
    """The units of a paired sample (those in `keep`, if given), in the order of paired_means."""
    return [u for u in next(iter(per_cell.values())) if keep is None or u in keep]


def contrast_values(per_cell: dict[str, dict[str, float]], ids: list[str], weights: dict[str, float]) -> list[float]:
    """One contrast per unit. A value that is zero up to float residue (~1e-17, from 0.5 weights and thirds) becomes an
    exact zero, so the sign-flip test does not count it as a difference; other values are left as they are."""
    values = [sum(w * per_cell[c][i] for c, w in weights.items()) for i in ids]
    return [0.0 if abs(v) < 1e-9 else v for v in values]


def contrast_set(per_cell: dict[str, dict[str, float]], ids: list[str]) -> dict[str, dict]:
    """Every contrast over `ids`: n, mean_diff, ci95 and sign-flip p (raw, see paired_diff); `ids` must not be empty."""
    zeros = [0.0] * len(ids)
    return {name: paired_diff(zeros, contrast_values(per_cell, ids, weights), seed=FACTORIAL_SEED)
            for name, weights in CONTRASTS.items()}


def add_holm(results: dict[str, dict]) -> None:
    """Set `p_holm` on each result (in place) from the raw `p` of the whole family."""
    for key, p in holm({k: r["p"] for k, r in results.items()}).items():
        results[key]["p_holm"] = p


def tidy(r: dict) -> dict:
    out = {"n": r["n"], "mean_diff": rnd(r["mean_diff"]), "ci95": [rnd(r["ci95"][0]), rnd(r["ci95"][1])],
           "p": rnd(r["p"])}
    if "p_holm" in r:
        out["p_holm"] = rnd(r["p_holm"])
    return out


def measure_block(per_cell: dict[str, dict[str, float]], ids: list[str]) -> dict | None:
    """Cell means and all contrasts of one outcome; the Holm family is the three primary contrasts."""
    if not ids:
        return None
    raw = contrast_set(per_cell, ids)
    add_holm({k: raw[k] for k in PRIMARY})
    return {"n": len(ids), "means": {c: rnd(mean([per_cell[c][i] for i in ids])) for c in CELLS},
            "effects": {k: tidy(v) for k, v in raw.items()}}


def marker_rates(records: list[dict], items: dict[str, RulesQAItem], cells: dict[str, str],
                 factors: dict[str, dict]) -> dict[str, dict]:
    """Per arm, over all generated rules-QA answers (graded or not): invalid [R#] markers, "Keeper ruling" phrases.

    A RAG arm's invalid markers are the ones the run recorded (not among the retrieved passages). A prompt-only arm
    retrieves nothing, so every [R#] in its answer is invalid; the run does not validate those, so they are counted
    from the answer text.
    """
    out = {}
    for cell, arm in cells.items():
        rs = [(r, items[r["item_id"]]) for r in records if r["arm"] == arm and r["item_id"] in items]
        rag = factors[arm]["use_rag"]
        invalid = [bool(r.get("invalid_markers")) if rag else bool(cited_markers(r.get("answer", ""))) for r, _ in rs]
        ruling = {True: [], False: []}
        for r, item in rs:
            ruling[item.answerable].append(bool(r["text_metrics"]["keeper_ruling_phrase"]))
        out[arm] = {"cell": cell, "n": len(rs), "invalid_marker_rate": rate(invalid),
                    "n_answerable": len(ruling[True]), "keeper_ruling_answerable": rate(ruling[True]),
                    "n_unanswerable": len(ruling[False]), "keeper_ruling_unanswerable": rate(ruling[False])}
    return out


def factorial_rules_qa(records: list[dict], judgments: dict[str, dict], items: dict[str, RulesQAItem],
                       cells: dict[str, str], factors: dict[str, dict], split: SectionSplit | None,
                       split_note: str | None) -> dict:
    arm_cell = {arm: cell for cell, arm in cells.items()}
    rows: dict[str, dict[tuple, dict]] = {c: {} for c in CELLS}
    for r in records:
        if r["arm"] in arm_cell and r["item_id"] in items:
            rows[arm_cell[r["arm"]]][(r["item_id"], r["lang"], r["seed"])] = qa_row(
                r, items[r["item_id"]], judgments.get(record_key(r)))
    keys = {m: {c: scored(rows[c], value) for c in CELLS} for m, value in QA_MEASURES.items()}
    values = {m: paired_means(keys[m]) for m in QA_MEASURES}
    common = set.intersection(*(set(d) for d in keys["score"].values()))
    seen = set().union(*(set(rows[c]) for c in CELLS))
    answerable = {i for i, item in items.items() if item.answerable}

    strata = contrast = None
    if split is not None:
        members: dict[str, set[str]] = defaultdict(set)
        for i, item in items.items():
            members[item_stratum(item, split.train_keys(), split.heldout_keys())].add(i)
        raw = {name: (ids, contrast_set(values["score"], ids)) for name in STRATA
               if (ids := unit_ids(values["score"], members[name]))}
        for primary in PRIMARY:   # Holm across the strata, per contrast
            add_holm({name: effects[primary] for name, (_, effects) in raw.items()})
        strata = {}
        for name, (ids, effects) in raw.items():
            means = {m: {c: rnd(mean([v[i] for i in ids if i in v])) for c, v in values[m].items()}
                     for m in ("score", "strict_accuracy", "key_fact_coverage", "confident_wrong")}
            strata[name] = {"n": len(ids), "means": means, "effects": {k: tidy(v) for k, v in effects.items()}}
        if "train" in raw and "heldout" in raw:
            # the per-item FT effect of the train stratum against the held-out one: different items, so independent
            diffs = {name: unpaired_diff(contrast_values(values["score"], raw["heldout"][0], CONTRASTS[name]),
                                         contrast_values(values["score"], raw["train"][0], CONTRASTS[name]),
                                         seed=FACTORIAL_SEED) for name in STRATUM_CONTRASTS}
            add_holm(diffs)
            contrast = {name: {"n_train": d["n_b"], "n_heldout": d["n_a"], "mean_diff": rnd(d["mean_diff"]),
                               "ci95": [rnd(d["ci95"][0]), rnd(d["ci95"][1])], "p": rnd(d["p"]),
                               "p_holm": rnd(d["p_holm"])} for name, d in diffs.items()}
    return {
        "items": {"graded_in_all_arms": len(unit_ids(values["score"])), "answers_graded_in_all_arms": len(common),
                  "dropped_answers": len(seen) - len(common)},
        "answerable": {m: measure_block(values[m], unit_ids(values[m], answerable))
                       for m in ("score", "key_fact_coverage")},
        "strata": strata, "strata_note": split_note, "train_vs_heldout": contrast,
        "markers": marker_rates(records, items, cells, factors),
    }


def factorial_conflict(records: list[dict], judgments: dict[str, dict], probes: dict[str, ConflictProbe],
                       cells: dict[str, str]) -> dict | None:
    arm_cell = {arm: cell for cell, arm in cells.items()}
    answers: dict[str, dict[str, dict[tuple, float | None]]] = {
        m: {c: {} for c in spec[2:]} for m, spec in CONFLICT_MEASURES.items()}   # None: recorded, not graded
    for r in records:
        cell = arm_cell.get(r["arm"])
        if cell is None or r["item_id"] not in probes:
            continue
        j = judgments.get(record_key(r))
        for name, (condition, fact, base, ft) in CONFLICT_MEASURES.items():
            if r["condition"] == condition and cell in (base, ft):
                answers[name][cell][(r["item_id"], r["lang"], r["seed"])] = None if j is None else \
                    float(j["follows"] == fact)
    raw = {}
    for name, (_, _, base, ft) in CONFLICT_MEASURES.items():
        graded = {c: {k: v for k, v in answers[name][c].items() if v is not None} for c in (base, ft)}
        per_cell = paired_means(graded)
        if ids := unit_ids(per_cell):
            effect = paired_diff([0.0] * len(ids), contrast_values(per_cell, ids, {ft: 1.0, base: -1.0}),
                                 seed=FACTORIAL_SEED)
            dropped = len(set(answers[name][base]) | set(answers[name][ft])) - len(set(graded[base]) & set(graded[ft]))
            raw[name] = (per_cell, ids, effect, dropped)
    if not raw:
        return None
    add_holm({name: x[2] for name, x in raw.items()})   # one family: the outcomes of the task
    return {name: {"n": len(ids), "dropped_answers": dropped, "base_cell": CONFLICT_MEASURES[name][2],
                   "ft_cell": CONFLICT_MEASURES[name][3],
                   "means": {c: rnd(mean([v[i] for i in ids])) for c, v in per_cell.items()},
                   "ft_effect": tidy(effect)} for name, (per_cell, ids, effect, dropped) in raw.items()}


def factorial_adjudication(records: list[dict], items: dict[str, AdjudicationItem], cells: dict[str, str],
                           factors: dict[str, dict]) -> tuple[dict | None, str | None]:
    """(contrasts on adjudication full_ok, note). Only when every arm adjudicates with its own generator: with the
    auxiliary model adjudicating, the fine-tuned model has no part in it. The unit is the group of a clean turn and its
    injection variants, resampled together."""
    arm_cell = {arm: cell for cell, arm in cells.items()}
    mine = [r for r in records if r["arm"] in arm_cell and r["item_id"] in items]
    if not mine:
        return None, None
    other = [f"`{a}` ({factors[a]['adjudicator']})" for a in cells.values() if factors[a]["adjudicator"] != "generator"]
    if other:
        return None, ("adjudication is not analysed: the adjudicator is not the generator in " + ", ".join(other)
                      + ", so the fine-tuned model does not adjudicate and the contrasts would not be about it")
    rows: dict[str, dict[tuple, float]] = {c: {} for c in CELLS}
    for r in mine:
        rows[arm_cell[r["arm"]]][(r["item_id"], r["seed"])] = float(
            adjudication_row(r["prediction"], items[r["item_id"]])["full_ok"])
    per_cell = paired_means(rows, unit=lambda key: items[key[0]].group)
    block = measure_block(per_cell, unit_ids(per_cell))
    if block:
        common = set.intersection(*(set(d) for d in rows.values()))
        block["dropped_answers"] = len(set().union(*(set(d) for d in rows.values()))) - len(common)
    return block, None


def factorial_metrics(config: dict, by_task: dict[str, list[dict]], judgments: dict[str, dict[str, dict]],
                      datasets: EvalDatasets, split_path: Path | None = None) -> dict | None:
    """The 2x2 analysis, or None when the run's arms do not contain one arm per (finetuned, use_rag) cell."""
    factors = arm_factors(config)
    design = detect_2x2(factors, config.get("model_profiles"))
    if design is None:
        return None
    cells = design.cells
    split, split_note = load_section_split(config, datasets, split_path)
    adjudication, adjudication_note = factorial_adjudication(
        by_task.get("adjudication", []), {a.id: a for a in datasets.adjudication}, cells, factors)
    return {
        "cells": cells, "extra_arms": design.extra_arms, "factors": factors,
        "confounds": [*design.cautions, *design_confounds(config, factors, list(cells.values()))],
        "rules_qa": factorial_rules_qa(by_task.get("rules_qa", []), judgments.get("rules_qa", {}),
                                       {i.id: i for i in datasets.rules_qa}, cells, factors, split, split_note),
        "conflict_probes": factorial_conflict(by_task.get("conflict_probes", []), judgments.get("conflict_probes", {}),
                                              {p.id: p for p in datasets.conflict_probes}, cells),
        "adjudication": adjudication, "adjudication_note": adjudication_note,
    }


# ------------------------------------------------------------------------------------------ report


def summarize(run_dir: Path, datasets: EvalDatasets, judge_signatures: dict[str, str],
              split_path: Path | None = None) -> dict:
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
    factorial = factorial_metrics(config, by_task, judgments, datasets, split_path)
    if factorial is not None:   # only when the arms form the 2x2: other runs keep their metrics.json and report.md
        metrics["factorial"] = factorial
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
    if m.get("factorial"):
        parts += render_factorial(m["factorial"])
    return "\n".join(parts)


CELL_LABELS = {"base_prompt": "base, prompt-only", "base_rag": "base, RAG", "ft_prompt": "fine-tuned, prompt-only",
               "ft_rag": "fine-tuned, RAG"}
EFFECT_LABELS = {"ft_main": "FT main effect", "rag_main": "RAG main effect", "interaction": "interaction FT x RAG",
                 "ft_given_prompt": "FT effect, prompt-only", "ft_given_rag": "FT effect, with RAG",
                 "rag_given_base": "RAG effect, base", "rag_given_ft": "RAG effect, fine-tuned"}
STRATUM_LABELS = {"train": "train (all gold sections seen)", "heldout": "held-out (no gold section seen)",
                  "mixed": "mixed (seen and held-out)", "unanswerable": "unanswerable (no gold section)",
                  "unsplit": "unsplit (a gold section is not in the split)"}
CONFLICT_LABELS = {"context_adherence": "context adherence", "control_accuracy": "control accuracy",
                   "parametric_knowledge": "parametric knowledge"}
QA_OUTCOMES = (("score", "score"), ("key_fact_coverage", "key facts"))


def effect_row(task: str, label: str, name: str, block: dict) -> list:
    e = block["effects"][name]
    return [task, label, EFFECT_LABELS[name], block["n"], e["mean_diff"], e["ci95"], e["p"], e.get("p_holm")]


def render_factorial(f: dict) -> list[str]:
    """The 2x2 sections of report.md (see the module comment above CELLS)."""
    arms = [f["cells"][c] for c in CELLS]
    qa = f["rules_qa"]
    parts = ["## 2x2: base vs fine-tuned x prompt-only vs RAG", "",
             "Cells (read from `finetuned` of the generator profile and `use_rag` of the arm): "
             + ", ".join(f"{CELL_LABELS[c]} = `{a}`" for c, a in f["cells"].items()) + ".", "",
             "Contrasts are paired per item (adjudication: per group, a clean turn with its injection variants "
             "resampled together). An item's value is the mean over its answers (language, seed) graded in all four "
             "arms; an answer that is missing or not graded in one arm is dropped from all four. The interval is a 95% "
             "bootstrap CI of the mean, p is a sign-flip test. FT effect = fine-tuned - base, RAG effect = RAG - "
             "prompt-only, interaction = (ft_rag - base_rag) - (ft_prompt - base_prompt), positive when the adapter "
             "helps more with RAG than without. p (Holm) adjusts the three contrasts of one outcome; the simple "
             "effects are descriptive. The fine-tuned model is one training run, so the FT contrast is not a "
             "distribution over training runs.", ""]
    if f["extra_arms"]:
        parts += [f"Arms outside the 2x2, left out of every contrast: {', '.join(f'`{a}`' for a in f['extra_arms'])}.",
                  ""]
    if f["confounds"]:
        parts += [f"> Caution: {'; '.join(f['confounds'])}, so the contrasts are not only about FT and RAG.", ""]
    if f.get("adjudication_note"):
        parts += [f"{f['adjudication_note'][0].upper()}{f['adjudication_note'][1:]}.", ""]

    dropped = qa["items"]["dropped_answers"]
    blocks = [("rules QA", label, qa["answerable"].get(key), dropped) for key, label in QA_OUTCOMES]
    adjudication = f.get("adjudication")
    blocks.append(("adjudication (n = groups)", "full ok", adjudication, (adjudication or {}).get("dropped_answers")))
    blocks = [(task, label, b, lost) for task, label, b, lost in blocks if b]
    conflict = f.get("conflict_probes") or {}
    if not (qa["answerable"]["score"] or conflict):
        parts += ["No rules QA or conflict-probe answer is graded in all four arms yet (run with `--judge-only`); "
                  "judge-based columns and contrasts appear after that.", ""]
    if blocks or conflict:
        rows = [[task, label, b["n"], lost, *[b["means"][c] for c in CELLS]] for task, label, b, lost in blocks]
        rows += [["conflict probes", CONFLICT_LABELS[k], v["n"], v["dropped_answers"],
                  *[v["means"].get(c) for c in CELLS]] for k, v in conflict.items()]
        effects = [effect_row(task, label, k, b) for task, label, b, _ in blocks for k in PRIMARY]
        effects += [["conflict probes", CONFLICT_LABELS[k], "FT effect, " + ("with RAG" if v["ft_cell"] == "ft_rag"
                                                                            else "prompt-only"),
                     v["n"], *[v["ft_effect"][x] for x in ("mean_diff", "ci95", "p", "p_holm")]]
                    for k, v in conflict.items()]
        items = qa["items"]
        parts += [f"Rules QA: {items['answers_graded_in_all_arms']} answers of {items['graded_in_all_arms']} items "
                  "graded in all four arms; dropped (recorded in some arm but missing or not graded in at least "
                  f"one): {dropped}.", "",
                  "### Cell means", "",
                  table(["task", "outcome", "n", "dropped answers", *arms], rows), "",
                  "### Effects", "",
                  table(["task", "outcome", "contrast", "n", "Δ", "95% CI", "p (sign-flip)", "p (Holm)"], effects), ""]
        if conflict:
            parts += ["Conflict probes have no interaction: the prompt-only arms answer without context, so context "
                      "adherence and control accuracy exist only in the RAG arms and parametric knowledge only in the "
                      "prompt-only arms. Their Holm family is the conflict outcomes.", ""]
        if blocks:
            simple = [effect_row(task, label, k, b)[:-1] for task, label, b, _ in blocks
                      for k in CONTRASTS if k not in PRIMARY]
            parts += ["### Simple effects (descriptive, p not adjusted)", "",
                      table(["task", "outcome", "contrast", "n", "Δ", "95% CI", "p (sign-flip)"], simple), ""]

    if qa["strata"]:
        parts += render_strata(qa["strata"], arms, qa.get("train_vs_heldout"))
    elif qa["strata_note"]:
        parts += [f"Rules QA strata by the SFT section split are not shown: {qa['strata_note']}.", ""]

    parts += ["### Markers in rules QA answers", "",
              table(["arm", "answers", "invalid [R#]", "answerable", "Keeper ruling (answerable)", "unanswerable",
                     "Keeper ruling (unanswerable)"],
                    [[arm, v["n"], v["invalid_marker_rate"], v["n_answerable"], v["keeper_ruling_answerable"],
                      v["n_unanswerable"], v["keeper_ruling_unanswerable"]] for arm, v in qa["markers"].items()]), "",
              "Over all generated answers, graded or not. Invalid [R#]: the share of answers with a marker that is not "
              "among the retrieved passages; a prompt-only arm retrieves nothing, so there every [R#] counts (taken "
              "from the answer text). Keeper ruling: the harness's phrase detector (`keeper ruling` and its "
              "Ukrainian variants); expected on unanswerable questions, on answerable ones it means the model "
              "declined to answer from the rules or made up its own ruling.", ""]
    return parts


def render_strata(strata: dict, arms: list[str], versus: dict | None) -> list[str]:
    """Rules QA by the SFT section split: n, the metrics per arm, the FT effect per stratum and train vs held-out."""
    arm_of = dict(zip(CELLS, arms, strict=True))
    rows = [[STRATUM_LABELS[name], s["n"], arm_of[c], s["means"]["score"][c], s["means"]["strict_accuracy"][c],
             s["means"]["key_fact_coverage"][c] if name != "unanswerable" else None,
             s["means"]["confident_wrong"][c] if name == "unanswerable" else None]
            for name, s in strata.items() for c in CELLS]
    ft_rows = [[STRATUM_LABELS[name], s["n"], s["effects"]["ft_given_prompt"]["mean_diff"],
                s["effects"]["ft_given_prompt"]["ci95"], s["effects"]["ft_given_rag"]["mean_diff"],
                s["effects"]["ft_main"]["mean_diff"], s["effects"]["ft_main"]["ci95"], s["effects"]["ft_main"]["p"],
                s["effects"]["ft_main"]["p_holm"]] for name, s in strata.items()]
    ia_rows = [[STRATUM_LABELS[name], s["n"], s["effects"]["interaction"]["mean_diff"],
                s["effects"]["interaction"]["ci95"], s["effects"]["interaction"]["p"],
                s["effects"]["interaction"]["p_holm"]] for name, s in strata.items()]
    answerable = sum(s["n"] for name, s in strata.items() if name != "unanswerable")
    held = strata.get("heldout", {}).get("n", 0)
    share = f"The held-out stratum holds {held} of {answerable} answerable items ({held / answerable:.0%}). " \
        if held and answerable else ""
    by_size = sorted(strata.items(), key=lambda kv: kv[1]["n"])
    smallest = " and ".join(f"{name} ({s['n']} items)" for name, s in by_size[:2])
    parts = ["### Rules QA by the SFT section split", "",
             "Strata come from the frozen split recorded in the model cards (`data/sft/coc7e/section_split.v1.json`): "
             "`train` = every gold section of the item was in the train half the adapter was fine-tuned on, "
             "`held-out` = none was, `mixed` = both, `unanswerable` = the item has no gold section (strict acc is then "
             "the share of appropriate answers, confident wrong the hallucinations). An adapter that only memorised "
             f"the corpus gains on `train` and not on `held-out`. {share}The smallest strata are {smallest}: low "
             "power, so read the intervals rather than the p-values. p (Holm) adjusts across the strata, per "
             "contrast.", "",
             table(["stratum", "n items", "arm", "score", "strict acc", "key facts", "confident wrong"], rows), "",
             table(["stratum", "n items", "FT effect, prompt-only", "95% CI", "FT effect, with RAG", "FT main effect",
                    "95% CI", "p (sign-flip)", "p (Holm)"], ft_rows), "",
             table(["stratum", "n items", "interaction FT x RAG", "95% CI", "p (sign-flip)", "p (Holm)"], ia_rows), ""]
    if versus:
        parts += ["Train against held-out: the difference of the per-item FT effect between the two strata. They are "
                  "different items, so the sample is independent (bootstrap CI resampling each stratum, permutation p "
                  "over the stratum labels, Holm over the two). A positive difference means the adapter helps more on "
                  "the sections it was trained on, the signature of memorisation.", "",
                  table(["contrast", "n train", "n held-out", "train - held-out", "95% CI", "p (permutation)",
                         "p (Holm)"],
                        [[EFFECT_LABELS[k], v["n_train"], v["n_heldout"], v["mean_diff"], v["ci95"], v["p"],
                          v["p_holm"]] for k, v in versus.items()]), ""]
    return parts
