"""Review page for the gold datasets: one self-contained HTML file for the author's item-by-item review.

    cd src/backend
    uv run python -m evals.review                                # -> artifacts/review/gold-review-coc7e.html
    uv run python -m evals.review --summary path/to/export.json  # verdict counts and every non-ok item

The page shows each item next to the text of its gold sections, a per-item checklist, verdict buttons and notes.
Adjudication is reviewed per group (the clean item and its injection variants together). Verdicts stay in the
browser (localStorage) until exported as JSON; an export stores each unit's content hash, so --summary can tell
which verdicts no longer match the current files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from keeper.rag.corpus import load_corpus
from keeper.scenarios import ScenarioRepo
from server.settings import REPO_ROOT

from .datasets import CORPUS_DIR, DATASET_DIR, EvalDatasets, corpus_sections

TEMPLATE = Path(__file__).with_name("review_page.html")
PLACEHOLDER = "/*__DATA__*/"
AMBIGUITIES = REPO_ROOT / ".claude" / "skills" / "gold-data" / "references" / "corpus-ambiguities.md"
WHEN_TO_ROLL = "Skill Rolls > When to Roll"
SETS = ("rules_qa", "adjudication", "conflict_probes")
VERDICTS = ("ok", "wording", "ambiguous", "wrong")

# Borderline v1 items noted while drafting (gold-data skill, references/corpus-ambiguities.md): review these first.
FLAGS = {
    "qa-030": "«форсований» кидок може означати інший навик",
    "qa-075": "давати підказки без кидка — неоднозначність корпусу #2",
    "qa-031": "факт є і в іншій секції (#16)",
    "qa-064": "факт є і в іншій секції (#16)",
    "qa-086": "без відповіді, але погоні частково згадані в корпусі",
    "qa-091": "без відповіді: Credit Rating",
    "qa-088": "без відповіді: шкода від .38, але загальна шкала шкоди підштовхує до власного рішення",
    "qa-098": "без відповіді: мер Львова — модель може знати з пам'яті",
    "cp-004": "складне редагування",
    "cp-010": "складне редагування; мітка familiarity — суб'єктивна",
    "cp-020": "складне редагування",
    "cp-022": "слабка правдоподібність зміненого правила",
    "cp-024": "слабка правдоподібність зміненого правила",
    "cp-008": "мітка familiarity — суб'єктивна",
    "adj-g01": "підняти килим без кидка — за сценарієм, але Spot Hidden спокусливий",
    "adj-g18": "кидати чи ні — спірно",
    "adj-g19": "одна ціль, хоча Spot Hidden теж можливий (Library Use vs Spot Hidden)",
    "adj-g22": "одна ціль: Persuade vs Charm",
    "adj-g23": "одна ціль: Fast Talk",
    "adj-g24": "одна ціль: Charm vs Persuade",
    "adj-g29": "одна ціль: Occult vs Know/EDU",
    "adj-g32": "кидати чи ні — спірно",
    "adj-g33": "порядок: спершу SAN?",
    "adj-g37": "кидок INT сформульований частково поза грою",
    "adj-g38": "кидати чи ні — спірно",
    "adj-g40": "SAN належав попередньому ходу?",
}


def content_hash(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def review_units(ds: EvalDatasets) -> dict[str, list[dict]]:
    """The units a reviewer gives verdicts to: one per QA item and conflict probe, one per adjudication group."""
    groups: dict[str, list[dict]] = {}
    for a in ds.adjudication:
        groups.setdefault(a.group, []).append(a.model_dump(mode="json"))
    qa = [i.model_dump(mode="json") for i in ds.rules_qa]
    cp = [p.model_dump(mode="json") for p in ds.conflict_probes]
    return {
        "rules_qa": [{"id": i["id"], "hash": content_hash(i), "item": i} for i in qa],
        "adjudication": [{"id": g, "hash": content_hash(items), "items": items} for g, items in groups.items()],
        "conflict_probes": [{"id": p["id"], "hash": content_hash(p), "item": p} for p in cp],
    }


def ambiguities(path: Path = AMBIGUITIES) -> list[dict]:
    """Rows of the known-ambiguities table; empty if the reference file is missing."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|(.*)\|(.*)\|(.*)\|\s*$", line)
        if m:
            clean = [re.sub(r"[`\\]", "", c).strip() for c in m.groups()[1:]]
            rows.append({"n": int(m.group(1)), "topic": clean[0], "sections": clean[1], "text": clean[2]})
    return rows


def dataset_label(directory: Path) -> str:
    try:
        return directory.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return directory.resolve().name


def page_data(ds: EvalDatasets, directory: Path = DATASET_DIR) -> dict:
    chunks = load_corpus([CORPUS_DIR], chunk_size=1000, overlap=200, root=REPO_ROOT)
    sections = corpus_sections(chunks)
    pack = ScenarioRepo(REPO_ROOT / "data" / "scenarios").load("demo-haunt")
    sheet = pack.investigator("uk")
    scenario = pack.scenario_state("uk")
    units = review_units(ds)
    ids = {u["id"] for us in units.values() for u in us}
    when = next(key for key in sections if key[1] == WHEN_TO_ROLL)
    return {
        "meta": {"dataset": dataset_label(directory), "hashes": ds.hashes,
                 "generated_at": datetime.now(UTC).isoformat(timespec="seconds")},
        "units": units,
        "sections": [{"source": src, "section": sec, "source_type": cs[0].metadata["source_type"],
                      "chunks": [c.text for c in cs]} for (src, sec), cs in sections.items()],
        "sheet": {"name": sheet["name"], "characteristics": sheet["characteristics"], "skills": sheet["skills"],
                  "luck": sheet["luck"], "san": sheet["san"]},
        "scenario": {"title": scenario["title"], "keeper_notes": scenario["keeper_notes"],
                     "sanity_events": scenario["sanity_events"]},
        "when_to_roll": {"source": when[0], "section": when[1]},
        "flags": {k: v for k, v in FLAGS.items() if k in ids},
        "ambiguities": ambiguities(),
    }


def render(data: dict) -> str:
    # Escaping every "<" keeps "</script>" and "<!--" in the corpus text from ending the JSON block early.
    payload = json.dumps(data, ensure_ascii=False).replace("<", chr(0x5C) + "u003c")
    template = TEMPLATE.read_text(encoding="utf-8")
    assert template.count(PLACEHOLDER) == 1, "review_page.html must contain the data placeholder once"
    return template.replace(PLACEHOLDER, payload)


def summarize(export: dict, units: dict[str, list[dict]]) -> str:
    verdicts = export.get("units") or {}
    known = {u["id"] for us in units.values() for u in us}
    lines = [f"review of {export.get('dataset', '?')} by {export.get('reviewer') or '?'}, "
             f"exported {export.get('exported_at', '?')}"]
    for name in SETS:
        counts: Counter[str] = Counter()
        details = []
        for u in units[name]:
            r = verdicts.get(u["id"]) or {}
            verdict = r.get("verdict")
            stale = bool(verdict) and r.get("item_hash") != u["hash"]
            counts[verdict or "unreviewed"] += 1
            counts["stale"] += stale
            if (verdict and verdict != "ok") or stale:
                tags = ", ".join(t for t in (verdict, "stale" if stale else "") if t)
                text = f"  {u['id']} [{tags}]"
                if r.get("note"):
                    text += f" note: {r['note'].strip()}"
                if r.get("fix"):
                    text += f" | fix: {r['fix'].strip()}"
                details.append(text)
        order = [*VERDICTS, "unreviewed"]
        line = f"{name}: {len(units[name])} units, " + ", ".join(f"{k} {counts[k]}" for k in order)
        lines += [line + (f" (stale {counts['stale']})" if counts["stale"] else ""), *details]
    unknown = sorted(set(verdicts) - known)
    if unknown:
        lines.append(f"verdicts for ids not in the current dataset: {', '.join(unknown)}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Build the gold-data review page or summarise an exported review.")
    p.add_argument("--dir", type=Path, default=DATASET_DIR)
    p.add_argument("--out", type=Path, help="HTML path (default: artifacts/review/gold-review-<dir>.html)")
    p.add_argument("--summary", type=Path, help="exported review JSON to summarise instead of building the page")
    a = p.parse_args()
    ds = EvalDatasets.load(a.dir)
    if a.summary:
        print(summarize(json.loads(a.summary.read_text(encoding="utf-8")), review_units(ds)))
        return
    out = a.out or REPO_ROOT / "artifacts" / "review" / f"gold-review-{a.dir.resolve().name}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(page_data(ds, a.dir)), encoding="utf-8", newline="\n")
    print(out)


if __name__ == "__main__":
    main()
