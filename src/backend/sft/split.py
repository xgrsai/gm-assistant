"""Deterministic train / held-out split of the corpus SECTIONS for Phase 4 fine-tuning.

    cd src/backend
    uv run python -m sft.split            # print the split summary (nothing is written)
    uv run python -m sft.split --write    # (re)write data/sft/coc7e/section_split.v1.json
    uv run python -m sft.split --check    # recompute; exit 1 if the committed file differs

Rules-related SFT examples may only quote and cite sections from the `train` half; the `heldout` half is never
seen in training, so evaluation results can be stratified into seen / held-out gold items (a LoRA that merely
memorised the corpus looks good on seen items and bad on held-out ones).

Method (`stratified-hash-v1`):
  1. Every section gets ONE stratum: the `rules_qa` category that cites it most often as a gold section (ties go to
     the alphabetically first category); sections no `rules_qa` item cites are `uncategorized`. A section can serve
     several items and categories, the dominant one is just a stable bucket for balancing.
  2. Inside a stratum the sections are ordered by sha256(seed, stratum, source, section) - a seeded hash, so the
     order does not depend on file order or on Python's hash randomisation.
  3. In a stratum of n sections the first n // 2 go to `train` and the rest to `heldout`; in strata of odd size the
     extra section alternates between `train` and `heldout` (strata in alphabetical order), so the halves stay
     within one section of each other.

The frozen file also records the seed, the gold-file hashes and `corpus_chunk_ids_hash`: if the corpus or a gold
file changes, `--check` fails and the split must be reviewed (and versioned) rather than silently re-derived.
The odd-stratum alternation cascades: adding, removing or relabelling one gold item or section can move the extra
section of every later odd stratum (and the per-stratum hash order shifts with categories). So regenerating after ANY
corpus or gold-file change is a new split version (`section_split.v2.json`, new SFT data and new seen / held-out
strata), never an update of v1 - models trained on v1 would otherwise have seen what v2 calls held-out.
Sections are keys `(source, section)` exactly as in the gold datasets (`GoldSection.key()`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evals.datasets import CORPUS_DIR, DATASET_DIR, EvalDatasets, corpus_sections
from keeper.rag.corpus import Chunk, load_corpus
from server.settings import REPO_ROOT

SPLIT_PATH = REPO_ROOT / "data" / "sft" / "coc7e" / "section_split.v1.json"
SEED = 1927
METHOD = "stratified-hash-v1"
CHUNK_SIZE, OVERLAP = 1000, 200
UNCATEGORIZED = "uncategorized"

SectionKey = tuple[str, str]


class SectionSplit(BaseModel):
    """The frozen split as read back from disk (what decontamination and the SFT builder consume)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    seed: int
    method: str
    corpus_chunk_ids_hash: str = ""
    gold_hashes: dict[str, str] = {}
    train: list[dict]
    heldout: list[dict]

    def train_keys(self) -> set[SectionKey]:
        return {(s["source"], s["section"]) for s in self.train}

    def heldout_keys(self) -> set[SectionKey]:
        return {(s["source"], s["section"]) for s in self.heldout}


def load_split(path: Path = SPLIT_PATH) -> SectionSplit:
    return SectionSplit.model_validate_json(path.read_text(encoding="utf-8"))


def chunk_ids_hash(chunks: list[Chunk]) -> str:
    return hashlib.sha256("\n".join(sorted(c.id for c in chunks)).encode("utf-8")).hexdigest()[:16]


def dominant_categories(ds: EvalDatasets, sections: list[SectionKey]) -> dict[SectionKey, str]:
    votes: dict[SectionKey, Counter] = defaultdict(Counter)
    for item in ds.rules_qa:
        for g in item.gold_sections:
            votes[g.key()][item.category] += 1
    out = {}
    for key in sections:
        if votes[key]:
            out[key] = min(votes[key], key=lambda c: (-votes[key][c], c))
        else:
            out[key] = UNCATEGORIZED
    return out


def _order(seed: int, category: str, key: SectionKey) -> str:
    return hashlib.sha256("\x1f".join([str(seed), category, *key]).encode("utf-8")).hexdigest()


def assign(categories: dict[SectionKey, str], seed: int = SEED) -> dict[SectionKey, str]:
    """section -> 'train' | 'heldout' (see the module docstring for the rule)."""
    strata: dict[str, list[SectionKey]] = defaultdict(list)
    for key, cat in categories.items():
        strata[cat].append(key)
    out: dict[SectionKey, str] = {}
    odd_seen = 0
    for cat in sorted(strata):
        members = sorted(strata[cat], key=lambda k: _order(seed, cat, k))
        n_train = len(members) // 2
        if len(members) % 2:
            n_train += 1 if odd_seen % 2 == 0 else 0
            odd_seen += 1
        for i, key in enumerate(members):
            out[key] = "train" if i < n_train else "heldout"
    return out


def gold_item_coverage(ds: EvalDatasets, side: dict[SectionKey, str]) -> dict[str, dict[str, int]]:
    """For each task: how many gold items lie fully in train / fully in held-out / span both / have no gold section."""
    out: dict[str, Counter] = {"rules_qa": Counter(), "conflict_probes": Counter()}
    for item in ds.rules_qa:
        sides = {side[g.key()] for g in item.gold_sections}
        out["rules_qa"][_bucket(sides)] += 1
    for p in ds.conflict_probes:
        out["conflict_probes"][_bucket({side[p.gold_section.key()]})] += 1
    return {task: {k: c[k] for k in ("train", "heldout", "mixed", "no_gold")} for task, c in out.items()}


def _bucket(sides: set[str]) -> str:
    if not sides:
        return "no_gold"
    return "mixed" if len(sides) > 1 else next(iter(sides))


def build_split(ds: EvalDatasets, chunks: list[Chunk], seed: int = SEED) -> dict:
    sections = sorted(corpus_sections(chunks))
    categories = dominant_categories(ds, sections)
    side = assign(categories, seed)
    gold_sections = ({g.key() for i in ds.rules_qa for g in i.gold_sections}
                     | {p.gold_section.key() for p in ds.conflict_probes})
    unknown = sorted(gold_sections - set(sections))
    if unknown:
        raise ValueError(f"gold sections missing from the corpus: {unknown}")

    def entries(which: str) -> list[dict]:
        return [{"source": k[0], "section": k[1], "category": categories[k]} for k in sections if side[k] == which]

    per_category = {cat: {w: sum(1 for k in sections if categories[k] == cat and side[k] == w)
                          for w in ("train", "heldout")} for cat in sorted(set(categories.values()))}
    return {
        "version": 1,
        "method": METHOD,
        "seed": seed,
        "chunking": {"chunk_size": CHUNK_SIZE, "overlap": OVERLAP},
        "corpus_chunk_ids_hash": chunk_ids_hash(chunks),
        "gold_hashes": dict(sorted(ds.hashes.items())),
        "counts": {
            "sections": len(sections),
            "train": sum(1 for k in sections if side[k] == "train"),
            "heldout": sum(1 for k in sections if side[k] == "heldout"),
            "gold_sections": len(gold_sections),
            "gold_items": gold_item_coverage(ds, side),
            "per_category": per_category,
        },
        "train": entries("train"),
        "heldout": entries("heldout"),
    }


def render(split: dict) -> str:
    return json.dumps(split, ensure_ascii=False, indent=2) + "\n"


def summary(split: dict) -> str:
    c = split["counts"]
    lines = [f"{split['method']} seed={split['seed']}: {c['sections']} sections -> {c['train']} train / "
             f"{c['heldout']} held-out ({c['gold_sections']} are gold sections)"]
    for task, cov in c["gold_items"].items():
        lines.append(f"  {task}: fully train {cov['train']}, fully held-out {cov['heldout']}, "
                     f"mixed {cov['mixed']}, no gold section {cov['no_gold']}")
    lines.append("  per category (train/held-out): "
                 + ", ".join(f"{k} {v['train']}/{v['heldout']}" for k, v in c["per_category"].items()))
    return "\n".join(lines)


def compute(dataset_dir: Path = DATASET_DIR, corpus_dir: Path = CORPUS_DIR, seed: int = SEED) -> dict:
    ds = EvalDatasets.load(dataset_dir)
    chunks = load_corpus([corpus_dir], chunk_size=CHUNK_SIZE, overlap=OVERLAP, root=REPO_ROOT)
    return build_split(ds, chunks, seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=Path, default=SPLIT_PATH)
    p.add_argument("--seed", type=int, default=SEED)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write the split file")
    mode.add_argument("--check", action="store_true", help="exit 1 if the committed file differs from a recompute")
    a = p.parse_args()
    split = compute(seed=a.seed)
    text = render(split)
    print(summary(split))
    if a.write:
        a.path.parent.mkdir(parents=True, exist_ok=True)
        with a.path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"wrote {a.path}")
    elif a.check:
        committed = a.path.read_bytes().decode("utf-8").replace("\r\n", "\n") if a.path.exists() else None
        if committed != text:
            print(f"DIFFERS: {a.path} is missing or does not match the recomputed split "
                  "(corpus, gold files or seed changed?)")
            raise SystemExit(1)
        print("OK: committed split reproduced")


if __name__ == "__main__":
    main()
