---
name: gold-data
description: Author, extend, review, fix or version the GM Assistant gold evaluation datasets in data/eval (rules_qa, conflict_probes, adjudication JSONL) — writing questions with key facts and gold sections, knowledge-conflict edits, adjudication turns with rhetorical-injection variants, validating them, and resolving label disagreements or corpus ambiguities. Use this whenever the user wants to add or change eval items, review the AI-drafted v1 sets, check whether a gold label is right, make a v2, add a new category of test, or asks why an item counts as correct — including Ukrainian requests like «додай питань в датасет», «перевір золотий набір», «ця відповідь насправді правильна», «зроби нові ін'єкції», «рев'ю датасету».
---

# Gold evaluation datasets

These files are the measuring instrument of the thesis. A wrong label silently moves every metric, so accuracy and unambiguity matter far more than volume. Everything must be derived only from the project's own corpus (`data/open`) and demo scenario — never from Chaosium books — because the sets are published with the code (MIT).

Files: `data/eval/coc7e/{rules_qa,conflict_probes,adjudication}.v1.jsonl` plus `README.md` (status, counts, review checklist). Schemas and validation: `src/backend/evals/datasets.py`. Field-by-field reference with real examples: `references/schemas.md` — read it before writing items.

## Ground rules

- **The corpus is the ground truth, not memory of the real rulebook.** Where the project's rules reference differs from the published game, labels follow the corpus. Read the relevant chunk text in `artifacts/corpus/<collection>/chunks.jsonl` (exact wording, section paths) rather than recalling it.
- **Unambiguous or out.** If two reasonable Keepers could label an item differently (roll vs no roll, Spot Hidden vs Library Use when only one is allowed), either widen `allowed_targets`, rephrase until it is clear-cut, or drop it. The known ambiguous spots in the corpus are listed in `references/corpus-ambiguities.md`; avoid building items on them unless the item is about the ambiguity and the corpus is fixed first.
- **Gold retrieval labels are sections** (`source` = front-matter `source_title`, `section` = section path), copied exactly from the chunk dump. They survive re-chunking; chunk ids do not.
- **Ukrainian wording as a player types in chat**, not a literal translation; the English wording must ask the same thing.
- **Frozen means frozen.** Once any thesis run used a file (its hash is in a run's `config.json`), never edit it in place — make a new version (below). While the set is still DRAFT v1 and unused for reported results, fixing items in place is fine; say so explicitly.

## Workflow: adding or fixing items

1. Read `references/schemas.md` and the chunk dump for the sections involved.
2. Write the items (keep ids sequential: `qa-101`, `cp-025`, `adj-101` / group `adj-g41`).
3. Validate from `src/backend`: `PYTHONUTF8=1 uv run python -m evals.datasets`. It checks schema consistency, unknown sections, that `find` occurs exactly once in a single-chunk section, that targets exist on the investigator sheet, and that injection variants keep the clean label. Iterate until it exits 0.
4. Self-review against `references/review-checklist.md`, especially: every key fact literally supported by its gold section; unanswerable questions really unanswered (search the corpus for their key terms).
5. Update the counts in `data/eval/coc7e/README.md`.
6. Tell the user what changed and that file hashes (and therefore run fingerprints) changed.

## Workflow: reviewing the AI-drafted v1

The v1 files are marked DRAFT: the author must review them before thesis runs. The author reviews in the review page: `uv run python -m evals.review` (from `src/backend`) writes `artifacts/review/gold-review-coc7e.html` — each item next to its gold-section text, the checklist, verdicts (ok / wording / ambiguous / wrong), notes, corpus search; adjudication is reviewed per group. Verdicts are exported as JSON (keep it as `data/eval/coc7e/review-v1.json`, the methodology record); `uv run python -m evals.review --summary <export.json>` lists every non-ok and stale verdict. Help by going item by item with the checklist, grouping findings as *wrong label*, *ambiguous → fix or drop*, *wording*, *fine*. Present proposed edits as a diff-style list before applying them, since the author owns the labels. When the review is done, change the README status line to reviewed (with date and reviewer) — that is the moment the set becomes frozen.

## Workflow: creating a new version

`evals/datasets.py` currently hard-codes the v1 file names (`FILES`) and experiments only choose `dataset_dir`. For a v2, the simplest path that keeps old runs reproducible is a new directory `data/eval/coc7e-v2/` containing the three files under the same names, plus a README noting what changed versus v1; then point new experiments at `dataset_dir: eval/coc7e-v2`. Old runs keep their hashes in `config.json`. (If the user prefers versioned file names, that needs a small code change in `datasets.py` and `experiment.py` first — propose it rather than improvising.)

## Designing new kinds of items

- **Rules QA:** balance query types (lookup, paraphrase, multi_hop, numeric); paraphrase items are where retrieval struggles, multi-hop items must list every needed section. ~15% unanswerable items test honesty; the reference answer describes the ideal behaviour, not an invented rule.
- **Conflict probes:** minimal, grammatical edits that are plausible but different; keep the passage internally consistent (one `find` span can cover both mentions of a number); 4 items per perturbation × familiarity cell keeps the design balanced.
- **Adjudication:** scenes from the demo scenario that reveal no unfound secrets; injections keep the same scene and action so the clean label still holds; include injections that try to *force* a roll on no-roll turns, not only ones that try to skip rolls.
