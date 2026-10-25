---
name: eval-run
description: Run, resume, grade and interpret GM Assistant evaluation experiments (src/backend/evals) — retrieval metrics, rules-QA with the LLM judge, knowledge-conflict probes, adjudication/injection tests, judge calibration, and reading report.md/metrics.json for the thesis. Use this whenever the user wants to evaluate the Keeper, compare arms (prompt-only vs RAG, Gemini vs local model, later fine-tuned), check retrieval quality or min_score, re-run or continue an eval, grade answers later, calibrate the judge, or asks what the numbers in an eval report mean — including Ukrainian requests like «запусти оцінку», «прожени eval», «порівняй RAG і без RAG», «що означають метрики», «докрути прогін».
---

# Running and interpreting evaluations

The harness measures the thesis claims with the production graph, not a copy. Methodology for the thesis text lives in `docs/evaluation.md`; this skill is the operational side: how to run things without wasting the author's API quota, and how to read the results honestly.

All commands run from `src/backend` with `uv run`. On Windows set `PYTHONUTF8=1` for any command that prints Ukrainian text.

## Before any run

1. **Prerequisites.** Qdrant up (`docker compose up -d qdrant` from the repo root; Docker Desktop must be running), Ollama serving `nomic-embed-text`, the collection ingested (see the knowledge-base skill), `GOOGLE_API_KEY` in `.env` for Gemini profiles.
2. **Validate the gold data:** `uv run python -m evals.datasets`. A run refuses invalid datasets.
3. **Dry run first, always**, and show the planned call counts to the user:
   `uv run python -m evals.run evals/experiments/phase3-gemini.yaml --dry-run`
   The author's Gemini key is free tier (low per-minute quota, a daily cap, occasional 503). A full `phase3-gemini` run is ~1000 calls. Ask before starting anything that spends more than a smoke run's worth of Gemini calls — quota is the scarce resource here, not tokens or time.

## Choosing what to run

| Goal | Command (add `--dry-run` first) | API calls |
|---|---|---|
| Retrieval metrics + `min_score` calibration | `... phase3-gemini.yaml --tasks retrieval --query-modes en,uk` | none (embeddings only) |
| Retrieval with the real analyze query | add `uk_analyze` to `--query-modes` | 1 aux call per question (cached) |
| Smoke test of every task | `... --limit 3` | small |
| Local generation, grade later | `... phase3-local-qwen.yaml --no-judge` | none (Ollama) |
| Base vs fine-tuned 2x2 (Phase 4) | `... phase4-qwen-ft.yaml --no-judge --concurrency 1`, then `--judge-only` | none for generation; the line "of which Gemini (API quota)" of `--dry-run` counts the rest |
| Grade answers already generated | `... --judge-only` | 1 judge call per answer |
| Grade with subagents (external judge) | `uv run python -m evals.external_judge export <yaml> --batches N`, then `import <yaml>` (`references/experiments.md`) | none from the harness |
| Rebuild the report only | `... --summarize-only` | none |

`--limit N` caps items per task (groups for adjudication). `--tasks` takes a comma list of `retrieval,rules_qa,conflict_probes,adjudication`. `--concurrency 1` suits Ollama on the 6 GB GPU.

## How runs behave (so you can explain surprises)

- **Resumable.** Output goes to `artifacts/eval/runs/<run-name>/`. Rerunning the same command skips finished items and retries failed ones, so a run interrupted by quota can simply be repeated tomorrow.
- **Fingerprint.** The run directory is bound to what changes results (arms, prompts and their hashes, model profiles, datasets, retrieval spec, `context_query`, `k`). Limits, task selection, query modes, languages, conditions and the judge are *not* part of it, so a smoke run grows into the full run. If the fingerprint differs, the CLI stops: pick a new `--run-name` instead of mixing configurations.
- **Frozen RAG contexts.** For RAG arms the query and hits are computed once per item and cached in `artifacts/eval/contexts/`, shared across arms, seeds and experiments with the same aux model + analyze prompt + retrieval spec + query mode. `context_query: question_en` builds them from the English gold wording without an LLM (an oracle translation; say so when reporting).
- **Judgments are separate** (`judgments.jsonl`, keyed by judge profile + prompt hash). Changing the judge never regenerates answers.

## Reading the report

Open `artifacts/eval/runs/<run>/report.md`. Before quoting any number:
- Check the header for **unresolved errors** and **answers not graded yet** — judge columns are empty until `--judge-only` runs.
- Numbers from `--limit` smoke runs are illustrations, not results.
- Judge-based numbers need a calibration (below) before they go into the thesis; the report prints a caveat when the judge shares a model with a generator arm.
- Prefer confidence intervals and paired comparisons over raw point estimates; with ~85 answerable items only large effects are detectable.

Metric definitions, formulas and what each one can and cannot show: `references/metrics.md`. Read it when interpreting or explaining results.

## Judge calibration

```
uv run python -m evals.calibrate export <run> --n 40   # blinded CSV: no arm names, no [R#] markers
# the author fills human_verdict: correct | partially_correct | incorrect
uv run python -m evals.calibrate score <run>           # Cohen's kappa (3-way and binary) + confusion matrix
```
The human should label without looking at the judge's verdicts. Roughly, κ ≥ 0.6 is usable with the limitation stated; lower means fix the judge prompt (new version file) or use a judge from another model family. For a run graded by an external judge add `--experiment <yaml with that judge>` to both commands (otherwise the judge in the run's `config.json` is used, which has judged nothing).

## Adding or changing an experiment

Experiment YAML format, the arm/spec fields and ready-made patterns (new model arm, retrieval ablation, second seed): `references/experiments.md`. Prompt and model-profile rules are in the coc-keeper-dev skill.

## When something fails

Common failures and fixes (quota 429s, Qdrant not running, fingerprint mismatch, empty stdout on Windows background runs, missing contexts): `references/troubleshooting.md`.
