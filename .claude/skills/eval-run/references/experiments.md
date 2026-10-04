# Eval experiment files

Location: `src/backend/evals/experiments/*.yaml` (NOT `src/backend/experiments/`, which holds the app's play arms). Schema: `evals/experiment.py` (`EvalExperiment`).

## Fields

```yaml
name: phase3-gemini              # also the default run directory name; [a-z0-9_-]
description: ...                 # free text, not part of the fingerprint
scenario_id: demo-haunt          # scenario pack used for bootstrap, sheet and adjudication scenes
dataset_dir: eval/coc7e          # relative to DATA_DIR (repo data/)
corpus_dirs: [open/coc7e]        # must match what the RAG collection was built from
seeds: [11]
judge: gemini-judge              # model profile in config/models.yaml, or null
judge_prompt_versions: {rules_qa: judge-qa-v1, conflict_probes: judge-conflict-v1}   # evals/prompts/*.md.j2

fixed:                           # merged into every arm -> ExperimentSpec
  aux: gemini-aux
  adjudicator: aux               # or generator (ablation: the generator adjudicates)
  prompt_version: keeper-v2
  adjudicate_prompt_version: adjudicate-v2
  analyze_prompt_version: analyze-v1
  history_max_tokens: 6000
  retrieval: {collection: coc7e__nomic15__c1000o200, embedder: nomic15, quotas: {rules: 4, lore: 1}, exact: true, min_score: 0.55}

arms:                            # the FIRST arm is the baseline for paired comparisons
  - {arm: prompt, generator: gemini-flash, use_rag: false}
  - {arm: rag, generator: gemini-flash, use_rag: true}

tasks:
  retrieval: {k: 10, query_modes: [en, uk, uk_analyze], threshold_grid: [...]}   # enabled, limit also allowed
  rules_qa: {languages: [uk], context_query: analyze}      # or question_en (no LLM for contexts)
  conflict_probes: {languages: [uk], conditions: [perturbed, original, no_context]}
  adjudication: {}
```

An arm may override any `ExperimentSpec` field, including a whole `retrieval` block.

## Patterns

**Another generator model** (e.g. a local Ollama model): add a profile to `config/models.yaml` (see coc-keeper-dev skill), then a new experiment file with arms using it. Keep `aux`, prompts and retrieval identical to the experiment you want to compare with, so frozen contexts are shared and the comparison is paired.

**Retrieval ablation** (chunk size, embedder): ingest a new collection (knowledge-base skill), then arms that differ only in `retrieval`. Retrieval-task metrics use the first RAG spec only (`rag_specs()[0]`), so for a pure retrieval comparison create one experiment per retrieval configuration, or compare rules-QA/gold-in-context across arms.

**Adjudicator ablation:** two arms identical except `adjudicator: aux` vs `adjudicator: generator`.

**More seeds:** append to `seeds`. This changes the fingerprint (new run name). Only useful where sampling is seedable.

**Thesis 2×2 (Phase 5):** arms `base-prompt`, `base-rag`, `ft-prompt`, `ft-rag` on one open model family served at the same precision; the base-prompt arm first.

## After editing

1. `uv run python -m evals.run <file> --dry-run` (also validates the YAML and profiles).
2. If the fingerprint changed and a run with that name exists, use `--run-name <new>`.
