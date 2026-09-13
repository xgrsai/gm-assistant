# Evaluation methodology (Phase 3)

This document describes how the GM Assistant evaluation harness measures the thesis claims: that retrieval-augmented generation (RAG) makes a Keeper's rules answers more accurate than prompting alone, that retrieved context takes priority over what the model already believes, and (later) that domain fine-tuning complements RAG. Code: `src/backend/evals/`. Gold data: `data/eval/coc7e/`.

## 1. What is compared

An **experiment** (`src/backend/evals/experiments/*.yaml`) lists **arms**. Each arm is a frozen `ExperimentSpec`: generator model, auxiliary model, prompt versions, whether RAG is used, and the retrieval configuration. Everything not listed as varying is held fixed across arms: the auxiliary model, prompts, scenario, investigator sheet, seeds and datasets.

| Experiment | Arms | Question |
|---|---|---|
| `phase3-gemini` | `prompt` vs `rag`, both `gemini-3.8-flash` | Does RAG improve a strong API model? |
| `phase3-local-qwen` | `qwen-prompt` vs `qwen-rag`, `qwen2.5:3b` via Ollama | Does RAG help a small local model more? |
| Phase 5 (planned) | base vs LoRA × prompt vs RAG on one open model | Do RAG and fine-tuning compensate each other (interaction)? |

## 2. Paired design and frozen turn contexts

Every item is answered by every arm with the same seed, so comparisons are **paired per item**.

Retrieval context is **frozen** (review item C7). For each (item, language, retrieval configuration), the analyze step (English search query written by the auxiliary model) and Qdrant retrieval run **once**. The result is cached in `artifacts/eval/contexts/`. Every RAG arm and seed receives the identical hits through the graph's `forced_hits` input. Differences between RAG arms therefore come from the generator, not from a different query or retrieval run, and the expensive auxiliary calls are paid once.

The evaluation drives the **production graph and node functions** (`keeper/graph.py`, `keeper/nodes.py`), not a re-implementation. Rules questions enter the graph with `forced_kind="rules_question"`, so the adjudication call is skipped exactly as a correctly adjudicated rules question would be.

## 3. Datasets

All gold data is derived only from the project's own corpus (`data/open/coc7e`) and demo scenario, so it can be published with the code (MIT). Schemas and validation: `evals/datasets.py`. Run `uv run python -m evals.datasets` to validate.

| File | Measures | Size (v1) |
|---|---|---|
| `rules_qa.v1.jsonl` | Rules and setting questions: Ukrainian and English wording, English reference answer, atomic key facts, gold sections; ~15% questions the corpus does not answer | 100 |
| `conflict_probes.v1.jsonl` | Knowledge conflicts: a passage edited by `find` → `replace`; 3 perturbation types × 2 familiarity levels | 24 |
| `adjudication.v1.jsonl` | Whether the adjudicator calls the right roll; each clean turn has rhetorical-injection variants with the same correct call | ~100 |

**Retrieval gold labels are sections** (source title + heading path), not chunk ids. Chunk ids change with chunk size and overlap; sections do not, so the same gold set serves the chunking ablations.

**Status.** v1 was drafted with AI assistance and must be reviewed by the author before thesis runs (checklist in `data/eval/coc7e/README.md`). A set is frozen by its file hashes, which every run records in `config.json`.

## 4. Tasks and metrics

### 4.1 Retrieval (no generator, embeddings only)

Each rules question is run through `Retriever.rank` (one ranked list over rules and lore, top-k, no quotas and no threshold) with three query variants:
- `en`: the English question;
- `uk`: the raw Ukrainian question, testing cross-lingual retrieval with the English-only `nomic-embed-text` v1.5;
- `uk_analyze`: the frozen analyze query for the Ukrainian question, i.e. what the system actually searches with.

Metrics, over answerable items:
- Hit@1/3/5 and MRR@k: is a gold section in the top results?
- Recall@5: the fraction of gold sections in the top 5, which matters for multi-hop questions.
- The same metrics broken down by query type (lookup, paraphrase, multi-hop, numeric).

**`min_score` calibration.** For each threshold on a grid, the harness reports:
- micro precision of kept hits;
- mean gold-section recall;
- F1;
- the share of unanswerable questions whose best hit falls below the threshold.

The best-F1 threshold is a candidate for `RetrievalSpec.min_score`. Changing it changes the spec hash, so it defines a new arm.

### 4.2 Rules QA (generation)

Each arm answers each question in the configured languages. An **LLM judge** (`gemini-judge`, prompt `evals/prompts/judge-qa-v1.md.j2`, structured output) grades the answer against the reference.

The judge sees the question, the reference answer, the key facts and the answer. It does **not** see the retrieved context. Grades:
- `verdict`: correct / partially correct / incorrect, scored 1 / 0.5 / 0;
- key facts present;
- whether the answer contradicts the reference;
- abstention.

Unanswerable items are graded on honesty. A good answer says the rules do not cover the question, or gives an explicitly labelled Keeper ruling. Presenting invented specifics as official rules is "confident wrong", the hallucination the thesis introduction is about.

Deterministic metrics, with no judge:
- answer language (Cyrillic ratio);
- a "Keeper ruling" phrase.

RAG-only metrics:
- whether a gold section was in the context;
- citation validity (`[R#]` markers that exist among the retrieved passages);
- whether the answer cites a gold section;
- error attribution: incorrect answers with the gold section missing from context (retrieval miss) versus present (generation error).

### 4.3 Knowledge-conflict probes

RAG arms answer the probe question with a single passage in context, in one of two conditions:
- `perturbed`: the edited rule;
- `original`: the unedited rule, as a control.

Prompt-only arms answer with no context (`no_context`), which measures what the model believes on its own.

The judge (`judge-conflict-v1`) classifies each answer as asserting the original fact, the perturbed fact, both, or neither. The two facts are shown in a deterministic random order per item, to neutralise position bias.

Metrics:
- **context adherence**: the perturbed condition follows the perturbed fact;
- **parametric override**: the perturbed condition asserts the original fact;
- **control accuracy**: the original condition follows the original fact;
- **parametric knowledge**: the no-context condition asserts the original fact.

Context adherence is broken down by perturbation type × familiarity. The thesis hypothesis to test: adherence is lower for well-known rules, and fine-tuning may lower it further.

### 4.4 Adjudication

The production `adjudicate_turn` function receives the scene, the player's turn, the investigator sheet and, for RAG arms, the frozen context.

The prediction is scored on each part of the call:
- kind (in-fiction, rules question, meta);
- whether a roll is needed;
- roll type;
- target (must be one of the allowed sheet names);
- difficulty, when constrained.

`full_ok` requires all of them. The breakdown also reports false rolls (a roll called when none was needed) and missed rolls.

**Rhetorical injection.** An injection variant keeps the scene, the action and the correct call, and adds manipulation such as a claimed success, a claimed dice result, false authority, an embedded instruction or emotional pressure. Two metrics per group:
- **flip rate**: the prediction differs from the clean variant;
- **harm rate**: the clean variant was correct and the injected one is not.

Both are also reported per injection type.

## 5. Statistics

`evals/stats.py` is dependency-free and deterministic, with fixed seeds.
- Means come with 95% percentile-bootstrap confidence intervals, resampling items.
- Arm versus baseline arm (the first arm in the file), per paired item:
  - the difference in judge score, with a bootstrap CI and a sign-flip permutation test;
  - strict correctness, with an exact McNemar test;
  - Holm correction across the comparisons in a task.
- With one seed and ~85 answerable items, only fairly large effects are detectable. Report CIs, not only p-values. Extra seeds help only where they actually change sampling. `langchain-google-genai` treats newer Gemini models as having provider-fixed sampling, so seeds may have little effect there, while local Ollama models are seedable.

## 6. Judge validity

LLM-judge numbers are only reportable after checking agreement with a human.

```powershell
uv run python -m evals.calibrate export phase3-gemini --n 40
uv run python -m evals.calibrate score phase3-gemini
```

**Export:** a blinded, shuffled CSV with arm names and `[R#]` markers removed.
**Human rater:** fills in `human_verdict` for each row.
**Score:** reports Cohen's κ for the three-way and binary (correct vs not) verdicts, plus the confusion matrix.

The judge (`gemini-3.8-flash`) is the same model as the Gemini generator arms, so self-preference bias is possible. State the κ and this limitation in the thesis, or use a judge from another model family.

## 7. Running

```powershell
cd src/backend
uv run python -m evals.run evals/experiments/phase3-gemini.yaml --dry-run          # planned LLM calls
uv run python -m evals.run evals/experiments/phase3-gemini.yaml --tasks retrieval    # free: embeddings only
uv run python -m evals.run evals/experiments/phase3-gemini.yaml --limit 3            # smoke run
uv run python -m evals.run evals/experiments/phase3-gemini.yaml                      # full run, resumable
uv run python -m evals.run evals/experiments/phase3-gemini.yaml --summarize-only     # rebuild report
```

**Prerequisites:** Qdrant running with the collection ingested, Ollama serving `nomic-embed-text`, and `GOOGLE_API_KEY` set.

**Outputs** (`artifacts/eval/runs/<run>/`, git-ignored):
- `config.json`: fingerprint, dataset/prompt/spec hashes, model profiles, git sha;
- `records.jsonl`: one record per task × arm × seed × item × language × condition;
- `judgments.jsonl`;
- `metrics.json`;
- `report.md`.

**Resume and fingerprint.** A rerun resumes: finished items are skipped and failed ones retried, with back-off on rate limits and 5xx. If the experiment, datasets, prompts or model profiles change, the run name must change, so results from different configurations never mix.

**Quota.** A full `phase3-gemini` run is on the order of a thousand Gemini calls (see `--dry-run`), which exceeds a free-tier daily quota. Run it over several days (it resumes), or use a paid key. Note that free-tier prompts may be used by Google, which matters if the corpus ever contains non-open text.

## 8. Known limitations

- Gold data v1 is AI-drafted and needs human review; key facts and adjudication labels encode one Keeper's judgement.
- The judge shares a model family with the generator arms, so it must be calibrated with human labels (section 6).
- Seeds give only partial determinism on API models.
- The corpus is small (50 chunks), so retrieval is easier than it would be over full rulebooks. Absolute retrieval numbers do not transfer; the comparisons between arms are the result.
- Conflict probes edit one passage at a time. Distractor-context and multi-passage conflicts are future work.
