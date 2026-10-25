# Evaluation methodology (Phase 3)

This document describes how the GM Assistant evaluation harness measures the thesis claims: that retrieval-augmented generation (RAG) makes a Keeper's rules answers more accurate than prompting alone, that retrieved context takes priority over what the model already believes, and (later) that domain fine-tuning complements RAG. Code: `src/backend/evals/`. Gold data: `data/eval/coc7e/`.

## 1. What is compared

An **experiment** (`src/backend/evals/experiments/*.yaml`) lists **arms**. Each arm is a frozen `ExperimentSpec`: generator model, auxiliary model, prompt versions, whether RAG is used, and the retrieval configuration. Everything not listed as varying is held fixed across arms: the auxiliary model, prompts, scenario, investigator sheet, seeds and datasets.

| Experiment | Arms | Question |
|---|---|---|
| `phase3-gemini` | `prompt` vs `rag`, both `gemini-3.8-flash` | Does RAG improve a strong API model? |
| `phase3-local-qwen` | `qwen-prompt` vs `qwen-rag`, `qwen2.5:3b` via Ollama | Does RAG help a small local model more? |
| `phase4-qwen-ft` | `base-prompt`, `base-rag`, `ft-prompt`, `ft-rag` on self-quantized Qwen2.5-3B-Instruct (Q4_K_M): base vs LoRA-fine-tuned narrator × prompt vs RAG | Do RAG and fine-tuning compensate each other (interaction)? |
| `phase4-qwen-ft-selfadj` | `base-selfadj` vs `ft-selfadj`, each model adjudicates its own turns | Does the narrator-only adapter change adjudication? |
| Phase 5 (planned) | the full matrix and ablations | Which retrieval and chunking choices matter once the models are fixed? |

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

### 6.1 External judge (Claude Code subagents)

**Why.** The free Gemini key allows 5 requests a minute and `gemini-3.8-flash` is often overloaded, so grading the 544 answers of `phase4-qwen-ft` with `gemini-judge` was not practical. The author decided to grade that run with Claude Sonnet subagents in a Claude Code session instead. The harness cannot call subagents, so it **exports the exact judge prompts to files and imports the judgments**; everything downstream (`judgments.jsonl`, `metrics.py`, the 2x2 report) is unchanged.

**Pieces.**
- profile `claude-sonnet-agent` in `config/models.yaml`: `provider: external`, `model: claude-sonnet-5-5`. `ModelFactory` refuses to build it ("graded outside the harness; use python -m evals.external_judge"), and `evals.run` never calls it: `--judge-only` only says so, and `--dry-run` counts its answers on the line "to grade outside the harness" instead of as Gemini calls;
- experiment `evals/experiments/phase4-qwen-ft-claude-judge.yaml`: `phase4-qwen-ft` with another `judge` and `description`. It keeps `name: phase4-qwen-ft` on purpose, because the name is part of the run fingerprint: the judge is not, so the file binds to the existing run (`--summarize-only` reports the 544 answers as not graded yet; `export` and `import` stop if the fingerprint differs);
- the same judge prompts (`judge-qa-v1`, `judge-conflict-v1`) and the same schemas (`QAJudgment`, `ConflictJudgment`) as `gemini-judge`: the functions that render a prompt and post-process a judgment (`rules_prompt`, `rules_result`, `conflict_prompt`, `conflict_result` in `evals/judge.py`) are shared by the Gemini path and the external path, so the two cannot drift apart.

**Workflow** (from `src/backend`):

```powershell
uv run python -m evals.external_judge export evals/experiments/phase4-qwen-ft-claude-judge.yaml --batches 10
# one subagent per artifacts/eval/runs/phase4-qwen-ft/external_judge/claude-sonnet-agent/batch-XX.jsonl
uv run python -m evals.external_judge import evals/experiments/phase4-qwen-ft-claude-judge.yaml
uv run python -m evals.run evals/experiments/phase4-qwen-ft-claude-judge.yaml --summarize-only
```

A line of a batch is `{"id", "schema", "prompt"}`. The agent answers each prompt as an object of the schema named in the line (`schemas.json` holds both pydantic JSON schemas) and writes `batch-XX.out.jsonl` next to the batch, one `{"id": ..., "judgment": {...}}` per line, where `judgment` has exactly the fields of the schema (for `ConflictJudgment`, `follows` is the letter the judge sees, `a`/`b`/`both`/`neither`; the harness maps it back to the original or perturbed fact).

**Blinding.** The batches contain an opaque id, the schema name and the prompt, never the arm, the condition, the record key or the run name; those stay in the private `map.json` (id -> record key, task, item, language, prompt version, whether the two facts were shown swapped). The export tag is part of every id, items inside a batch are shuffled and ids are assigned after shuffling, so neither the id nor the position says anything about the arm. **The agent must be given only its batch file and `schemas.json`**: `map.json`, `records.jsonl` and `config.json` in the same run directory name the arms. What stays visible is the answer text itself; a fine-tuned or RAG answer can often be recognised by its style, `[R#]` markers or the dice line, as with any judge, and the prompt (unchanged, so that the two judges stay comparable) tells the judge to ignore the markers.

**Pairing.** All answers to one (task, item, language) go into the same batch, so the arms and conditions of an item, hence every paired contrast of the 2x2 analysis, are graded by the same agent; groups are dealt to batches with a seeded RNG (`--seed`, default 7, balanced by item count). Differences in strictness between agents therefore add noise to the contrasts instead of favouring an arm. The batch of every judgment is kept in its `external` field.

**Import.** `import` reads every `batch-XX.out.jsonl`, validates each judgment with the pydantic model of its item, applies the same post-processing as the harness (key-fact indices, score, mapping of the swapped facts) and appends records shaped like the harness's own (`record_key`, `task`, `judge` = `claude-sonnet-agent|<prompt version>|<prompt hash>`, `judgment`, `status`, plus `external: {batch, export}`; `metrics.py` ignores the extra field except to print a note in the report). It is idempotent. Invalid JSON, unknown or duplicate ids and schema errors are reported and the valid lines are still imported; the command exits non-zero and lists the ids still missing until every exported answer has a judgment. For a second pass, run `export --force` after importing: it exports only the answers not graded yet under new ids and moves the previous export to `archive/` (nothing is deleted). It refuses to run if the judge prompts or the gold data changed since the export, and notes the import in the `sessions` of `config.json`, whose `experiment` block still names the judge of the run's first session.

**Reproducibility.** The prompts are exact and the export is a function of the answers, the seed and the batch count, but the agents' judgments are not reproducible (no seed or temperature control). The imported `judgments.jsonl` together with the `external_judge/` files (batches, outputs, `map.json`) are the record; keep them with the run.

**Limitations.** (1) The SFT targets were written by Claude, so a Claude judge may favour the style of the fine-tuned model (self-preference); the risk is in the direction of the effect the thesis measures. (2) By the author's decision this run has no second judge, so there is no cross-judge check. (3) Agents are not deterministic, and each batch is one agent, so a batch effect is possible (see Pairing). (4) The only validity check is the human calibration, which must be reported with the numbers: `uv run python -m evals.calibrate export phase4-qwen-ft --n 40 --experiment evals/experiments/phase4-qwen-ft-claude-judge.yaml`, label without looking at the verdicts, then `score` the same way (`--experiment` makes calibrate use the external judge's signature instead of the judge in the run's `config.json`). It covers rules-QA verdicts only, not the conflict-probe classification.

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
- `external_judge/<judge>/`: only for an external judge (section 6.1);
- `metrics.json`;
- `report.md`.

**Resume and fingerprint.** A rerun resumes: finished items are skipped and failed ones retried, with back-off on rate limits and 5xx. If the experiment, datasets, prompts or model profiles change, the run name must change, so results from different configurations never mix.

**Quota.** A full `phase3-gemini` run is on the order of a thousand Gemini calls (see `--dry-run`), which exceeds a free-tier daily quota. Run it over several days (it resumes), or use a paid key. Note that free-tier prompts may be used by Google, which matters if the corpus ever contains non-open text.

## 8. Phase 4: base vs LoRA-fine-tuned narrator

**Models.** The author's Ollama rejects `ADAPTER`, so the adapter cannot be applied on top of the official `qwen2.5:3b` blob. Instead (the merged-weights fallback of review item C4, which keeps precision parity between the arms) two GGUFs are converted and quantized to Q4_K_M by the same llama.cpp and created in Ollama with the official qwen2.5:3b chat template:
- `keeper-qwen25-3b:sft-v1` (profile `keeper-qwen25-3b-ollama`): the LoRA merged into full-precision Qwen2.5-3B-Instruct;
- `qwen2.5-3b-selfq4:v1` (profile `qwen25-3b-selfq4-ollama`): the same weights, same conversion, no LoRA.

Sampling (temperature 0.7), `num_ctx`, context window and output budget are copied from `qwen25-3b-ollama`, so the two differ only by the model. **The Phase 4 base is not the official `qwen2.5:3b` blob that `phase3-local-qwen` used** (another quantizer), so Phase 3 and Phase 4 numbers are not directly comparable; compare base and fine-tuned arms inside one Phase 4 run. The frozen RAG contexts are shared with `phase3-local-qwen` (same aux model, analyze prompt and retrieval).

**Design.** `phase4-qwen-ft` copies `fixed`, retrieval, `context_query: question_en`, the tasks and the judge from `phase3-local-qwen`; only the generator differs. Adjudication is off (v1 trains the narrator only), and the arms are in the order `base-prompt` (baseline of every paired comparison), `base-rag`, `ft-prompt`, `ft-rag`. Generation is local and calls no Gemini model: `--dry-run --no-judge` prints 0 on the line "of which Gemini (API quota)". Only the judge uses Gemini, so answers are generated first (`--no-judge --concurrency 1`) and graded later with `--judge-only`, in chunks that fit the quota. The `phase4-qwen-ft` run itself is graded by an external judge instead (`phase4-qwen-ft-claude-judge.yaml`, section 6.1), because the free Gemini quota is too small.

**Adjudication ablation.** `phase4-qwen-ft-selfadj` uses `adjudicator: generator`, `use_rag: false`, judge `null` and only the adjudication task: each model adjudicates the gold turns itself. The adjudication metrics are deterministic (section 4.4), so no judge is needed.

**Model card and checks.** One JSON card (`training/model_cards/keeper-qwen25-3b-sft-v1.json`, written by `python -m sft.check_adapter`) describes the pair: the `status` of the adapter check, the Ollama digest of each tag (`ollama_digests.ft`, `ollama_digests.base`), the hashes of each model's weights, of the train/val files and of the frozen section split, plus the training and conversion settings. Both profiles name it in `model_card`. Before anything is generated, `evals.run` requires that:
- the card exists and its `status` is `passed`. The check writes the card even when its gates fail (for instance when the adapter has no effect on the outputs), and the review item C4 demands it passes before any fine-tuned arm runs. `--allow-failed-card` overrides this deliberately; the override is recorded in `config.json` (`allow_failed_card`);
- Ollama serves each tag under the digest in the card, so a tag that was re-created or re-pulled cannot silently change a run.

`--dry-run` only warns about all of this, `--judge-only` checks the cards but not Ollama, and `--summarize-only` checks nothing.

The run's `config.json` records a `model_cards` entry per profile (status, digest, weights, data and split hashes, and the informational hash of the card file). The fingerprint covers all of these except the card file hash, because the card also holds timestamps and git state: running the check again on the same models does not orphan a run, while another status, digest, weights, data or split does. Runs without such a profile record exactly what they recorded before.

**The 2x2 analysis.** When a run has one arm per cell of `finetuned` (generator profile) x `use_rag` (arm), read from `config.json` and not from arm names, `metrics.py` adds the section "2x2" to `report.md` and a `factorial` key to `metrics.json`. If a base cell has several arms, the one running the base declared by the fine-tuned profile is used and the others are listed and ignored. A caution is printed when anything but the model differs between the arms: other spec fields, sampling or context settings of the profiles, a base arm that is not the declared base. Any other run, Phase 3 or the two-arm `phase4-qwen-ft-selfadj`, gets neither: its report stays byte for byte what it was. The statistics follow section 5 (same seed), per item and over the answers graded in all four arms; an answer that is missing or not graded in one arm is dropped from all four and counted:
- **Contrasts**: the fine-tuning and RAG main effects, and the interaction `(ft_rag - base_rag) - (ft_prompt - base_prompt)`; a negative interaction means the adapter helps less once RAG supplies the facts (the two partly substitute). Each has a bootstrap CI and a sign-flip p, with Holm over these three per outcome. The four simple effects (FT with and without RAG, RAG for each model) are reported descriptively. Outcomes: the judge score and key-fact coverage on answerable rules questions, and adjudication `full_ok` per group (a clean turn with its injection variants), analysed only when the adjudicator is the generator in every arm.
- **Conflict probes** have no interaction, because the prompt-only arms answer without context: the report gives the paired FT effect on context adherence and control accuracy (RAG arms) and on parametric knowledge (prompt-only arms).
- **Strata**: the rules questions are split by the frozen section split (`data/sft/coc7e/section_split.v1.json`) whose hash the model cards record: all gold sections in the train half (the adapter has seen them), all held-out, mixed, and the unanswerable questions as their own stratum. A memorising adapter gains on the first and not on the held-out one, so the report also gives the difference of the per-item FT effect between the train and the held-out items (independent samples: bootstrap CI, permutation p, Holm over the without-RAG effect and the main effect). The held-out stratum holds 33 of the 85 answerable questions on the v1 gold set (the report computes the share), and the mixed and unanswerable strata are smaller still, so the intervals are wide. The strata are skipped, with the reason stated, if the split file is not the one the cards hash or the cards disagree.
- **Markers**: per arm, the share of rules answers with an invalid `[R#]` marker (prompt-only arms retrieve nothing, so any marker is invalid) and with the "Keeper ruling" phrase, for answerable and unanswerable questions separately. They need no judge and are available before grading.

## 9. Known limitations

- Gold data v1 is AI-drafted and needs human review; key facts and adjudication labels encode one Keeper's judgement.
- The judge shares a model family with the generator arms, so it must be calibrated with human labels (section 6).
- The `phase4-qwen-ft` run is graded by Claude subagents (section 6.1): the SFT targets were written by Claude, so the judge may favour the fine-tuned style, and there is no second judge.
- Seeds give only partial determinism on API models.
- The corpus is small (50 chunks), so retrieval is easier than it would be over full rulebooks. Absolute retrieval numbers do not transfer; the comparisons between arms are the result.
- Conflict probes edit one passage at a time. Distractor-context and multi-passage conflicts are future work.
- Phase 4 fine-tunes one small model on one adapter run (v1, one seed): the base-vs-fine-tuned contrast is a single training run, not a distribution over runs.
