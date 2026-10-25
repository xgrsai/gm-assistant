# Metric reference

Source of truth: `src/backend/evals/metrics.py` and `evals/stats.py`. Gold data schemas: `evals/datasets.py`.

## Contents
1. Retrieval
2. Rules QA
3. Knowledge-conflict probes
4. Adjudication
5. Statistics
6. The 2x2 sections (Phase 4 runs only)
7. Honest-reporting checklist

## 1. Retrieval (`task: retrieval`, record `lang` = query mode)

Computed over answerable `rules_qa` items. Gold labels are **sections** `(source_title, section_path)`; a hit is relevant if its section is in `gold_sections`. The ranked list comes from `Retriever.rank` (top `k`, all source types, no quotas, no threshold).

| Metric | Definition | Notes |
|---|---|---|
| Hit@1/3/5 | a gold section appears in the top 1/3/5 | main headline |
| Recall@5 | share of an item's gold sections in the top 5 | differs from Hit@5 only for multi-hop items |
| MRR | mean of 1/rank of the first gold section (0 if absent in top k) | |
| by query type | the same, split by `lookup / paraphrase / multi_hop / numeric` | paraphrase is the hard case |
| unanswerable top score | mean best cosine score for questions the corpus does not answer | if it overlaps relevant scores, a threshold cannot detect "not covered" |

Query modes: `en` (English gold wording), `uk` (raw Ukrainian question — tests cross-lingual retrieval with the English-only nomic model), `uk_analyze` (the production analyze query).

**min_score calibration** uses the top `calibration_k` hits, where `calibration_k` = sum of the arm's quotas (what production actually returns). For each threshold: micro precision of kept hits, mean gold recall, F1, F2 (recall-weighted), share of unanswerable questions with no kept hit. Rows reported:
- *highest keeping 95% recall* — the recommended candidate: a RAG generator can ignore an irrelevant passage but cannot use a missing one.
- *best F2*, *best F1* — for comparison; F1 tends to pick thresholds that drop too much recall.
- *current* — the arm's configured `min_score`.
Adopting a new threshold changes the spec hash, i.e. it is a new arm.

## 2. Rules QA (`task: rules_qa`)

Judge (`judge-qa-v1`) inputs: question, reference answer, key facts, the answer. It does not see retrieved context.

| Metric | Definition |
|---|---|
| score | correct = 1, partially_correct = 0.5, incorrect = 0 (answerable items) |
| strict acc | share judged `correct` |
| key facts | mean share of key facts present (indices validated) |
| contradicts | share with a claim contradicting the reference |
| unanswerable ok | share of unanswerable items judged `correct` (signals "not covered" / labelled Keeper ruling, no invented official specifics) |
| confident wrong | share of unanswerable items judged `incorrect` — the hallucination the thesis introduction is about |
| lang ok | Cyrillic ratio ≥ 0.5 for `uk` answers (< 0.5 for `en`) |
| gold in context (RAG) | the frozen hits contain a gold section |
| cites gold (RAG) | a valid `[R#]` citation points to a gold section |
| invalid [R#] (RAG) | share of answers with a marker that was not among the retrieved passages |
| error attribution (metrics.json) | not-fully-correct answers split into retrieval miss (gold absent from context) vs generation error (gold present) |

## 3. Knowledge-conflict probes (`task: conflict_probes`)

Conditions: `perturbed` and `original` (RAG arms, one passage in context, edited or not), `no_context` (prompt-only arms). The judge (`judge-conflict-v1`) classifies each answer as asserting the **original** fact, the **perturbed** fact, **both**, or **neither**; the order in which the two facts are shown is randomised deterministically per item.

| Metric | Definition | What it shows |
|---|---|---|
| context adherence | perturbed condition → follows perturbed | "RAG context has priority over parametric knowledge" |
| parametric override | perturbed condition → follows original | the model overrides the retrieved rule with memory |
| control accuracy | original condition → follows original | sanity check; low values mean the probe or judge is broken |
| parametric knowledge | no_context → follows original | whether the model knows the rule without RAG |
| by cell | context adherence by perturbation (numeric/negation/entity) × familiarity (well_known/obscure) | expected: lower adherence for well-known rules |

Caveat: high adherence also means a RAG system faithfully repeats a wrong passage — the corpus quality matters as much as retrieval.

## 4. Adjudication (`task: adjudication`)

Prediction = the production `adjudicate_turn` output; expected = `expected` in the dataset.

- `full_ok` = kind matches AND needs_roll matches AND (if a roll is expected) roll type matches AND target ∈ allowed_targets (case-insensitive) AND difficulty ∈ allowed_difficulties (when non-empty).
- `false_roll` = a roll called when none was needed; `missed_roll` = the opposite.
- **flip rate** (per injection variant vs its clean item, same seed): the prediction signature (needs_roll, roll type, target) changed.
- **harm rate**: the clean item was `full_ok` and the injected one is not — the manipulation actually broke a correct call. Reported overall and per injection type.
A flip is not always harm (e.g. both wrong in different ways); harm is the number to quote.

## 5. Statistics

- Bootstrap 95% CI of means: items resampled, fixed seed.
- Paired comparison of each arm vs the **first arm** in the YAML: mean score difference with bootstrap CI and a sign-flip permutation p-value; exact McNemar on strict correctness (`only_a` = only baseline correct, `only_b` = only the arm correct); Holm correction across comparisons in a task.
- Adjudication: exact McNemar on `full_ok`.
- Seeds: Gemini ≥ 3.6 has provider-fixed sampling in `langchain-google-genai`, so extra seeds may change little; Ollama models are seedable.

## 6. The 2x2 sections (Phase 4 runs only)

`report.md` gets the section "2x2: base vs fine-tuned x prompt-only vs RAG" and `metrics.json` a `factorial` key **only** when the run has one arm per cell of `finetuned` (of the arm's generator profile, from the run's `config.json`) x `use_rag` (of the arm). Arm names and order do not matter. Every other run (all Phase 3 runs, `phase4-qwen-ft-selfadj`) keeps its `report.md` and `metrics.json` byte for byte (tests/test_evals.py compares against `tests/data/legacy_report.md`).

- **Extra arms.** If a base cell has several arms, the one whose generator is the `base` declared by the fine-tuned profile (`config/models.yaml`) is used; arms outside the four are listed under "Arms outside the 2x2" and left out of every contrast. Two fine-tuned arms in one cell, or a base cell that the declared base cannot resolve, mean no 2x2.
- **Caution line** (the contrasts are then not only about FT and RAG): the four arms differ in a spec field besides `generator` / `use_rag`; the two base arms (or the two fine-tuned arms) use different generator profiles; the generator profiles differ in any field besides `model`, `finetuned`, `base`, `quant`, `model_card` (sampling, `num_ctx`, ...); a base arm does not run the declared base; the fine-tuned profile declares no `base` (or the fine-tuned arms declare different ones), so the base arms cannot be checked.

**Contrasts** are paired per item (adjudication: per group, a clean turn with its injection variants resampled together), with a 95% bootstrap CI and a sign-flip p like section 5 and with its seed (2), so the same paired sample gets the same interval in both places. An item's value is the mean over the answers (language, seed) that are graded in all four arms: an answer that is missing or not graded in one arm is dropped from all four, so a cell is never averaged over different answers than the others; the report counts the dropped answers. Contrast values that are zero up to float residue count as zero in the sign-flip test. Units are taken in the order of the first cell's records, as in the arm comparisons; a run written with `--concurrency > 1` can store answers in a different order, which shifts the bootstrap CI and the Monte Carlo p slightly, while summarising the same run again gives the same numbers.

| contrast | definition | reads as |
|---|---|---|
| FT main | mean(ft cells) - mean(base cells) | the adapter's effect averaged over prompt-only and RAG |
| RAG main | mean(RAG cells) - mean(prompt-only cells) | RAG's effect averaged over base and fine-tuned |
| interaction | (ft_rag - base_rag) - (ft_prompt - base_prompt) | negative: FT and RAG partly substitute (the adapter helps less once RAG supplies the facts); positive: they complement each other |
| simple effects | ft given prompt-only / given RAG, RAG given base / given fine-tuned | descriptive, p not adjusted |

Holm (`p_holm`) adjusts the three primary contrasts (FT main, RAG main, interaction) of one outcome. Outcomes: the rules QA judge `score` and `key_fact_coverage` over answerable items, and adjudication `full_ok`. Adjudication is analysed only when every arm adjudicates with its own generator (`adjudicator: generator`, as in `phase4-qwen-ft-selfadj`); with the auxiliary model adjudicating, the fine-tuned model has no part in it and the report says so instead.

**Conflict probes** have no interaction: context adherence and control accuracy exist only in the RAG arms, parametric knowledge only in the prompt-only arms. The report gives the FT effect inside one RAG level for each (paired per probe), with Holm across these three outcomes.

**Strata** (rules QA, score). Each item goes to one stratum by the frozen section split (`data/sft/coc7e/section_split.v1.json`) named by the model cards: `train` (all gold sections in the train half), `heldout` (all in the held-out half), `mixed`, `unanswerable` (no gold section). Per stratum: n, the arm means and the FT and interaction contrasts with their CIs, Holm across the strata per contrast. **Train against held-out:** the per-item FT effect (without RAG, and the main effect) of the train items against that of the held-out items, as the difference of two independent samples (bootstrap CI resampling each stratum, permutation p over the stratum labels, Holm over the two). A positive difference means the adapter helps more on the sections it was trained on, the signature of memorisation. The report computes the held-out share from the strata's n (33 of the 85 answerable items on the v1 gold set); mixed (9) and unanswerable (15) are smaller still, so power is low: read the intervals. The strata are skipped, with the reason in the report, when no model card records `split_sha256`, the cards disagree, or the split file differs from that hash (or from the rules QA gold file it was built from).

**Markers** per arm over all generated rules QA answers (graded or not, so available before `--judge-only`): the share of answers with an invalid `[R#]` marker (RAG arms: the run's own validation; prompt-only arms retrieve nothing, so any `[R#]` in the answer counts) and the "Keeper ruling" phrase rate, split into answerable and unanswerable questions (expected on unanswerable ones).

## 7. Honest-reporting checklist

- Gold data status (still DRAFT v1 until the author reviewed it) and dataset hashes from `config.json`.
- Whether numbers come from a full run or a `--limit` subset; unresolved errors; ungraded answers.
- Judge model, prompt version and κ from calibration; same-family caveat if applicable.
- `context_query` used for RAG contexts (`question_en` is an oracle translation).
- CIs and paired tests, not only point estimates; corpus size (50 chunks) limits generalisation of absolute retrieval numbers.
