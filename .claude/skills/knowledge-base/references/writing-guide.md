# Writing retrievable corpus sections

The retriever embeds whole chunks and the analyze node searches with a short English query written from the player's turn. Sections that match how players ask, and that stand alone when cut out, retrieve well and cite cleanly.

## Structure

- One mechanic or topic per `###` section, grouped under a `##` chapter (`## Sanity` → `### Sanity Rolls`, `### Temporary Insanity`). The `##/###` names become the citation shown to the player and the gold label in evals — make them short, stable and descriptive.
- Aim for ≤ ~900 characters per section including the heading, so it stays one chunk (chunk size 1000). Split longer topics into several `###` sections instead of relying on overlap. Conflict probes can only use single-chunk sections.
- Put the rule first, then conditions and exceptions, then an example. A chunk that begins mid-argument ("This also means…") retrieves poorly.

## Wording

- Self-contained: repeat the subject in each section ("A Sanity roll…", not "It…"); spell out numbers and formulas (`(CON + SIZ) divided by 10, rounded down`), because numeric questions are common.
- Use the words players search with, alongside the official term: "Spot Hidden — noticing hidden details, searching a room". Paraphrase questions were the weakest retrieval category.
- Mark uncertain or optional rules explicitly ("Keeper Rulebook, via secondary sources", "Optional rule"), and never state two conflicting versions of a rule in different sections without saying which applies — the Keeper prompt tells the model the retrieved text is authoritative, so contradictions become confident wrong answers. Known open ambiguities in the current rules reference are listed in the gold-data skill (`references/corpus-ambiguities.md`); resolving one is a good corpus edit (then re-check gold data).
- Original wording only in `data/open` (see SKILL.md § 1). Summarise mechanics; do not paraphrase book prose sentence by sentence.

## Lore and setting

- Facts useful at the table (who, where, attitudes, what an investigator can learn and how), not history essays.
- Keep scenario secrets out of `lore`: secret material belongs in the scenario pack's `keeper_notes` or a `scenario` source with `visibility: keeper`.
- Mark simplified or invented history as such in the intro section.

## Checklist before ingesting

- [ ] Front matter complete; `source_type` correct; licence stated.
- [ ] h1 equals the beginning of `source_title` (new documents).
- [ ] No section longer than ~900 characters; no duplicate `###` names under one `##`.
- [ ] No existing heading renamed or moved (or gold data v2 planned).
- [ ] `uv run python -m evals.datasets` passes.
- [ ] After ingest: `keeper.rag.search` finds the new section for 2–3 realistic questions.
