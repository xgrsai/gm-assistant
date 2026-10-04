# Gold data review checklist

Use per item; record findings as *wrong label*, *ambiguous → fix or drop*, *wording*, or *fine*.

## Rules QA
- [ ] Every `key_fact` is literally supported by the text of its `gold_sections` (open the chunk dump, don't rely on memory).
- [ ] `reference_answer` says nothing the corpus does not say.
- [ ] Multi-hop: `gold_sections` lists every section needed; others: the single primary section (note duplicates, see corpus-ambiguities #16).
- [ ] `question_uk` asks the same thing as `question_en` and reads like a real player's chat message.
- [ ] `query_type` matches (paraphrase really avoids the section's keywords).
- [ ] Unanswerable: the corpus truly does not answer it, not even partially (search for the key terms); a knowledgeable model answering from memory would be *presenting unofficial specifics*, which is what the item tests.
- [ ] Not built on a known ambiguity (corpus-ambiguities.md) unless intentionally.

## Conflict probes
- [ ] The edited passage reads naturally and stays internally consistent (all mentions of the changed value edited).
- [ ] The question is answerable from the edited passage alone and targets exactly the edited detail.
- [ ] `original_fact` / `perturbed_fact` are short, mutually exclusive statements a judge can tell apart.
- [ ] The perturbed rule is plausible (not absurd), so following it is a real test of context priority.
- [ ] `familiarity` label is defensible; balance stays 4 per cell.

## Adjudication
- [ ] The clean call is unambiguous under "Skill Rolls > When to Roll" and the scenario notes; otherwise widen targets, rephrase or drop the group.
- [ ] `allowed_targets` lists every reasonable sheet name and nothing unreasonable.
- [ ] The scene reveals no secret the investigator has not found yet.
- [ ] Each injection variant keeps the same scene and underlying action, so the clean label still holds.
- [ ] Injection types are varied, including attempts to force a roll on no-roll turns.

## After review
- [ ] `uv run python -m evals.datasets` exits 0.
- [ ] README counts updated; status line changed from DRAFT to reviewed (date, reviewer).
- [ ] Hashes recorded (the validator prints them); later changes go to a new version.
