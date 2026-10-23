# SFT data for the narrator LoRA (Phase 4)

Pipeline, all commands from `src/backend`. Every file is JSONL (UTF-8, LF); schemas are in `sft/schemas.py`.

| Step | File | Made by |
|---|---|---|
| 1. seeds | `seeds.v1.jsonl` (`SftSeed`) | written by hand / by Claude; training scenarios in `scenarios/` (see its README) |
| 2. inputs | `inputs.v1.jsonl` (`SftInput`) | `uv run python -m sft.build inputs --write`: code-rolled dice, KB chunks (oracle + distractors, train sections only) and the chat rendered with `narrate_messages`. Frozen: the target must agree with these rolls. |
| 3. targets | `targets.v1.jsonl` (`SftTarget`) | the Keeper replies, one per input, with the `input_hash` they were written against (required) |
| 4. review | `review-v1.csv` (committed) -> `reviews.v1.jsonl` (`SftReview`) | see below |
| 5. build | `artifacts/sft/coc7e-v1/{train,val}.jsonl` + `manifest.json` (git-ignored) | `uv run python -m sft.build build --version v1` |

`build --lint-only` runs everything except writing (inputs without a target yet are fine); use it while writing targets.
`build` refuses missing targets unless `--allow-missing` (the manifest then says `partial: true` and lists the ids),
and targets without an `input_hash` unless `--allow-no-input-hash`.

A row is `{"messages": [system, ...history, user, assistant]}`; the last message is the target (or the reviewer's edit;
a rejected example is dropped).

## Review

`review-export --sample 100` writes `artifacts/sft/coc7e-v1/review-sample.csv`: a seeded sample stratified by kind x KB
mode with the scene, player text, rolls, passages, the target, its lint status, a `target_hash` (`h:<16 hex>`; the
letter prefix keeps Excel from turning it into a number) and the columns `status`, `current_verdict`, `current_edit`.
The file is `;`-separated UTF-8 with a BOM, so Excel in a Ukrainian / European locale opens it in columns by
double-click (in an English locale use Data -> From Text/CSV, delimiter `;`, encoding UTF-8). Fill `verdict`
(`accept` / `edit` / `reject`), `edited_text` (only for `edit`) and `note`; leave `target_hash` alone. Save it as CSV
(use **"CSV UTF-8"**; a Windows-1251 file is refused with a hint) at `data/sft/coc7e/review-v1.csv`, then
`review-export --review data/sft/coc7e/review-v1.csv` merges the verdicts into `reviews.v1.jsonl`. An empty `verdict`
means not reviewed: the target is used as is.

`reviews.v1.jsonl` is a history: one verdict per (id, `target_hash`), where the hash identifies the target text the
reviewer saw. Only the verdict for the CURRENT text of a target applies. `--review` refuses a CSV row whose hash is not
the current text (the message names the CSV line: the target was rewritten after the export; re-export), and merges
the rest by (id, hash), so an old CSV can never overwrite a verdict for newer text.

If a target is rewritten after it was reviewed (e.g. rejected, then rewritten), the build does not fail: that example is
built as unreviewed, with a warning and `reviews.rewritten` in the manifest. `review-export` offers such rows first
(`status` = "rewritten after review"); `--sample-stale` offers only them. Rows that already have a verdict for their
current text are left out unless `--include-reviewed` (then `current_verdict` / `current_edit` show it).

## Ids and the val split

Val is 10% per `kind`, picked by a hash of the **group key** = the id without a trailing `-uk` / `-en`. Name
translation pairs `<base>-uk` / `<base>-en` (the same scene in both languages) so that they land on the same side and
val does not contain a translation of a training example. Ids without such a suffix are their own group.

## Lints (the build fails and lists every failing id with the reason)

`gate` (NFC, no outer whitespace, role order), `input` (stale input: hash, prompt or corpus changed), `stale` (the
target's `input_hash` differs from its input), `no-input-hash`, `number`, `dice-line`, `outcome`,
`citation`, `citation-placement`, `ruling`, `language`, `length`, `tokens`, `decontam:*`. Details and thresholds: the
docstring of `src/backend/sft/dataset.py`. In short:

- **Numbers.** Every number in the reply must occur in the system prompt from SCENE OPENING on (opening, notes, sheet,
  MECHANICS), in an ORACLE passage (never in a distractor), in the player's text or the history; the decade of the
  era year ("1920-ті") is fine. Whitelist: `100`. On the `🎲` line only the d100 value(s) (and the SAN lost for a
  sanity roll) may appear. Write small counts in words or leave them out.
- **Rolls.** The `🎲` line appears exactly once if the example has rolls, and never otherwise. It must state the rolled
  outcome in words, as the record says: «успіх» / "success" for a passed roll, «провал» / «невдача» / «не вдалося» /
  "failure" for a failed one, and not the opposite. Negations are read for what they mean: «без успіху», «успіху
  немає», «майже успіх», "no success" are failures; «провалу не сталося», «уникнув провалу», "not a failure" are not.
  A level named directly before «успіх» («звичайний», «складний», «екстремальний», «критичний успіх»; regular / hard /
  extreme / critical success) must lie between the required level of the check and the rolled level; level `critical`
  needs «критичний успіх», level `extreme` on a check that was not extreme needs «екстремальний успіх», level `fumble`
  needs «фатальний провал» / "fumble" (and none of them may be used otherwise). For a failed roll a level at or below
  the rolled one explains the failure («звичайний успіх, але для складного кидка замало — провал»). Difficulty phrases
  («складна перевірка», "hard check") are fine. A SAN loss stated right after «втрата» / "lost" must equal the record's.
- **`[R#]`** (keeper-v2 rule 7: only out of character): only in RAG examples, only markers that are in the prompt.
  `distractor_only` / `none`: none at all. `oracle`: never a distractor, and by kind - `rules_question`: cite an oracle
  passage (required); `meta`: allowed, not required; `in_fiction` with a roll: cite an oracle passage on the `🎲` line
  only (required), never in the narration; `in_fiction` without a roll: no markers (there is no out-of-character place).
- **"Keeper ruling"** (or «рішення Хранителя», exactly in this nominative form, with a capital «Хранителя») is required if and only if RAG is on,
  the turn is a rules question and the KB does not answer (`distractor_only` / `none`). It is forbidden in every
  other example (the KB answers; in-character and meta turns; no-RAG prompts, where rule 6 is not even rendered).
- Ukrainian replies are at least 70% Cyrillic letters; narration is 70-220 words for `in_fiction` (the prompt asks for
  80-200), 25-240 for `rules_question`, 12-160 for `meta`; `manifest.json` records the length distribution.
- The estimated chat length (`CHARS_PER_TOKEN`) is at most 7000 tokens; training `--max-len` is 8192 with the exact
  tokenizer, and the Colab length gate is the final check. The manifest records the estimate's max and p95.

### Design effect to report in the thesis

The number lint is deliberately strict for rules questions the KB does not answer: the fine-tuned model is trained NOT to
state rule numbers without a passage that gives them. This can lower the FT no-RAG scores on numeric rules questions
(thresholds, dice, percentages) compared with a base model that answers from memory. Report it as a design effect of
the training data (it favours "Keeper ruling" over a confident guess), not as a bug.

`manifest.json` has `files.<name>.sha256` (what `training/train_unsloth.py` verifies) plus the prompt hash (`keeper-v2`),
split file hash, gold-file hashes, git sha and `git_dirty`, source file hashes, counts by kind / RAG / language / KB
mode, review counts, `partial` + `missing_targets`, the narration-length distribution and the token estimate.

## Review of v1 (2026-10-23)

- Targets were authored by Claude agents from the frozen briefs; every target passes the lints above.
- **Review:** all 617 examples were reviewed by four further Claude agents that did not author them (one per
  scenario), against the keeper-v2 rules, the scenario keeper notes, the rolls and the ORACLE / DISTRACTOR passages.
  Result: 574 accept, 43 edit (language, small factual slips against the notes, the Keeper deciding what an
  investigator says or feels), 0 reject. Verdicts, edited texts and reasons: `review-v1.csv`; applied verdicts:
  `reviews.v1.jsonl` (every note starts with "LLM review (Claude)"). This is an LLM review, not a human one; the
  author approves the list of edits — record the approval here.
- **Mechanical post-review fixes** (carried over with the verdicts, noted in each affected row): the 🎲 line moved
  before the closing question in 132 targets (keeper-v2 FORMAT; now enforced by the `dice-line` lint), «кастелянка»
  → «доглядачка» (seeds and targets), one spelling of the NPC Юзеф Павлак, one Ukrainian label per skill on the
  dice line.
- Known residual variation the lints cannot see: small invented details that differ between examples of the same
  scenario (where a key hangs, what a label says), and two names for indefinite insanity («безстрокове» /
  «тривале»).
