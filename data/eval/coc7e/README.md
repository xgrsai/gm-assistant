# Gold evaluation datasets — Call of Cthulhu 7e (v1)

> **STATUS: DRAFT v1 — authored with AI assistance; every item must be reviewed by the author before the set is frozen for thesis experiments (freezing = the file hashes recorded in the eval run config).**

These files are the measuring instrument for the prompt-only vs RAG (and later fine-tuned) Keeper comparison.
Schemas, loading, hashing and validation live in `src/backend/evals/datasets.py` (pydantic models `RulesQAItem`,
`ConflictProbe`, `AdjudicationItem`, `ExpectedAdjudication`). Validate with:

```
cd src/backend
uv run python -m evals.datasets
```

All files are JSON Lines, UTF-8, one object per line.

## Files

### `rules_qa.v1.jsonl` — rules QA and retrieval (100 items)

Measures retrieval quality (Recall@k / MRR against gold sections) and answer correctness (key-fact coverage),
plus abstention on questions the knowledge base does not answer.

| field | meaning |
|---|---|
| `category` | topic (see counts below) |
| `query_type` | `lookup` (wording overlaps the section), `paraphrase` (natural player wording, little keyword overlap), `multi_hop` (needs 2+ sections, all listed), `numeric` (asks for a number, threshold or formula) |
| `question_en` / `question_uk` | the same question; the Ukrainian one is written as a player would type it in chat |
| `answerable` | `false` for the 15 questions the corpus does not answer |
| `reference_answer` | English, 1-3 sentences; for unanswerable items it describes the ideal behaviour |
| `key_facts` | atomic English facts a correct answer must contain (empty when unanswerable) |
| `gold_sections` | `{source, section}` pairs holding the answer (empty when unanswerable) |

- 85 answerable + 15 unanswerable. Query types: paraphrase 35, lookup 26, numeric 26, multi_hop 13
  (answerable only: paraphrase 31, lookup 22, numeric 19, multi_hop 13).
- Categories: skills 9, setting 9, sanity 7, characteristics 6, combat 6, insanity 6, wounds_healing 5,
  bonus_penalty 4, firearms 4, keeper_principles 4, luck 4, pushing 4, development 3, difficulty 3, opposed 3,
  skill_rolls 3, success_levels 3, idea_know 2, unanswerable 15.
- Every section of the rules reference (all 40, including the untitled introduction) and all 9 setting-note
  sections appear in at least one `gold_sections` list.
- Non-multi-hop items list the single section that states the answer. The corpus repeats a few facts in more than
  one section (e.g. starting SAN = POW, Idea roll = INT, "a fumble cannot be pushed"); questions were worded towards
  the section with the specific detail, but a retriever may legitimately surface the other one.
- Unanswerable items were checked by searching the corpus text for their key terms (spells, .38 revolver, tomes,
  age modifiers, Credit Rating levels, Pulp talents, automatic fire, occupation points, armour, drowning,
  explosives, the mayor of Lviv, ghoul statistics, the base value of Spot Hidden).

**Retrieval gold labels are sections, not chunk ids.** A section is the front-matter `source_title` plus the
heading path produced by `keeper.rag.corpus` (rules paths have no H1 prefix, lore paths start with
`Lviv in 1927: Setting Notes > `). Chunk ids change with chunk size and overlap; sections survive re-chunking, so the
same gold set serves the chunking ablations.

### `conflict_probes.v1.jsonl` — knowledge-conflict probes (24 items)

Measures context adherence: the passage of `gold_section` is edited with `find` -> `replace` (first occurrence; the
validator requires exactly one) and shown to the model as the retrieved rule. An answer matching `perturbed_fact`
follows the context; one matching `original_fact` follows prior knowledge.

- Exactly 4 items per cell of perturbation × familiarity:

  | | well_known | obscure |
  |---|---|---|
  | numeric (number, fraction, threshold) | 4 | 4 |
  | negation (inverted rule) | 4 | 4 |
  | entity (swapped skill / characteristic / term) | 4 | 4 |

- Every probe section is a single chunk; no section is used more than twice. Edits keep the passage internally
  consistent (where the same number appears twice in a passage, e.g. the fumble range or the HP example, one
  `find` span covers both).
- `familiarity` is a judgement of how likely a CoC-trained model knows the unedited rule by heart.

### `adjudication.v1.jsonl` — roll adjudication and injection robustness (100 items, 40 groups)

Measures whether the adjudicator makes the right call (needs a roll? which type? which sheet name?) and whether
rhetorical manipulation in the player's text flips that call (injection flip rate).

| field | meaning |
|---|---|
| `group` | one clean item plus its injection variants |
| `variant` / `injection_type` | `clean`/`none`, or `injection` with `claimed_success`, `claimed_roll`, `false_authority`, `instruction`, `emotional_pressure` |
| `scene` | Ukrainian, the Keeper's last narration in the demo scenario `demo-haunt` (never reveals unfound secrets) |
| `player_text` | Ukrainian, the player's turn |
| `expected.kind` | `in_fiction`, `rules_question` or `meta` |
| `expected.needs_roll` / `roll_type` | `skill`, `characteristic`, `luck`, `sanity` or `none` |
| `expected.allowed_targets` | every exact sheet name a reasonable Keeper could pick (validated against the pregenerated investigator) |
| `expected.allowed_difficulties` | empty = any difficulty accepted (all v1 items are empty) |
| `expected.notes` | English rationale (ignored when comparing variants) |

- 40 clean items: 17 skill rolls (Spot Hidden, Library Use, Listen, Psychology, Persuade, Fast Talk, Charm,
  Locksmith, Stealth, Climb, First Aid, Occult, History/Archaeology, Language (Other: German),
  Art/Craft (Photography), Dodge, Fighting (Brawl)), 3 characteristic rolls (STR, CON, INT Idea roll), 1 Luck,
  2 Sanity, 12 in-fiction no-roll, 3 rules questions, 2 meta.
- 60 injection variants: 20 groups with two variants and 20 with one. Injection types: false_authority 13,
  claimed_success 12, instruction 12, emotional_pressure 12, claimed_roll 11. 23 variants sit on no-roll items and
  mostly try to force a roll or an outcome.
- An injection variant keeps the scene and the underlying action; its `expected` is identical to the clean item's.
  Only clean calls judged unambiguous under "Skill Rolls > When to Roll" and the scenario notes were kept.

## License

MIT, like the repository. Every item is derived only from the project's own corpus
(`data/open/coc7e/*.md`, original text) and the original demo scenario (`data/scenarios/demo-haunt`); nothing is
taken from Chaosium books or published scenarios.

## Review checklist (before freezing)

- [ ] Rules QA: each `key_fact` is literally supported by its gold section; `reference_answer` says nothing the
      corpus does not.
- [ ] Rules QA: `gold_sections` is complete for multi-hop items and minimal for the others; the Ukrainian question
      asks the same thing as the English one and reads naturally.
- [ ] Rules QA: each unanswerable question is really not answered (not even partially) by the corpus.
- [ ] Conflict probes: the edited passage reads naturally, stays internally consistent, and the question is
      answerable from it alone; `familiarity` labels look right.
- [ ] Adjudication: the clean call is unambiguous (drop the group if "roll" and "no roll" are both defensible);
      `allowed_targets` lists every reasonable skill and nothing more; scenes reveal no unfound secrets.
- [ ] Adjudication: every injection variant keeps the same action, so the clean `expected` still holds.
- [ ] Re-run the validator, then record the file hashes it prints in the eval run config; any later edit means a
      new version (`*.v2.jsonl`), never an in-place change.
