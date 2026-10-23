# Training scenarios (Phase 4 SFT)

One directory per scenario: `data/sft/coc7e/scenarios/<id>/scenario.yaml`, in the same `ScenarioPack` schema as the app's
scenarios (see `.claude/skills/coc-keeper-dev/references/scenario-format.md`): `id` equal to the directory name,
localized `title` / `era` / `blurb` / `opening`, secret `keeper_notes`, `sanity_events`, and `investigators` (the first
pregen is used unless a seed names another).

These packs are loaded by `sft.build` through a separate `ScenarioRepo` root, so they never appear in the app's
scenario list. They exist only to give training examples a scene, an investigator sheet and sanity magnitudes.

Rules:

- Original text only (no published Chaosium scenarios or anything adapted from them).
- `demo-haunt` is the evaluation scenario: never copy it, quote it or use it here; the builder and the
  decontaminator refuse it.
- Skill names in seeds must match the sheet exactly (`"Spot Hidden"`, not `"spot hidden"`); characteristics are
  `STR CON SIZ DEX APP INT POW EDU`.
- A `sanity_event` named in a seed must be a key of the pack's `sanity_events`.
- A seed's investigator is always the pack's fresh pregen sheet (full SAN and HP, unspent Luck). The seed `history`
  is context only and must not describe mechanical consequences (Sanity or HP lost, wounds, spent Luck) that the sheet
  in the prompt does not show.
- Texts (`player_text`, history turns) must be NFC-normalised and stripped of leading/trailing whitespace; the builder
  rejects anything else instead of fixing it silently.
- In `distractor_only` seeds list in `kb.exclude_sections` every corpus section that could answer the question; those
  sections and their siblings (same parent heading path) are never shown as distractors, so "no passage answers this"
  is true. In `oracle` seeds the oracle sections and their siblings are excluded automatically.

Seeds are written to `data/sft/coc7e/seeds.v1.jsonl` (schema: `src/backend/sft/schemas.py`, `SftSeed`) and turned into
frozen inputs with `cd src/backend && uv run python -m sft.build inputs --write`.
