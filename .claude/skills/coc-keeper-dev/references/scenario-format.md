# Scenario packs

Location: `data/scenarios/<id>/scenario.yaml` (repo root). Loader and schema: `src/backend/keeper/scenarios.py` (`ScenarioPack`, `Pregen`). Example: `data/scenarios/demo-haunt/scenario.yaml`. Only original or openly licensed content — never adapt a published Chaosium scenario into the repo.

## Schema

```yaml
id: demo-haunt                  # must equal the directory name; [a-z0-9-]
game_system: coc7e              # retrieval filters the knowledge base by this
title:   {uk: Тиша на Пекарській, en: Silence on Pekarska Street}
era:     {uk: "Львів, листопад 1927", en: "Lviv, November 1927"}
blurb:   {uk: ..., en: ...}     # shown in the New Campaign dialog
opening: {uk: |, en: |}         # first Keeper message; ends with an open question; written in second person
keeper_notes: |                 # SECRET, English is fine: truth, clues (with the roll that reveals each), threat, pacing, endings
  ...
sanity_events:                  # key -> X/Y notation; the adjudicator may name a key, code uses this value
  gnawed_bones: "0/1D3"
  ghoul_seen: "0/1D6"
investigators:                  # the first one is used by default
  - id: olena-koval
    name: {uk: Олена Коваль, en: Olena Koval}
    occupation: {uk: ..., en: ...}
    age: 29
    characteristics: {STR: 40, CON: 60, SIZ: 50, DEX: 50, APP: 50, INT: 80, POW: 60, EDU: 70}
    luck: 55
    skills: {"Spot Hidden": 70, "Library Use": 60, "Language (Other: German)": 40, ...}
```

Localized fields fall back to `en`, then to the first value. HP, MP, SAN and max SAN are derived by `build_investigator`, not written in the file.

## Writing guidance

- **Keeper notes** are injected into every narration prompt as secret context; the prompt forbids revealing them before they are earned. Structure them as TRUTH / CLUES (numbered, each with the action or roll that reveals it) / THREAT & PACING / ENDINGS — the adjudicator and narrator use clue conditions to decide when a roll is needed.
- **Sanity events:** give every predictable horror a key with a value, so SAN magnitude comes from the author, not the model.
- **Skills:** use exact CoC 7e skill names; they become the only targets the adjudicator may pick (`sheet_names`) and the allowed targets in gold adjudication data. Include `Cthulhu Mythos` (usually 0). Keep base values realistic (Dodge = DEX/2, Language (Own) = EDU).
- **Quick-start characteristic array:** 40, 50, 50, 50, 60, 60, 70, 80.
- **Opening:** both languages, same content, immersive, 2–3 short paragraphs, ending with «Що ви робите?» / "What do you do?".

## After adding or editing

- `GET /api/scenarios` lists a new pack automatically; edits to an already loaded pack need a backend restart (packs are cached per id). Existing campaigns keep the scenario state and investigator they were seeded with.
- Tests use `demo-haunt`; if you change its pregen skill names, the gold adjudication set's `allowed_targets` validation may fail (`uv run python -m evals.datasets`) — treat `demo-haunt` as frozen for evaluation and create a new pack instead of reshaping it.
- Scenario lore that should be retrievable belongs in the knowledge base as `lore` (public) or `scenario` with `visibility: keeper` (not yet wired into prompts) — see the knowledge-base skill.
