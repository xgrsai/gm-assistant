---
name: coc-keeper-dev
description: Change the GM Assistant Keeper itself — the LangGraph turn pipeline (analyze/retrieve/adjudicate/mechanics/narrate), Call of Cthulhu 7e dice and rules code, Keeper and adjudication prompts, model profiles and experiment arms (Gemini, Ollama, vLLM/LoRA), scenario packs and pregenerated investigators, and the SSE API/frontend contract. Use this whenever the user wants to add or fix a game mechanic (combat, Luck spending, opposed rolls, sanity), tune how the Keeper narrates or when it calls for rolls, add a model or a new arm, write a new scenario, change graph nodes or state, or touch routes/streaming — including Ukrainian requests like «додай механіку», «Хранитель неправильно кидає», «зміни промпт», «додай модель», «новий сценарій», «новий вузол у графі».
---

# Developing the Keeper

The Keeper is the thesis's system under test, so two things matter besides working code: **outcomes stay deterministic** (the model decides *what* to roll and *how* to narrate; Python decides *what happens*), and **experiments stay reproducible** (a campaign or eval run can always say exactly which prompts, models and retrieval produced it). Most rules below protect one of these.

Layout (all under `src/backend` unless noted): `keeper/graph.py` + `keeper/nodes.py` (pipeline), `keeper/game/coc7e/` (dice, rules), `keeper/prompts/*.md.j2`, `keeper/spec.py` (`ExperimentSpec`), `keeper/llm.py` + `config/models.yaml`, `experiments/play.yaml` (app arms), `keeper/scenarios.py` + `data/scenarios/<id>/scenario.yaml` (repo root), `server/` (FastAPI, SSE), `docs/api.md` (contract), `src/frontend/src` (UI).

## Invariants — check every change against these

1. **Code decides outcomes.** Dice, success levels, SAN loss and sheet changes happen in `keeper/game/coc7e/`. Skill values come from the server-side investigator sheet, never from model output. Prompts may *ask* the model to call for a roll; they must never let it invent results. Scenario-defined values (e.g. `sanity_events`) beat the model's guesses.
2. **Prompts are versioned, not edited.** Campaigns freeze `prompt_version` etc. in their spec and runs record prompt hashes. To change behaviour, copy `keeper-v2.md.j2` → `keeper-v3.md.j2`, edit the copy, and point a new or updated arm at it. Editing a template in place silently changes the meaning of old campaigns and eval runs.
3. **Specs are frozen per campaign.** `ExperimentSpec` (generator, aux, prompts, `use_rag`, retrieval, seed) is stored whole with each campaign and hashed, so editing an app arm in `experiments/play.yaml` affects only new campaigns. Eval experiment YAMLs that already have runs are never edited — create a new one.
4. **Player text is data.** Wrap it with `player_turn(...)` (`<player_turn>` tags) for aux calls; never interpolate it into instructions.
5. **The graph topology is shared with evals.** Eval-only behaviour enters through input keys (`forced_hits`, `forced_kind`) consumed and cleared by nodes; shared logic lives in helpers (`search_query`, `adjudicate_turn`) used by both nodes and `evals/harness.py`. Keep that pattern when adding nodes, or the evaluation stops measuring production.
6. **Streaming and resume.** Play runs use `durability="sync"` so a crash never re-rolls dice on Retry. `narrate` has no `RetryPolicy` (tokens are already sent). Checks that must return HTTP status codes (404/409/422) go in FastAPI dependencies, never inside the SSE generator.
7. **Gemini specifics.** `gemini-3.8-flash` rejects thinking level `minimal` (use `low`); history must end with a player turn (no model prefill — `assert_ends_with_human`); don't pass `api_key=None` explicitly; temperature is ignored on ≥3.6 models. The author's key is free tier: avoid bursts of live calls, use fakes in tests.
8. **Copyright.** Scenario packs and prompts in the repo contain only original or openly licensed text.

## Common tasks

- **Add or fix a mechanic:** read `references/coc7e-mechanics.md` (what is implemented, rule sources and confidence, known gaps), change `dice.py`/`rules.py`, add table tests and Hypothesis invariants in `tests/test_dice.py` / `tests/test_rules.py`. If the adjudicator must express something new (e.g. an opposed roll), extend the `Adjudication` model in `nodes.py` (portable JSON-schema types only; clamp numbers in code), bump `adjudicate-vN` (new fields go in a schema subclass registered in `ADJUDICATION_SCHEMAS`), and document new roll-record fields in `docs/api.md` and the frontend `RollCard`.
- **Change Keeper behaviour (narration, when to roll, citations):** new prompt version per invariant 2; for adjudication changes also consider the gold adjudication set (eval-run skill) so the effect can be measured.
- **Add a model or arm:** `references/models-and-arms.md`.
- **New scenario pack or investigator:** `references/scenario-format.md`.
- **Change nodes, state, API or streaming:** `references/graph-and-api.md` (node table, state channels, SSE contract, where the frontend consumes each event).

## Verify

From `src/backend`: `uv run ruff check .` and `uv run pytest` (in-memory Qdrant and fake models, no API calls; includes a subprocess crash-and-retry test). From `src/frontend`: `npm run build` when the contract or UI changed. For behaviour changes, a live check in the app (`uv run uvicorn server.main:app --reload` + `npm run dev`) with a couple of Gemini turns — tell the user it spends quota. Update `docs/api.md` whenever the HTTP/SSE contract changes.
