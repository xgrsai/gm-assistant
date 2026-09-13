# GM Assistant — project notes for Claude

AI Game Master (Keeper) for tabletop RPGs; practical part of the author's master's thesis on combining **RAG** with **domain LoRA fine-tuning**. Example system: Call of Cthulhu 7e. The author writes in Ukrainian — answer in Ukrainian unless asked otherwise.

## Current state (Phase 1 MVP)

- `src/backend` — Python 3.12 + uv. FastAPI (`server/`) + LangGraph (`keeper/`).
  - `keeper/graph.py`: `prepare_turn → adjudicate → [mechanics] → narrate`; `keeper/nodes.py` holds the nodes.
  - `keeper/game/coc7e/`: pure dice/rules code with tests — outcomes are computed here, never by the LLM.
  - `keeper/spec.py` `ExperimentSpec` (frozen per campaign), `experiments/*.yaml` arms, `config/models.yaml` model profiles, `keeper/prompts/*.md.j2` versioned prompts.
  - `server/routes.py`: campaigns CRUD, SSE turns, retry (resume from checkpoint), discard. Contract: `docs/api.md`.
  - Persistence: `AsyncSqliteSaver` (`var/checkpoints.db`) + campaigns table (`var/app.db`). Run log: `artifacts/runs/*.jsonl`.
- `src/frontend` — React 18 + Vite 5 + Tailwind 3; talks to `/api` through the Vite proxy (`src/api/`).
- `data/scenarios/demo-haunt/scenario.yaml` — self-written demo scenario (uk/en).
- Run natively: `cd src/backend; uv run uvicorn server.main:app --reload` and `cd src/frontend; npm run dev`. Tests: `uv run pytest`, lint: `uv run ruff check .`.
- Gemini: `gemini-3.8-flash` works; thinking level `minimal` is rejected (use `low`). The author's key is free tier with a low per-minute quota — avoid bursts of live calls; tests use `tests/fakes.py`.

## Roadmap

Plan: `docs/final-architecture.md`; corrections to apply: `docs/critique.md`. Next: Phase 2 RAG (Qdrant, nomic/bge-m3 embeddings, `analyze`/`retrieve` nodes, `[R#]` citations), Phase 3 eval harness, Phase 4 LoRA fine-tuning, Phase 5 full matrix.

## Rules

- Never commit copyrighted game content (Chaosium rulebooks, purchased scenarios) or anything derived from it; it goes in git-ignored `data/private/`. Same for model weights/adapters.
- Never read or print `.env`; secrets come from environment variables.
- Dice outcomes are computed by code, never by the LLM; skill values come from server state, not from model output.
- Keep experiment runs reproducible: frozen spec + hashes, fixed seeds, no silent changes to prompts or model ids mid-study.
- Every play-graph run uses `durability="sync"`; checks that must return HTTP status codes belong in FastAPI dependencies, not inside the SSE generator.
- Windows 11 host: write text files with explicit `encoding="utf-8"`; line endings are normalized to LF via `.gitattributes`.
- Small, purposeful commits; don't commit unless asked.
