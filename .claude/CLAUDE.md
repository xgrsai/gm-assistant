# GM Assistant — project notes for Claude

AI Game Master (Keeper) for tabletop RPGs; practical part of the author's master's thesis on combining **RAG** with **domain LoRA fine-tuning**. Example system: Call of Cthulhu 7e. The author writes in Ukrainian — answer in Ukrainian unless asked otherwise.

## Current state (Phase 3: evaluation harness)

- `src/backend` — Python 3.12 + uv. FastAPI (`server/`) + LangGraph (`keeper/`).
  - `keeper/graph.py`: `prepare_turn → [analyze → retrieve] → adjudicate → [mechanics] → narrate`. `prepare_turn` routes with `Command` by `spec.use_rag` (routers don't get the run context).
  - `keeper/game/coc7e/`: pure dice/rules code with tests — outcomes are computed here, never by the LLM.
  - `keeper/rag/`: `corpus.py` (front matter, heading + 1000/200 chunking, stable uuid5 ids), `embeddings.py` (task prefixes by composition — never subclass OllamaEmbeddings), `retriever.py` (exact search, per-type quotas, `min_score`), `ingest.py` / `search.py` CLIs. `keeper/citations.py` validates `[R#]` markers.
  - `keeper/spec.py` `ExperimentSpec` + `RetrievalSpec` (frozen per campaign), `experiments/play.yaml` arms (`play-gemini-rag` default, `play-gemini`), `config/models.yaml` and `config/embedders.yaml` profiles, `keeper/prompts/*.md.j2` versioned prompts (v1 kept for old campaigns; add a new version instead of editing one in place).
  - `server/routes.py`: meta (arms), campaigns CRUD with optional arm, SSE turns, retry (resume from checkpoint), discard. Contract: `docs/api.md`.
  - Persistence: `AsyncSqliteSaver` (`var/checkpoints.db`) + campaigns table (`var/app.db`). Run log: `artifacts/runs/*.jsonl`. Chunk dumps: `artifacts/corpus/<collection>/chunks.jsonl`.
- Knowledge base: `data/open/coc7e/*.md` (original rules reference + Lviv 1927 lore), collection `coc7e__nomic15__c1000o200` in Qdrant (`docker compose up -d qdrant`), embeddings `nomic-embed-text` via the native Ollama app. Re-ingest with `uv run python -m keeper.rag.ingest ../../data/open/coc7e --recreate` after editing the corpus. Startup verifies the collection and embedder.
- `src/backend/evals` — evaluation harness (methodology: `docs/evaluation.md`).
  - `datasets.py` schemas + validator for `data/eval/coc7e/*.v1.jsonl` (gold retrieval labels are SECTIONS, not chunk ids); `experiment.py` eval YAMLs in `evals/experiments/`; `harness.py` runs the production graph/nodes (`forced_hits`, `forced_kind` inputs; shared helpers `search_query`/`adjudicate_turn` in `keeper/nodes.py`); `judge.py` + `evals/prompts/`; `metrics.py` report; `stats.py` (no scipy); `calibrate.py` human-label κ; `review.py` + `review_page.html` build the gold-data review page (`artifacts/review/`, verdicts exported as JSON, `--summary`).
  - Frozen RAG contexts (critique C7) cached in `artifacts/eval/contexts/`, keyed by aux + analyze prompt + retrieval spec + query mode (`analyze` uses the aux LLM, `question_en` uses none). Runs in `artifacts/eval/runs/<name>/` are resumable and bound to a fingerprint that ignores judge and cell selection (limits, tasks, query modes, languages, conditions).
  - Gold data v1 is AI-drafted: the author must review it before thesis runs.
- `src/frontend` — React 18 + Vite 5 + Tailwind 3; `/api` via the Vite proxy (`src/api/`); citations rendered by a remark plugin (`src/markdown/citations.js`, `components/Citations.jsx`).
- Run natively: `cd src/backend; uv run uvicorn server.main:app --reload` and `cd src/frontend; npm run dev`. Tests: `uv run pytest` (in-memory Qdrant + fake models), lint: `uv run ruff check .`.
- Gemini: `gemini-3.8-flash` works; thinking level `minimal` is rejected (use `low`). The author's key is free tier with a low per-minute quota and occasional 503 "high demand" — avoid bursts of live calls.
- Hardware: GTX 1660 Ti (6 GB) — too small for QLoRA on Gemma 4 E4B; plan Phase 4 training on Colab/Kaggle.

## Roadmap

Plan: `docs/final-architecture.md`; corrections to apply: `docs/critique.md`. Phase 3 harness is built; open items: human review of gold data, full `phase3-gemini` run (≈1000 Gemini calls, exceeds the free daily quota — spread over days or use a paid key), judge calibration, adopting the calibrated `min_score` as a new arm. Next: Phase 4 LoRA fine-tuning (Colab/Kaggle), Phase 5 full matrix and ablations (bge-m3, hybrid BM25, chunk sizes).

## Rules

- Never commit copyrighted game content (Chaosium rulebooks, purchased scenarios) or anything derived from it; it goes in git-ignored `data/private/`. Same for model weights/adapters.
- Never read or print `.env`; secrets come from environment variables.
- Dice outcomes are computed by code, never by the LLM; skill values come from server state, not from model output.
- Keep experiment runs reproducible: frozen spec + hashes, fixed seeds, no silent changes to prompts, model ids or collections mid-study.
- Every play-graph run uses `durability="sync"`; checks that must return HTTP status codes belong in FastAPI dependencies, not inside the SSE generator.
- Windows 11 host: write text files with explicit `encoding="utf-8"`; line endings are normalized to LF via `.gitattributes`. Don't put `\uXXXX` escapes in files written with the Write tool (they become literal characters); use `chr(0x...)`.
- Don't run large live eval runs without asking: they spend the author's Gemini quota.
- Small, purposeful commits; don't commit unless asked.
