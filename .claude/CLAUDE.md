# GM Assistant — project notes for Claude

AI Game Master (Keeper) for tabletop RPGs; practical part of the author's master's thesis on combining **RAG** with **domain LoRA fine-tuning**. Example system: Call of Cthulhu 7e. The author writes in Ukrainian — answer in Ukrainian unless asked otherwise.

## Current state (prototype)

- `src/backend/app` — Django 5.2 + DRF. One stateless endpoint `POST /api/chat/` (`core/views.py`) → `core/services.py` calls Gemini via `google-genai`; system prompt in `core/prompts.py`.
- `src/frontend` — React 18 + Vite 5 + Tailwind 3. API client in `src/lib/api.js`; history lives only in React state.
- Dev runs through `docker compose up --build` (backend :8000, frontend :5173, both bound to 127.0.0.1). No local Python venv is set up on this machine.

## Planned direction (decided, not yet built)

Rewrite the backend as **FastAPI + LangGraph `StateGraph`** (LangChain only for model/embedding/splitter components), server-owned history in LangGraph checkpoints (SQLite first), SSE streaming, deterministic CoC 7e dice in Python, Qdrant RAG, experiment spec for {prompt-only, RAG, FT, RAG+FT}. Full plan: `docs/final-architecture.md`; corrections to it: `docs/critique.md` (apply them — e.g. `durability="sync"`, retry policy, SFT truncation gate); known issues in the old code: `docs/issues.md`. Do not invest in fixing Django-specific issues — that backend will be deleted.

## Rules

- Never commit copyrighted game content (Chaosium rulebooks, purchased scenarios) or anything derived from it; it goes in git-ignored `data/private/`. Same for model weights/adapters.
- Never read or print `.env`; secrets come from environment variables.
- Dice outcomes are computed by code, never by the LLM; skill values come from server state, not from model output.
- Keep experiment runs reproducible: frozen spec + hashes, fixed seeds, no silent changes to prompts or model ids mid-study.
- Windows 11 host: write text files with explicit `encoding="utf-8"`; line endings are normalized to LF via `.gitattributes`.
- Small, purposeful commits; don't commit unless asked.
