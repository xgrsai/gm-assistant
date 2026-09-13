# GM Assistant

An AI Game Master (Keeper) for tabletop role-playing games, built as the practical part of a master's thesis:

> *Метод реалізації AI-агента для домену настільних рольових ігор на основі поєднання RAG та доменного fine-tuning*
> (A method for implementing an AI agent for the tabletop RPG domain by combining RAG and domain fine-tuning).

The target method combines retrieval-augmented generation over a game's rules, lore and scenarios with a LoRA fine-tuned language model. Call of Cthulhu 7th edition, with the agent acting as the Keeper, is the example system.

## Status: Phase 1 (MVP)

**Works today:**

- **Persistent campaigns.** History, the investigator sheet and the roll log are stored server-side in LangGraph checkpoints (SQLite) and survive reloads and restarts.
- **Keeper pipeline as an explicit LangGraph graph:** `prepare_turn → adjudicate → [mechanics] → narrate`.
  - `adjudicate`: an auxiliary model decides, as structured output, whether the action needs a roll and which one.
  - `mechanics`: plain Python rolls the dice under Call of Cthulhu 7e rules (success levels, critical/fumble, bonus/penalty dice, pushed rolls, Sanity with INT roll for temporary insanity). The model never decides outcomes.
  - `narrate`: the generator model tells the result, streamed token by token.
- **Streaming over SSE**, with Retry that resumes a failed turn from its last checkpoint without re-rolling dice, and Discard.
- **Reproducibility hooks:** every campaign freezes an `ExperimentSpec` (hashed), prompts are versioned and hashed, and each turn writes a JSONL research record to `artifacts/runs/`.
- A self-written demo scenario, *Тиша на Пекарській / Silence on Pekarska Street* (Lviv, 1927), with a pregenerated investigator, in Ukrainian and English.

**Next phases:** RAG over rules/lore (Qdrant), evaluation harness, LoRA fine-tuning of an open model, and the prompt / RAG / FT / RAG+FT comparison. See [`docs/final-architecture.md`](docs/final-architecture.md) (plan), [`docs/critique.md`](docs/critique.md) (review of the plan) and [`docs/api.md`](docs/api.md) (HTTP/SSE contract).

## Tech stack

- **Backend** (`src/backend`): Python 3.12, FastAPI, LangGraph 1.2, LangChain core components, `langchain-google-genai` (Gemini), SQLite checkpointer, uv.
- **Frontend** (`src/frontend`): React 18, Vite 5, Tailwind CSS 3, react-markdown.

## Running locally (Windows, PowerShell)

Prerequisites: [uv](https://docs.astral.sh/uv/), Node.js 20+.

1. Copy `.env.example` to `.env` in the repo root and set `GOOGLE_API_KEY`.

2. Backend (first terminal):

   ```powershell
   cd src/backend
   uv sync
   uv run uvicorn server.main:app --reload --port 8000
   ```

3. Frontend (second terminal):

   ```powershell
   cd src/frontend
   npm install
   npm run dev
   ```

4. Open http://localhost:5173. Vite proxies `/api` to the backend.

Tests and lint:

```powershell
cd src/backend
uv run pytest
uv run ruff check .
```

Docker Compose (`docker compose up --build`) also works, but on Windows bind mounts are slow; the native setup above is recommended for development.

Both services bind to `127.0.0.1` only. The API has no authentication and spends your Gemini quota, so do not expose it to a network. Free-tier Gemini keys have low per-minute quotas; `GOOGLE_RPM` in `.env` caps the request rate (each turn makes two calls).

## Game content and copyright

Commercial rulebooks and scenarios (for example Chaosium's Call of Cthulhu books) are copyrighted and must not be committed to this repository.

- Put any such material, and datasets derived from it, under `data/private/`, which is git-ignored.
- Only openly licensed or self-written content belongs in the repo (for example `data/scenarios/demo-haunt`).
- Model weights and adapters (`*.gguf`, `*.safetensors`, `/models/`, `/adapters/`) are git-ignored as well.

## License

The code and the demo scenario are released under the [MIT License](LICENSE). The license does not cover third-party game content.
