# GM Assistant

An AI Game Master (Keeper) for tabletop role-playing games, built as the practical part of a master's thesis:

> *Метод реалізації AI-агента для домену настільних рольових ігор на основі поєднання RAG та доменного fine-tuning*
> (A method for implementing an AI agent for the tabletop RPG domain by combining RAG and domain fine-tuning).

The target method combines retrieval-augmented generation over a game's rules, lore and scenarios with a LoRA fine-tuned language model. Call of Cthulhu 7th edition, with the agent acting as the Keeper, is the example system.

## Status: Phase 2 (RAG)

**Works today:**

- **Persistent campaigns.** History, the investigator sheet, the roll log and citations are stored server-side in LangGraph checkpoints (SQLite) and survive reloads and restarts.
- **Keeper pipeline as an explicit LangGraph graph:**
  `prepare_turn → [analyze → retrieve] → adjudicate → [mechanics] → narrate`.
  - `analyze` (RAG campaigns): an auxiliary model rewrites the player's action, in any language, as a standalone English search query.
  - `retrieve` (RAG campaigns): exact cosine search in Qdrant over the knowledge base, with per-source-type quotas (rules, lore) and a similarity threshold.
  - `adjudicate`: the auxiliary model decides, as structured output, whether the action needs a roll and which one, using the retrieved rules.
  - `mechanics`: plain Python rolls the dice under Call of Cthulhu 7e rules (success levels, critical/fumble, bonus/penalty dice, pushed rolls, Sanity with INT roll for temporary insanity). The model never decides outcomes.
  - `narrate`: the generator model tells the result, streamed token by token. In RAG campaigns it cites passages as `[R1]`, `[R2]`; the server checks every marker against what was actually retrieved and reports invalid ones.
- **Two modes per campaign** (experiment arms): with the knowledge base (`play-gemini-rag`, default) or without it (`play-gemini`). The mode is frozen when the campaign is created.
- **Knowledge base** (`data/open/coc7e`): an original Call of Cthulhu 7e rules reference written for this project, plus setting notes for Lviv in 1927. Markdown with YAML front matter, chunked by headings and then into 1000-character chunks with 200 characters of overlap, embedded with `nomic-embed-text` (Ollama).
- **Streaming over SSE**, with Retry that resumes a failed turn from its last checkpoint without re-rolling dice, and Discard.
- **Reproducibility hooks:** every campaign freezes an `ExperimentSpec` (hashed, including retrieval settings), prompts are versioned and hashed, and each turn writes a JSONL research record to `artifacts/runs/` (search query, retrieved chunk ids and scores, citations, rolls, usage, timings).
- A self-written demo scenario, *Тиша на Пекарській / Silence on Pekarska Street* (Lviv, 1927), with a pregenerated investigator, in Ukrainian and English.

**Next phases:** evaluation harness (gold rules questions, retrieval metrics, conflict probes), LoRA fine-tuning of an open model, and the prompt / RAG / FT / RAG+FT comparison. See [`docs/final-architecture.md`](docs/final-architecture.md) (plan), [`docs/critique.md`](docs/critique.md) (review of the plan) and [`docs/api.md`](docs/api.md) (HTTP/SSE contract).

## Tech stack

- **Backend** (`src/backend`): Python 3.12, FastAPI, LangGraph 1.2, LangChain core components, `langchain-google-genai` (Gemini), Qdrant, Ollama embeddings, SQLite checkpointer, uv.
- **Frontend** (`src/frontend`): React 18, Vite 5, Tailwind CSS 3, react-markdown.

## Running locally (Windows, PowerShell)

Prerequisites: [uv](https://docs.astral.sh/uv/), Node.js 20+, Docker Desktop (for Qdrant), [Ollama](https://ollama.com/).

1. Copy `.env.example` to `.env` in the repo root and set `GOOGLE_API_KEY`.

2. Knowledge base (once, and again after editing `data/open`):

   ```powershell
   docker compose up -d qdrant
   ollama pull nomic-embed-text
   cd src/backend
   uv sync
   uv run python -m keeper.rag.ingest ../../data/open/coc7e --recreate
   ```

   Check retrieval from the command line: `uv run python -m keeper.rag.search "how do bonus dice work"`. The Qdrant dashboard is at http://localhost:6333/dashboard.

3. Backend (first terminal, in `src/backend`):

   ```powershell
   uv run uvicorn server.main:app --reload --port 8000
   ```

   Startup fails with a clear message if Qdrant, the collection or the embedding model is missing. To run without the knowledge base, set `DEFAULT_ARM=play-gemini` and remove the RAG arm from `experiments/play.yaml`.

4. Frontend (second terminal):

   ```powershell
   cd src/frontend
   npm install
   npm run dev
   ```

5. Open http://localhost:5173. Vite proxies `/api` to the backend.

Tests and lint (no Qdrant, Ollama or Gemini needed; they use in-memory Qdrant and fake models):

```powershell
cd src/backend
uv run pytest
uv run ruff check .
```

Docker Compose (`docker compose up --build`) also works: the api container reaches Qdrant as a service and Ollama on the host through `host.docker.internal`. Ingest from inside the container with `docker compose exec api python -m keeper.rag.ingest /data/open/coc7e --recreate`. On Windows bind mounts are slow, so the native setup above is recommended for development.

All services bind to `127.0.0.1` only. The API has no authentication and spends your Gemini quota, so do not expose it to a network. Free-tier Gemini keys have low per-minute quotas; `GOOGLE_RPM` in `.env` caps the request rate (a RAG turn makes three calls).

## Game content and copyright

Commercial rulebooks and scenarios (for example Chaosium's Call of Cthulhu books) are copyrighted and must not be committed to this repository.

- Put any such material, and datasets derived from it, under `data/private/`, which is git-ignored. It can be ingested alongside the open corpus: `uv run python -m keeper.rag.ingest ../../data/open/coc7e ../../data/private/coc7e --recreate`.
- Only openly licensed or self-written content belongs in the repo (for example `data/open/coc7e` and `data/scenarios/demo-haunt`).
- Model weights and adapters (`*.gguf`, `*.safetensors`, `/models/`, `/adapters/`) are git-ignored as well.

## License

The code, the demo scenario and the texts in `data/open` are released under the [MIT License](LICENSE). The license does not cover third-party game content.
