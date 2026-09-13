# GM Assistant

An AI Game Master (Keeper) for tabletop role-playing games, built as the practical part of a master's thesis:

> *Метод реалізації AI-агента для домену настільних рольових ігор на основі поєднання RAG та доменного fine-tuning*
> (A method for implementing an AI agent for the tabletop RPG domain by combining RAG and domain fine-tuning).

The target method combines retrieval-augmented generation over a game's rules, lore and scenarios with a LoRA fine-tuned language model. Call of Cthulhu, with the agent acting as the Keeper, is the example system.

## Status

Early prototype. **What works today:**

- A chat UI (React + Vite + Tailwind) with Markdown rendering of Game Master replies.
- A single stateless endpoint, `POST /api/chat/`, that forwards the conversation to Google Gemini with a Game Master system prompt.

**Not implemented yet:**

- Conversation history is kept only in the browser tab and is lost on reload.
- There is no RAG, no fine-tuned model, no dice or game mechanics, and campaigns in the sidebar are placeholders.

The planned architecture (FastAPI + LangGraph, Qdrant RAG, deterministic Call of Cthulhu 7e mechanics, a LoRA-tuned open model, an evaluation harness) and the phased roadmap are in [`docs/final-architecture.md`](docs/final-architecture.md). A review of that plan is in [`docs/critique.md`](docs/critique.md), and the known issues in the current code are listed in [`docs/issues.md`](docs/issues.md).

## Tech stack (current)

- **Frontend:** React 18, Vite 5, Tailwind CSS 3, react-markdown
- **Backend:** Django 5.2, Django REST Framework, `google-genai`
- **Dev environment:** Docker Compose

## Running locally

1. Copy `.env.example` to `.env` and set `GOOGLE_API_KEY`.

2. Start the app with Docker, or run backend and frontend by hand.

   **With Docker:**

   ```bash
   docker compose up --build
   ```

   **Without Docker (PowerShell):**

   ```powershell
   # backend
   cd src/backend
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   $env:GOOGLE_API_KEY = "your-api-key"
   python app/manage.py runserver 8000

   # frontend (second terminal)
   cd src/frontend
   npm install
   npm run dev
   ```

3. Open http://localhost:5173.

Both services bind to `127.0.0.1` only. The API has no authentication and spends your Gemini quota, so do not expose it to a network.

## Game content and copyright

Commercial rulebooks and scenarios (for example Chaosium's Call of Cthulhu books) are copyrighted and must not be committed to this repository.

- Put any such material, and datasets derived from it, under `data/private/`, which is git-ignored.
- Only openly licensed or self-written content belongs in the repo.
- Model weights and adapters (`*.gguf`, `*.safetensors`, `/models/`, `/adapters/`) are git-ignored as well.

## License

The code is released under the [MIT License](LICENSE). The license does not cover third-party game content.
