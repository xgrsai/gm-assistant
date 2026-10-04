# Model profiles and experiment arms

## Model profile (`src/backend/config/models.yaml`, class `ModelProfile` in `keeper/llm.py`)

```yaml
gemini-flash:
  provider: google_genai      # google_genai | ollama | vllm
  model: gemini-3.8-flash
  thinking_level: low         # Gemini 3+: low | medium | high ("minimal" is rejected by 3.8-flash)
  max_tokens: 4096            # Gemini thinking tokens count toward output — keep generous
  context_window: 1000000     # used to budget history trimming
  # optional: temperature (ignored by Gemini >= 3.6), num_ctx (Ollama), finetuned, base, quant
```

Current profiles: `gemini-flash` (generator), `gemini-aux` (analyze/adjudicate), `gemini-judge` (evals), `qwen25-3b-ollama` (local, pulled), `gemma4-base-ollama` (not pulled).

`ModelFactory` builds each `(profile, seed)` model once, shares one Gemini rate limiter (`GOOGLE_RPM`, prefilled so the first turn doesn't stall), and makes structured calls with `with_structured_output(schema, method="json_schema")` tagged `nostream` (they never leak into the token stream).

Adding a provider (e.g. Anthropic, which the author decided not to add for now): new `provider` literal, a `case` in `_build` using the official LangChain integration, key handling in `server/settings.py`, fail-fast key check in `server/main.py::build_deps` and `evals/run.py::build_eval_deps`, profile entries, tests with fakes. Ask before adding a paid provider.

Local models: Ollama runs as the native Windows app (`OLLAMA_BASE_URL`, default `http://localhost:11434`). Pulling a model is a multi-GB download — ask first. The GPU is a GTX 1660 Ti (6 GB): 3–4B models are practical for inference; LoRA training belongs on Colab/Kaggle (Phase 4). vLLM (`provider: vllm`, OpenAI-compatible, LoRA adapter selected by model name) needs Linux/WSL2 + NVIDIA.

## Experiment spec (`keeper/spec.py`)

`ExperimentSpec` fields: `arm`, `generator`, `aux`, `adjudicator` (aux|generator), `use_rag`, `retrieval` (`RetrievalSpec`: collection, embedder, quotas, exact, min_score), `prompt_version`, `adjudicate_prompt_version`, `analyze_prompt_version`, `seed`, `history_max_tokens`. Frozen and hashed (`spec_hash`).

## App arms (`src/backend/experiments/play.yaml`)

```yaml
fixed: {aux: gemini-aux, adjudicator: aux, prompt_version: keeper-v2, adjudicate_prompt_version: adjudicate-v3,
        analyze_prompt_version: analyze-v1, history_max_tokens: 6000, retrieval: {...}}
arms:
  - {arm: play-gemini-rag, generator: gemini-flash, use_rag: true}
  - {arm: play-gemini, generator: gemini-flash, use_rag: false}
```
- `DEFAULT_ARM` (env) chooses the default; the UI lists all arms from `/api/meta` and the user picks one when creating a campaign.
- Every campaign stores its full resolved spec (`spec_json`) and keeps it forever, so editing an app arm only affects new campaigns; run logs tell the versions apart by `spec_hash` and prompt hashes. Add a separate arm only when both variants should stay selectable in the UI. Eval experiment YAMLs (`evals/experiments/`) are different: never edit one that already has runs — create a new file.
- Startup verifies Gemini keys for the default arm and, for every RAG arm, the Qdrant collection and embedder.
- Evaluation arms live separately in `src/backend/evals/experiments/*.yaml` (eval-run skill).

## Prompt templates (`keeper/prompts/`)

`keeper-vN` (narration), `adjudicate-vN`, `analyze-vN`; Jinja with `StrictUndefined`; if a new adjudicate version needs new output fields, subclass the schema and register it in `ADJUDICATION_SCHEMAS` (`keeper/nodes.py`) so older versions keep sending their exact schema; a `json` filter that keeps Cyrillic; hashes are computed on LF-normalised source. New version = new file; add it to an arm; update tests that render prompts if variables change. Judge prompts for evals are in `src/backend/evals/prompts/`.
