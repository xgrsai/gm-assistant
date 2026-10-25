# Model profiles and experiment arms

## Model profile (`src/backend/config/models.yaml`, class `ModelProfile` in `keeper/llm.py`)

```yaml
gemini-flash:
  provider: google_genai      # google_genai | ollama | vllm | external (graded outside the harness)
  model: gemini-3.8-flash
  thinking_level: low         # Gemini 3+: low | medium | high ("minimal" is rejected by 3.8-flash)
  max_tokens: 4096            # Gemini thinking tokens count toward output — keep generous
  context_window: 1000000     # used to budget history trimming
  # optional: temperature (ignored by Gemini >= 3.6), num_ctx (Ollama), finetuned, base, quant, model_card
```

Current profiles: `gemini-flash` (generator), `gemini-aux` (analyze/adjudicate), `gemini-judge` (evals), `claude-sonnet-agent` (external judge: `provider: external`, never built by `ModelFactory`; graded by Claude Code subagents through `evals/external_judge.py`), `qwen25-3b-ollama` (local, official blob, pulled; phase3 runs), `qwen25-3b-ollama-aux`, `gemma4-base-ollama` (not pulled), and the Phase 4 pair `qwen25-3b-selfq4-ollama` (base) and `keeper-qwen25-3b-ollama` (`finetuned: true`).

**Phase 4 pair.** The author's Ollama rejects `ADAPTER`, so both models are GGUFs converted and quantized to Q4_K_M by the same llama.cpp and created in Ollama with the official qwen2.5:3b template: `keeper-qwen25-3b:sft-v1` (LoRA merged into Qwen2.5-3B-Instruct) and `qwen2.5-3b-selfq4:v1` (same weights, same conversion, no LoRA). The base is therefore NOT the official `qwen2.5:3b` blob of `qwen25-3b-ollama` (phase3 runs use that profile; never edit it). Sampling, `num_ctx`, `context_window` and `max_tokens` are copied from `qwen25-3b-ollama` into both, so base and fine-tuned differ only by the model. Both carry the same `model_card` (repo-relative JSON, `training/model_cards/keeper-qwen25-3b-sft-v1.json`, written by `python -m sft.check_adapter`), which records the check `status`, the Ollama digest of each tag and the hashes of the weights, data and split. `evals/model_cards.py` reads it: the role of a profile is `ft` when `finetuned: true`, else `base`; the digest is `ollama_digests.<role>` and the weights are the `gguf.files` entry of that role. `python -m evals.run` stops unless the card's `status` is `passed` (override: `--allow-failed-card`, recorded in `config.json`) and Ollama serves each such tag under the card's digest (a dry run only warns), and the run's `config.json` gets a `model_cards` key; the card file hash in it is informational (the card holds timestamps), the rest is in the fingerprint.

`ModelProfile.snapshot()` is what run configs, run logs and frozen SFT inputs record instead of `model_dump()`: it leaves `model_card` out while it is unset, so profiles without a card dump key-for-key as they did before the field existed and the fingerprints of recorded runs do not change (`tests/test_llm.py` pins the phase3 profiles). A field added to `ModelProfile` later must follow the same rule.

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
