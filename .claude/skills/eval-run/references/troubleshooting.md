# Eval troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `GoogleRateLimitError ... RESOURCE_EXHAUSTED`, items retried then `status: error` | Free-tier per-minute or daily quota | Per-minute: lower `GOOGLE_RPM` in `.env`, `--concurrency 1`. Daily: stop and rerun the same command later — the run resumes. Tell the user rather than looping. |
| `503 UNAVAILABLE ... high demand` | Gemini overload | Retried with back-off automatically; rerun if some items still failed. |
| `Qdrant collection ... does not exist` / connection refused on :6333 | Docker Desktop or the qdrant container is not running, or not ingested | Ask the user to start Docker Desktop if it is closed; `docker compose up -d qdrant`; check `curl 127.0.0.1:6333/collections`; ingest if the collection is missing. |
| embedding errors / connection refused on :11434 | Ollama app not running or model missing | Start Ollama; `ollama pull nomic-embed-text`. |
| `belongs to a different configuration (fingerprint ...)` | Something that affects results changed (arms, prompts, profiles, datasets, retrieval, context_query, k) | Use `--run-name <new>`; never delete an old run to reuse its name without asking. |
| `skip ...: no frozen context` | Context freezing failed for that item (usually quota during analyze) | Rerun: missing contexts are retried first. With no Gemini available use `context_query: question_en` in a separate experiment. |
| `uk_analyze` retrieval mode makes Gemini calls | It needs the analyze query | Use `--query-modes en,uk` when the API is unavailable. |
| Background run shows empty output on Windows | Python block-buffers stdout when not attached to a terminal | Watch `artifacts/eval/runs/<run>/records.jsonl` line count instead (e.g. a Monitor until-loop). |
| Mojibake when printing records | Windows console code page | Prefix commands with `PYTHONUTF8=1`. |
| `dataset validation failed` | Gold data or corpus headings changed | Run `uv run python -m evals.datasets` and fix (gold-data skill); renamed corpus headings break gold sections. |
| Judge columns empty in report | Run used `--no-judge` or judge calls failed | `--judge-only`; report header lists pending counts. |
| Local model very slow | 6 GB GPU, large system prompt | `--concurrency 1`, `--limit` for smoke runs; a 3B model takes ~5–15 s per answer. |
