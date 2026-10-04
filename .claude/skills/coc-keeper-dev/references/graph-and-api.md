# Graph, state, API and frontend contract

## Pipeline (`keeper/graph.py`)

```
START ─(bootstrap)→ seed_campaign → END
START ─(player turn)→ prepare_turn ─Command(goto)─┬→ analyze → retrieve ─┐     (use_rag)
                                                  └──────────────────────┴→ adjudicate ─(needs_roll or new_day)→ mechanics → narrate → END
                                                                                        └────────────────────────→ narrate
```

| Node | Model | Reads | Writes | Notes |
|---|---|---|---|---|
| `seed_campaign` | – | `bootstrap` | `scenario`, `investigator`, opening `AIMessage`, empty logs, `bootstrap=None` | no LLM call on campaign creation |
| `prepare_turn` | – | spec | resets `search_query, rules_hits, adjudication, turn_rolls, errors`; `Command(goto=analyze|adjudicate)` | routers don't get the run context, so routing by arm uses `Command` |
| `analyze` | `spec.aux` (structured `SearchQuery`) | last messages | `search_query` | skipped if `forced_hits` present; parse failure → raw player text + `analyze_parse_error` |
| `retrieve` | embedder | `search_query`, `spec.retrieval` | `rules_hits` | consumes and clears `forced_hits` |
| `adjudicate` | `spec.aux` or generator (`adjudicator`) — structured schema paired with the prompt version (`adjudication_schema`: v1/v2 `Adjudication`, v3 `AdjudicationV3` with `new_day`) | sheet names, last roll, recent transcript, `rules_hits` (RAG) | `adjudication` | consumes `forced_kind`; non-in_fiction kinds never roll; retry policy for timeouts/transport |
| `mechanics` | – | `adjudication`, sheet, `roll_log` | `investigator`, `turn_rolls`, `roll_log` (append), `errors` | streams `roll` custom events; also runs for `new_day` without a roll |
| `narrate` | `spec.generator` | prompt with scenario, keeper notes, sheet, mechanics, engine notes, knowledge base (RAG) | `messages` (original `AIMessage`), `citation_log` (RAG) | streams tokens; `EmptyGeneration` on blank output; history trimmed to a token budget; no RetryPolicy |

Progress events are emitted at the **start** of `analyze, retrieve, adjudicate, mechanics, narrate` via `runtime.stream_writer`.

## State (`keeper/state.py`)

Durable: `messages` (add_messages; only player HumanMessages and Keeper AIMessages — aux calls never write here), `scenario`, `investigator`, `roll_log` (append), `citation_log` (append). Inputs: `bootstrap`, `forced_hits`, `forced_kind`. Per-turn scratch: `search_query`, `rules_hits`, `adjudication`, `turn_rolls`, `errors`. Keep values JSON-serialisable (checkpoints).

Context (not checkpointed): `KeeperContext(spec: ExperimentSpec, campaign_id)` passed as `context=`.

Adding a node: async function on `Nodes` (or a shared helper + thin node if evals need it), `add_node` with `timeout=` (async only) and a `RetryPolicy` only for non-streaming calls, edges, new state keys with explicit lifetimes, progress event, then extend `TURN_NODES` in `server/routes.py`, `docs/api.md`, the stage label in `src/frontend/src/i18n/uk.js`, and tests in `tests/test_api.py` / `test_rag_api.py` (fake models: `tests/fakes.py`).

## HTTP / SSE (`server/routes.py`, contract `docs/api.md`)

- JSON: `GET /api/health`, `/api/meta` (arms), `/api/scenarios`, campaigns CRUD (`POST /api/campaigns` with optional `arm`), `GET /api/campaigns/{id}/messages` (`incomplete` flag).
- SSE: `POST /api/campaigns/{id}/turns` `{client_turn_id, text}`, `POST .../turns/retry` (resume from checkpoint with `None` input), `POST .../turns/discard`.
- Events: `progress {node}`, `roll {record}`, `token {text}`, `done {message, investigator}`, `error {code, retryable, message}`; exactly one of done/error ends a stream.
- Pre-stream checks (404 not_found, 409 turn_in_progress / duplicate_turn / turn_incomplete / nothing_to_retry, 422) are FastAPI dependencies. Error codes are mapped in `server/errors.py`; never send `str(exc)`.
- One uvicorn worker; per-campaign `asyncio.Lock`. Each turn writes a JSONL research record (`artifacts/runs/`).

## Frontend consumers (`src/frontend/src`)

| Contract piece | Consumer |
|---|---|
| SSE parsing, uuid | `api/sse.js` |
| JSON endpoints | `api/http.js` |
| turn state machine (stream, retry, discard, errors) | `hooks/useTurnStream.js` |
| campaigns, meta/arms | `hooks/useCampaigns.js`, `hooks/useMeta.js` |
| roll records | `components/RollCard.jsx` |
| citations `[R#]`, sources | `markdown/citations.js`, `components/Citations.jsx` |
| stage labels, error messages | `i18n/uk.js` |

Verify UI changes with `npm run build`; for behaviour, run the app and drive a turn.
