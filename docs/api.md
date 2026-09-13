# HTTP API (Phase 2: RAG)

Base path: `/api`. In development the Vite dev server proxies `/api` to the backend (`http://localhost:8000`), so the frontend always uses relative URLs and there is no CORS.

The server owns the conversation. The client never sends history — only the text of the player's new action.

## Conventions

- JSON everywhere except the two streaming endpoints, which return `text/event-stream`.
- Errors **before** a stream starts are ordinary HTTP errors with body `{"detail": {"code": "...", "retryable": bool}}`. FastAPI validation errors are `422` with its standard body.
- Errors **inside** a stream arrive as an SSE `error` event (see below); the HTTP status is already `200` at that point.

### Error codes

| code | where | retryable | meaning |
|---|---|---|---|
| `not_found` | 404 | no | campaign does not exist |
| `turn_in_progress` | 409 / SSE | yes | another turn of this campaign is streaming |
| `duplicate_turn` | 409 | no | this `client_turn_id` was already submitted |
| `turn_incomplete` | 409 | yes | the previous turn failed midway; call `retry` or `discard` first |
| `nothing_to_retry` | 409 | no | no incomplete turn to retry/discard |
| `empty_generation` | SSE | yes | the model returned no text (e.g. safety block) |
| `timeout` | SSE | yes | a model call timed out |
| `rate_limited` | SSE | yes | provider quota / 429 |
| `upstream_unavailable` | SSE | yes | provider 5xx / network error |
| `internal_error` | SSE | no | anything else (details only in server logs) |

## Data shapes

### Campaign

```json
{
  "id": "3f2c9a...",
  "title": "Вечір у Львові",
  "scenario_id": "demo-haunt",
  "scenario_title": "Тиша на Пекарській",
  "language": "uk",
  "arm": "play-gemini-rag",
  "use_rag": true,
  "created_at": "2026-09-13T12:00:00Z",
  "updated_at": "2026-09-13T12:05:00Z"
}
```

### Investigator (public sheet)

```json
{
  "name": "Олена Коваль",
  "occupation": "Журналістка",
  "age": 29,
  "characteristics": {"STR": 40, "CON": 60, "SIZ": 50, "DEX": 50, "APP": 50, "INT": 80, "POW": 60, "EDU": 70},
  "skills": {"Spot Hidden": 70, "Library Use": 60, "...": 0},
  "hp": 11, "hp_max": 11,
  "mp": 12, "mp_max": 12,
  "san": 60, "san_max": 99,
  "luck": 55
}
```

### Roll record

```json
{
  "kind": "skill",              // skill | characteristic | luck | sanity
  "target": "Spot Hidden",
  "value": 70,                  // skill value from the server-side sheet
  "difficulty": "regular",      // regular | hard | extreme
  "bonus": 0, "penalty": 0,
  "tens": [4], "units": 2,      // individual dice (tens digits, units digit)
  "roll": 42,                   // 1..100
  "level": "regular",           // critical | extreme | hard | regular | failure | fumble
  "passed": true,
  "pushed": false,
  "turn_no": 3,
  // sanity rolls only:
  "san_loss": 3, "san_after": 57,
  "involuntary_action": false,
  "temporary_insanity": false,  // present only when an INT roll was made (loss >= 5)
  "indefinite_insanity": false,
  "permanent_insanity": false,
  // optional:
  "purpose": "temporary_insanity_check"
}
```

### Citation

Retrieved knowledge-base chunk that the Keeper cited with a `[R<marker>]` marker in its text.

```json
{
  "marker": 2,                         // matches "[R2]" in the Keeper text
  "chunk_id": "6f1c...",
  "source": "Call of Cthulhu 7e Rules Reference (GM Assistant)",
  "section": "Sanity > Losing Sanity",   // heading path inside the source
  "source_type": "rules",              // rules | lore
  "score": 0.71,                       // cosine similarity
  "excerpt": "When an investigator faces ..."   // first ~300 characters of the chunk
}
```

### Message

```json
{"id": "h-6c1e...", "role": "player", "text": "Я оглядаю кімнату."}
{"id": "run-...", "role": "keeper", "text": "Пил лежить ... [R1]", "rolls": [ /* roll records made in this turn */ ],
 "citations": [ /* Citation, only markers that were actually retrieved */ ], "invalid_markers": [ /* e.g. [7] if the text cites [R7] that was not retrieved */ ]}
```

The first message of every campaign is the scenario opening (`role: "keeper"`, no rolls, no citations). In campaigns with `use_rag: false` `citations` and `invalid_markers` are always empty.

## Endpoints

| Method & path | Response |
|---|---|
| `GET /api/health` | `{"status": "ok"}` |
| `GET /api/meta` | `{"version", "default_arm", "generator_model", "arms": [{"id": "play-gemini-rag", "use_rag": true, "generator_model": "gemini-3.8-flash"}, ...]}` |
| `GET /api/scenarios` | `{"items": [{"id", "title": {"uk","en"}, "era": {"uk","en"}, "blurb": {"uk","en"}}]}` |
| `GET /api/campaigns` | `{"items": [Campaign]}` newest first |
| `POST /api/campaigns` | body `{"title": 1..120 chars, "scenario_id": "demo-haunt", "language": "uk"\|"en", "arm"?: one of meta.arms[].id}` → `201 Campaign`. `arm` defaults to `default_arm`; unknown arm → `422 {"detail": {"code": "unknown_arm"}}`. The arm is frozen for the campaign's lifetime. Creates the thread and writes the opening message (no LLM call). |
| `GET /api/campaigns/{id}` | `{...Campaign, "investigator": Investigator}` |
| `GET /api/campaigns/{id}/messages` | `{"items": [Message], "incomplete": bool, "investigator": Investigator}` — `incomplete` means the last turn failed and needs `retry` or `discard` |
| `DELETE /api/campaigns/{id}` | `204` |
| `POST /api/campaigns/{id}/turns` | body `{"client_turn_id": uuid, "text": 1..4000 chars}` → **SSE** |
| `POST /api/campaigns/{id}/turns/retry` | no body → **SSE**; resumes the incomplete turn from its last checkpoint (dice already rolled are not re-rolled) |
| `POST /api/campaigns/{id}/turns/discard` | no body → `{"ok": true}`; drops the incomplete turn (the player message is removed) |

## SSE events (turns and retry)

`EventSource` cannot POST, so read the `fetch` response body as a stream. Frames are `event: <name>\ndata: <json>\n\n`; lines starting with `:` are keep-alive pings.

| event | data | notes |
|---|---|---|
| `progress` | `{"node": "analyze" \| "retrieve" \| "adjudicate" \| "mechanics" \| "narrate"}` | a pipeline stage started — for a status indicator. `analyze` (search query) and `retrieve` (knowledge-base search) appear only in RAG campaigns |
| `roll` | Roll record | 0..n per turn, before narration tokens |
| `token` | `{"text": "..."}` | narration text chunk; append to the Keeper bubble |
| `done` | `{"message": Message, "investigator": Investigator}` | final Keeper message (replace the streamed bubble with it) and the updated sheet |
| `error` | `{"code", "retryable", "message"}` | the turn stopped; if `retryable`, offer Retry (and Discard) |

Exactly one of `done` / `error` ends every stream.
