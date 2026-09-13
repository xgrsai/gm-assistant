# Verified issues

## [critical][frontend][confirmed] src/frontend/src/lib/api.js is gitignored by the root 'lib/' rule, so a fresh clone cannot build
File: .gitignore:17
Category: devops

The root .gitignore is the stock Python template. Its unanchored 'lib/' pattern matches src/frontend/src/lib/, so the only API client file was never committed. App.jsx imports it, so on a fresh clone (or in CI, or in a docker build from a clean checkout) Vite fails with 'Failed to resolve import ./lib/api'. The author's machine works only because the file happens to exist locally.

Evidence: `git check-ignore -v src/frontend/src/lib/api.js` -> `.gitignore:17:lib/  src/frontend/src/lib/api.js`. `git status --ignored` lists `!! src/frontend/src/lib/`. `git ls-files src/frontend` has no lib/api.js. App.jsx:5 `import { sendMessage } from './lib/api'`.

Fix: Anchor the Python rule (`/lib/`, `/lib64/`) or add `!src/frontend/src/lib/` after line 17, then `git add src/frontend/src/lib/api.js`. Also add a Node section to .gitignore (node_modules/, dist/, .vite/). node_modules is currently NOT ignored: `git check-ignore` returns nothing for src/frontend/node_modules, and it is only absent from `git status` because the host folder is empty (created by the compose anonymous volume). The first local `npm install` will flood git status.

Verifier: Verified. `git check-ignore -v src/frontend/src/lib/api.js` gives `.gitignore:17:lib/`. `git status --ignored` lists `!! src/frontend/src/lib/`. `git ls-files src/frontend` has no api.js, and App.jsx:5 imports './lib/api'. The .gitignore has no node_modules rule, and src/frontend/node_modules exists on the host but is empty, which fits the anonymous-volume explanation. Any fresh clone, CI job or clean docker build fails to resolve the import.

## [critical][infra][confirmed] Root .gitignore 'lib/' rule silently drops src/frontend/src/lib/api.js: fresh clone of HEAD cannot build
File: .gitignore:17
Category: bug

The root .gitignore is the stock GitHub Python template. Its unanchored 'lib/' rule (meant for Python build output) matches any directory named lib at any depth. That includes src/frontend/src/lib/, so api.js was never committed. App.jsx imports it, which means the public repo at HEAD 2799aaf has a frontend that fails to build or run. The unanchored 'dist/' rule (line 13) works the same way. The deps clsx + tailwind-merge suggest a planned src/lib/utils.js 'cn' helper, and it would be ignored too (verified).

Evidence: git check-ignore -v src/frontend/src/lib/api.js -> '.gitignore:17:lib/  src/frontend/src/lib/api.js'; git check-ignore -v --no-index src/frontend/src/lib/utils.js -> '.gitignore:17:lib/'; git ls-files has no src/frontend/src/lib/*; git log --all -- src/frontend/src/lib returns nothing; src/frontend/src/App.jsx:5 'import { sendMessage } from './lib/api''. On a fresh clone, 'docker compose up' starts Vite, and the first page load fails with 'Failed to resolve import "./lib/api"'.

Fix: Anchor the Python-template rules (/lib/, /lib64/, /build/, /dist/ where Python-specific), or add `!src/frontend/src/lib/`. Then `git add src/frontend/src/lib/api.js` and commit. Keep ignoring the frontend dist/ on purpose.

Verifier: I verified this directly. `git check-ignore -v src/frontend/src/lib/api.js` returns `.gitignore:17:lib/`. `git ls-files` lists nothing under src/frontend/src/lib. App.jsx:5 has `import { sendMessage } from './lib/api'`. The file exists on disk but is not in HEAD, so a fresh clone plus `docker compose up` (which bind-mounts ./src/frontend) fails to resolve the import. The utils.js check-ignore result also matches. One small overreach: `dist/` ignoring src/frontend/dist is actually wanted, because it is build output, so that rule is not a problem the way lib/ is. Critical is reasonable because the thesis's public artifact cannot run from the repo.

## [high][backend][confirmed] Root .gitignore 'lib/' excludes src/frontend/src/lib/api.js, so a fresh clone cannot build and never reaches the backend
File: .gitignore:17
Category: devops

The Python gitignore template's `lib/` pattern matches any directory named lib, including src/frontend/src/lib. api.js exists on disk but is untracked (`git status --ignored` shows `!! src/frontend/src/lib/`), and App.jsx imports it. A fresh clone plus `docker compose up` fails the frontend build or import, and the committed repo does not contain the only client of the backend API. The same pattern would also silently ignore any future backend `lib/` package.

Evidence: .gitignore:17 `lib/`; `git ls-files` has no src/frontend/src/lib/api.js; src/frontend/src/App.jsx:5 `import { sendMessage } from './lib/api'`.

Fix: Anchor the pattern (`/lib/`), or add `!src/frontend/src/lib/`, then `git add src/frontend/src/lib/api.js`. Review other broad template patterns (build/, dist/, env/, var/) the same way.

Verifier: .gitignore:17 `lib/` (Python template). `git status --ignored` shows `!! src/frontend/src/lib/`, git ls-files lacks api.js, and App.jsx:5 imports './lib/api'. A fresh clone of the public repo fails `vite build`, and the dev server shows an unresolved-import error, so the only backend client is missing from the repository. High fits because the published project cannot run from a clone.

## [high][frontend][confirmed] No persistence: refresh, a clicked link or a render crash loses the whole campaign, but the README advertises 'Persistent conversation history'
File: src/frontend/src/App.jsx:8
Category: missing-feature

The whole campaign exists only in `useState` in App. Nothing is saved to localStorage or the backend: core/models.py is empty and the API is stateless, taking the full history each call. Reloading the tab, closing it, or clicking a markdown link in a GM reply (it navigates the same tab, see FE-08) wipes the session. There is also no React ErrorBoundary in main.jsx, so any render exception unmounts the app and loses state. For example, react-markdown 9 throws on non-string children. README.md:9 says 'Persistent conversation history', which is false.

Evidence: App.jsx:8 `const [messages, setMessages] = useState([])`; no localStorage or IndexedDB anywhere in src (grep); src/backend/app/core/models.py is empty; main.jsx:6-10 renders <App/> with no error boundary; README.md:9 '- Persistent conversation history'.

Fix: Short term: persist to localStorage per campaign id, wrapped in try/catch, and add an ErrorBoundary. Proper fix, which fits the planned LangGraph move: the server owns the thread. Use a LangGraph checkpointer (SqliteSaver or PostgresSaver) keyed by thread_id, and have the frontend store only campaign/thread ids and send `{thread_id, message}` instead of the whole transcript. Until then, fix the README.

Verifier: State lives only in useState, grep finds no localStorage, main.jsx has no ErrorBoundary, and README.md:9 says 'Persistent conversation history'. One detail in the evidence is wrong: src/backend/app/core/models.py does not exist at all (core/ has only __init__, prompts, services, urls, views), so it is not 'empty'. The conclusion is the same. react-markdown 9's `children || ''` plus its throw on non-string values is confirmed from the 9.0.1 source.

## [medium][backend][confirmed] Hardcoded Django SECRET_KEY committed to public git history
File: src/backend/app/config/settings.py:24
Category: security

SECRET_KEY is a string literal in settings.py rather than read from the environment. It is in both commits (99f2e69 and 2799aaf) pushed to https://github.com/xgrsai/gm-assistant.git. Anyone with the repo can forge signed session cookies, password-reset tokens and signed values for any deployment that uses this file, and the admin/auth/sessions apps are enabled.

Evidence: settings.py:24  SECRET_KEY = 'django-insecure-&d5oh=cu)(24y12-...'
`git grep django-insecure $(git rev-list --all)` finds it in 99f2e69 and 2799aaf.

Fix: Use SECRET_KEY = os.environ['DJANGO_SECRET_KEY'], or a dev-only fallback that is used only when DEBUG is on. Add the variable to .env.example. Generate a new key, because the current one is already public.

Verifier: settings.py:24 has the literal 'django-insecure-&d5oh=...' key. git grep finds it in 99f2e69 and 2799aaf, and the GitHub repo xgrsai/gm-assistant is public. Current practical impact is small, though. There is no deployment and the DB is empty (0 bytes). Sessions use the DB backend, so a key alone does not forge a login, and password-reset tokens need an existing user's hash. The key is still burned and must come from env before any deployment. High overstates it for an undeployed prototype.

## [medium][backend][confirmed] DEBUG hardcoded to True, so compose DEBUG env var does nothing; ALLOWED_HOSTS empty
File: src/backend/app/config/settings.py:27
Category: bug

docker-compose passes DEBUG=true, but settings.py never reads it: DEBUG is always True, including in the Docker image. With DEBUG on, any uncaught exception returns Django's technical 500 page (code, paths, settings, request data) to the caller. ALLOWED_HOSTS=[] only works because DEBUG=True adds the localhost variants. If DEBUG is switched off, every request gets 400 DisallowedHost. Host 'backend' (compose service name) or a LAN IP is rejected even now.

Evidence: settings.py:27 `DEBUG = True`; settings.py:29 `ALLOWED_HOSTS = []`; docker-compose.yml:24 `- DEBUG=true` (never read); Dockerfile:23 runs runserver with those settings.

Fix: DEBUG = os.environ.get('DJANGO_DEBUG','false').lower() in ('1','true','yes'); ALLOWED_HOSTS = os.environ.get('DJANGO_ALLOWED_HOSTS','localhost,127.0.0.1').split(','). Default to safe values, and fail on startup when DEBUG is off and SECRET_KEY is missing.

Verifier: settings.py:27 `DEBUG = True` and :29 `ALLOWED_HOSTS = []`. docker-compose sets DEBUG=true, but os.environ is never read for it. Django validates against ['.localhost','127.0.0.1','[::1]'] only when DEBUG is on and ALLOWED_HOSTS is empty, so the behaviour is as described. The technical 500 page appears for uncaught exceptions: DRF re-raises anything that is not an APIException. Django's SafeExceptionReporterFilter masks settings named *KEY*, so GOOGLE_API_KEY would not be shown. Medium fits because this is a dev-only setup, but the env var being ignored is a real bug.

## [medium][backend][confirmed] Unauthenticated, unthrottled endpoint spends the Gemini API key and is reachable on all interfaces
File: src/backend/app/core/views.py:7
Category: security

POST /api/chat/ has no authentication, permission class or throttling (no REST_FRAMEWORK settings, so DRF defaults to AllowAny). It forwards arbitrary history to Gemini on the server's paid key. The container binds 0.0.0.0:8000 and compose publishes 8000 on all host interfaces, so anyone on the network can script requests and use up quota or run up cost. CORS only restricts browsers, not curl.

Evidence: views.py:7-8 `@api_view(["POST"])\ndef chat(request):` with no permission_classes or throttle_classes; settings.py has no REST_FRAMEWORK dict; Dockerfile:23 `runserver 0.0.0.0:8000`; docker-compose.yml:19 `- "8000:8000"`.

Fix: Add at least AnonRateThrottle/UserRateThrottle (REST_FRAMEWORK DEFAULT_THROTTLE_CLASSES/RATES) and some auth (API token or session login). Publish the port as 127.0.0.1:8000:8000 for local dev. In a FastAPI/LangGraph rewrite, add an auth dependency and a rate limiter (e.g. slowapi) from the start.

Verifier: views.py has only @api_view(["POST"]), with no permission or throttle classes, and there is no REST_FRAMEWORK dict, so the defaults are AllowAny and no throttling. The Dockerfile runs runserver on 0.0.0.0:8000, and compose publishes "8000:8000" on all host interfaces. The key abuse from the LAN is real. It is not internet-facing unless deployed, so medium rather than high.

## [medium][backend][confirmed] No validation of the 'messages' payload: malformed input causes uncaught 500s
File: src/backend/app/core/views.py:17
Category: bug

The view only checks truthiness and the last item's role. Lines 9 and 17 run outside the try block, so type errors there are uncaught 500s (DEBUG technical page). Errors inside services.py are caught and returned as 500 when they should be 400. Roles, content type, emptiness, per-message length and list size are never checked.

Evidence: views.py:9 `request.data.get("messages", [])`: a JSON array body such as `[1,2]` gives AttributeError 'list' has no attribute 'get' (uncaught 500). A form-encoded body gives messages as a str, and views.py:17 `messages[-1].get(...)` on a character raises AttributeError (500). `{"messages": {"a":1}}` gives KeyError -1 (500). `{"messages": 5}` gives TypeError (500). `{"messages":["hi",{"role":"user","content":"x"}]}` fails at services.py:19 `msg["role"]` on a str (TypeError, caught, returned as 500). `{"messages":[{"role":"user"}]}` fails at services.py:24 with KeyError returned as `{"error": "'content'"}` 500. Non-string content such as `{"content": {}}` fails pydantic validation in types.Part(text=...) (500 with the validation dump).

Fix: Add a DRF serializer, or a Pydantic model in FastAPI: MessageSerializer(role=ChoiceField(['user','assistant']), content=CharField(max_length=N, allow_blank=False, trim_whitespace=True)) and ListField(child=MessageSerializer(), min_length=1, max_length=M). Call is_valid(raise_exception=True) before any indexing, and return 400 with field errors.

Verifier: I traced each payload against views.py:9-17 and services.py. `[1,2]` gives list.get AttributeError outside the try. Form-encoded input makes QueryDict.get return a str, then 'c'.get raises AttributeError. A dict gives KeyError -1, and an int gives TypeError. All four are uncaught 500s. A str history item gives TypeError inside the try (500), and a missing 'content' gives `{"error": "'content'"}` with 500. Non-str content fails pydantic in types.Part. Every case is correct. The impact is wrong status codes and error leaks, not a data or security break, so medium.

## [medium][backend][confirmed] Raw exception text returned to the client and shown in the UI
File: src/backend/app/core/views.py:26
Category: security

Every exception, including google.genai APIError, pydantic ValidationError, the missing-API-key ValueError and KeyError, is sent to the client as str(e) with status 500. APIError's string includes the upstream status and response JSON (quota/project details, 'API key not valid' messages). The frontend shows err.error directly. Upstream 429/503 are also flattened into 500, so the client cannot tell 'retry later' from a bug.

Evidence: views.py:26-30 `except Exception as e: return Response({"error": str(e)}, status=500)`; src/frontend/src/lib/api.js:12 `throw new Error(err.error || ...)`.

Fix: Log the exception server-side (logger.exception) and return a generic message with a correlation id. Map google.genai.errors.ClientError/ServerError codes to 400/429/502/503/504 explicitly. Catch validation errors earlier (see BE-04).

Verifier: views.py:26-30 returns str(e) with status 500. In google-genai v2.4.0, errors.py APIError builds its message as f'{self.code} {self.status}. {self.details}', where details is the full response JSON, so upstream error JSON reaches the client. api.js:12 throws err.error, and App.jsx renders it in the header. 429/503 are flattened to 500. Confirmed.

## [medium][backend][confirmed] response.text can be None; null content returned with HTTP 200 and sent back in the next turn's history
File: src/backend/app/core/services.py:40
Category: bug

In google-genai 2.4.0, GenerateContentResponse.text returns None when there are no candidates (prompt blocked by safety, prompt_feedback.block_reason), when the candidate has no content or parts (finish_reason SAFETY/RECITATION/PROHIBITED_CONTENT), or when there are no text parts (e.g. MAX_TOKENS hit during thinking). The view returns {"content": null} as a success. The frontend stores it (App.jsx:21) and sends it back next turn, where services.py:20 builds types.Part(text=None), an empty part. This will probably make later requests fail. finish_reason and prompt_feedback are never checked.

Evidence: services.py:40 `return response.text`; google-genai v2.4.0 types.py `_get_text`: `if not self.candidates or not self.candidates[0].content or not self.candidates[0].content.parts: return None`; views.py:25 `Response({"content": reply})`; App.jsx:21 `{ role: 'assistant', content: reply }`.

Fix: Check response.prompt_feedback and candidates[0].finish_reason. If text is empty, return a 502/422 with a readable message such as 'response blocked (SAFETY)' or a fallback in-character line. Never return null content. Also reject null/empty content in request validation.

Verifier: services.py:40 `return response.text`. In google-genai, _get_text returns None when there are no candidates, content or parts, or when there are only thought/non-text parts. The view returns {"content": null} with 200, and App.jsx:21 stores it and resends it next turn. types.Part(text=None) then serializes to an empty part, which the API rejects, so the whole conversation is stuck until 'New game'. finish_reason and prompt_feedback are never checked.

## [medium][backend][confirmed] No HTTP timeout and no retries on the Gemini call; a stalled request blocks a worker indefinitely
File: src/backend/app/core/services.py:12
Category: bug

The client is created without http_options. In google-genai 2.4.0, HttpOptions.timeout defaults to None, which is passed to httpx as timeout=None (no timeout). retry_options=None means `stop_after_attempt(1)`, so transient 429/500/503/504 errors are never retried. The call is synchronous and the frontend fetch has no AbortController, so a hung upstream ties up a runserver thread (or a gunicorn sync worker) and leaves the UI spinning.

Evidence: services.py:12 `genai.Client(api_key=settings.GOOGLE_API_KEY)` has no http_options. _api_client.py (v2.4.0) retry_args: `if options is None: return {'stop': tenacity.stop_after_attempt(1), 'reraise': True}`; get_timeout_in_seconds returns None when timeout is unset, and it is passed to `httpx_client.build_request(timeout=http_request.timeout)`.

Fix: genai.Client(api_key=..., http_options=types.HttpOptions(timeout=60_000, retry_options=types.HttpRetryOptions(attempts=3, http_status_codes=[429,500,503,504]))). Make the timeout configurable via settings. In LangGraph, use the model's timeout/max_retries parameters or a RetryPolicy on the node.

Verifier: I checked google-genai v2.4.0 _api_client.py. retry_args returns {'stop': stop_after_attempt(1), 'reraise': True} when options is None. get_timeout_in_seconds returns None when timeout is unset, and that value becomes the request timeout, so there is no timeout. services.py:12 passes no http_options. api.js has no AbortController (grep found none). Confirmed.

## [medium][backend][confirmed] System prompt contradicts thesis scope (dark fantasy vs Call of Cthulhu Keeper) and has no RAG grounding rules
File: src/backend/app/core/prompts.py:1
Category: architecture

The thesis describes a Call of Cthulhu Keeper that grounds answers in retrieved rules/lore/scenarios and prefers RAG context over parametric knowledge when they conflict. The prompt is a generic 'dark fantasy' GM ('Bloodborne meets Witcher'), with no slot for retrieved context, no instruction to cite or prefer it, and no way to answer rules questions: 'Never break character' blocks out-of-character rules clarifications, which the thesis uses to measure correctness. The 150-word limit is only in the prompt; no max_output_tokens is set.

Evidence: prompts.py:1 `You are a Game Master for a dark fantasy tabletop RPG.`; prompts.py:8 `- Never break character`; prompts.py:11 `Tone: dark, mysterious, tense. Think Bloodborne meets Witcher.`; services.py:28-30 config has only system_instruction.

Fix: Make prompts per game system (CoC Keeper). Add a templated `<context>{retrieved_chunks}</context>` section with rules: 'If context conflicts with your knowledge, follow context; if context lacks the rule, say so'. Allow a marked out-of-character mode for rules questions. Version prompts so evaluation runs are reproducible.

Verifier: prompts.py is a 'dark fantasy' GM ('Bloodborne meets Witcher', 'Never break character'), with no context slot or conflict-priority rule. The thesis targets a Call of Cthulhu Keeper with RAG priority over parametric knowledge. GenerateContentConfig sets only system_instruction. This is accurate and matters for the thesis, so medium is fair even though the code is an MVP.

## [medium][backend][confirmed] No LLM/RAG abstraction: model id, provider and prompt hardwired into one function
File: src/backend/app/core/services.py:7
Category: architecture

get_gm_response combines client construction, message conversion, prompt selection and the provider call. There is no retriever, vector store (Qdrant), embedding step (nomic-embed-text), or interface for swapping Gemini with a LoRA fine-tuned model. That makes the thesis's planned comparison (prompt-only vs RAG vs fine-tune vs RAG+FT) impossible without a rewrite.

Evidence: services.py:12 client creation; services.py:27 `model="gemini-3.5-flash"` literal; services.py:29 fixed SYSTEM_PROMPT; no retrieval code anywhere in src/backend.

Fix: In the LangGraph migration, define a StateGraph with nodes: validate/load thread → retrieve (Qdrant retriever, query rewrite optional) → generate (chat model chosen from config: ChatGoogleGenerativeAI or a local OpenAI-compatible endpoint serving the LoRA model via vLLM/Ollama) → persist. Put model/provider/top_k in a config object so each experiment variant is a config change. FastAPI + uvicorn fits this async/streaming service better than Django+DRF.

Verifier: services.py hardwires client, model literal 'gemini-3.5-flash', SYSTEM_PROMPT and conversion in one function. There is no retrieval, embedding or vector-store code, so none of the thesis RAG/fine-tuning architecture exists yet. Medium is fair as the main gap for the planned LangGraph work.

## [medium][frontend][confirmed] New Campaign during an in-flight request brings back the old conversation and leaves the new game stuck in loading
File: src/frontend/src/App.jsx:20
Category: bug

handleSend closes over `updated`, a snapshot of the old conversation, and when the request resolves it writes `[...updated, reply]` without checking whether the conversation is still the same one. handleNewGame (lines 29-32) clears messages and error but does not reset `loading` or cancel the request. The Sidebar's New Campaign button is not disabled while loading. So during a GM reply, clicking New Campaign: (1) shows an empty game that still has the typing indicator and a disabled input, (2) then the old conversation plus its reply replaces the new game when the fetch resolves, or (3) the old request's error appears in the new game. Concurrent sends from the input itself are prevented by `disabled` (MessageInput.jsx:10, 31, 40), so New Campaign is the only way to trigger this today. It will get worse once streaming or regenerate is added.

Evidence: App.jsx:14-21 `const updated = [...messages, userMsg] ... const reply = await sendMessage(updated); setMessages([...updated, { role: 'assistant', content: reply }])`; App.jsx:29-32 `const handleNewGame = () => { setMessages([]); setError(null) }` (no setLoading(false), no abort); Sidebar.jsx:46-47 button with `onClick={onNewGame}` and no disabled state.

Fix: Give each conversation an id (or keep an AbortController in a ref). handleNewGame should abort the controller, bump the id, and call setLoading(false). In handleSend, ignore results whose id no longer matches, and use functional updates (`setMessages(prev => [...prev, reply])`) instead of the captured snapshot. Pass `signal` into fetch in api.js.

Verifier: The code matches the quote. handleSend captures `updated` and calls `setMessages([...updated, reply])` at App.jsx:21. handleNewGame (29-32) does not reset loading or abort the request, and the Sidebar button has no disabled state. The race is real, but it only happens when the user clicks New Campaign during a pending request. That is a narrow trigger, so high is inflated.

## [medium][frontend][confirmed] A failed request leaves an orphan user turn in history, clears the typed text and offers no retry
File: src/frontend/src/App.jsx:22
Category: bug

On error the user message stays in `messages`, and it looks exactly like a message that succeeded. The input was already cleared (MessageInput.jsx:12 runs setValue('') right after onSend, whatever the result), so the player has to retype. If they do, the next request contains [..., user(failed), user(retyped)]. Both go to the backend, which maps them 1:1 into Gemini contents (services.py:16-23). The model then sees two consecutive player actions, one of them a duplicate, and no GM response between them. There is no retry or edit option.

Evidence: App.jsx:15 `setMessages(updated)` runs before the request; the catch at App.jsx:22-23 only does `setError(err.message)` and never rolls back. MessageInput.jsx:11-12 `onSend(text); setValue('')`.

Fix: Mark the optimistic message `{status:'pending'|'failed'}`. On failure, either remove it and restore the text to the input (make MessageInput controlled from App, or have onSend return a promise and clear only on success), or show 'Failed - Retry / Edit' on the bubble. Before sending, filter failed messages out of history.

Verifier: setMessages(updated) runs before the request and the catch only calls setError. MessageInput clears the value right after the synchronous onSend. services.py maps every message into Gemini contents, so the next send includes consecutive user turns. The bug is real. Medium fits better than high, because the text still goes out and the model usually copes with a duplicated user turn.

## [medium][frontend][confirmed] Hard-coded campaigns in Sidebar and header do nothing; 'New Campaign' wipes the current game without confirmation
File: src/frontend/src/components/Sidebar.jsx:3
Category: ux

CAMPAIGNS is a static array. The campaign buttons have no onClick, so clicking 'Sunken City' does nothing and 'The Ashen Crown' is always highlighted. The header title and subtitle in App.jsx are hard-coded too. 'New Campaign' does not create a campaign entry. It calls handleNewGame, which destroys the current and only conversation with one click, no confirm and no undo. Given there is no persistence (FE-04), that is irreversible data loss. The UI looks like multi-campaign support that does not exist.

Evidence: Sidebar.jsx:3-6 `const CAMPAIGNS = [{ id: 1, name: 'The Ashen Crown', active: true }, { id: 2, name: 'Sunken City', active: false }]`; Sidebar.jsx:29-35 `<button className=...>` with no onClick; App.jsx:42-45 hard-coded 'The Ashen Crown' / 'Dark fantasy campaign'; Sidebar.jsx:46-53 'New Campaign' -> App.jsx:29-32 clears messages.

Fix: Lift campaigns into state (`campaigns[]`, `activeCampaignId`, messages per campaign), persist them, wire onClick to select and add `aria-current`, and derive the header from the active campaign. Either rename the button to 'Restart' with a confirm dialog, or make it actually create a new campaign and leave the old one intact. Until then, remove the fake entries.

Verifier: CAMPAIGNS is a static array and its buttons have no onClick. The header text is hard-coded. New Campaign calls handleNewGame, which clears the only conversation with no confirmation.

## [medium][frontend][confirmed] User bubbles collapse Shift+Enter newlines and do not wrap long words
File: src/frontend/src/components/MessageBubble.jsx:11
Category: bug

The input explicitly offers 'Shift+Enter for new line' (MessageInput.jsx:48), but the user message is rendered as plain text in a div without `whitespace-pre-wrap`, so newlines show as single spaces and multi-line actions or dialogue get flattened. There is also no `break-words`/`overflow-wrap:anywhere`, so a long URL or unbroken string overflows the 75% bubble horizontally. The flex child also lacks min-w-0.

Evidence: MessageBubble.jsx:11-13 `<div className="bg-ash-800 ... text-base text-parchment leading-relaxed">{message.content}</div>`; MessageInput.jsx:48 'Enter to send · Shift+Enter for new line'.

Fix: Add `whitespace-pre-wrap break-words min-w-0` to the user bubble, and `break-words min-w-0` to the assistant bubble container.

Verifier: The user bubble at MessageBubble.jsx:11 has no whitespace-pre-wrap or break-words, while the hint at MessageInput.jsx:48 invites Shift+Enter. Newlines collapse to spaces. A long unbroken string widens the flex item past the 75% wrapper because its min-width is auto.

## [medium][frontend][confirmed] No request timeout or cancel: a hung backend or LLM call leaves the UI loading indefinitely
File: src/frontend/src/lib/api.js:4
Category: bug

fetch has no AbortSignal and no timeout, and the UI has no Stop button. If the Gemini call hangs (the backend calls generate_content synchronously with no timeout in services.py) or the dev server stalls, `loading` stays true, the input stays disabled and the only escape is reloading the page, which loses the campaign (FE-04). The same missing signal is what makes FE-02 impossible to fix cleanly.

Evidence: api.js:4-8 `await fetch(`${API_URL}/api/chat/`, { method: 'POST', headers: ..., body: JSON.stringify({ messages }) })` has no `signal`; App.jsx has no abort or stop handler.

Fix: Accept a `signal` in sendMessage, combine a user-cancel AbortController with `AbortSignal.timeout(60000)` (or AbortSignal.any), show a Stop button while loading, and treat AbortError as a non-error.

Verifier: api.js fetch has no signal or timeout and there is no Stop control. services.py calls generate_content synchronously and sets no HttpOptions timeout.

## [medium][frontend][confirmed] Null or undefined `content` from the API gives an empty GM bubble and bad history (plausible)
File: src/frontend/src/lib/api.js:16
Category: bug

sendMessage returns `data.content` without validating it. The backend returns `{"content": response.text}`, and in google-genai `response.text` can be None, for example when a candidate is blocked by safety filters or has no text part. The frontend then stores `{role:'assistant', content:null}`. react-markdown treats `children || ''` as empty, so a blank bubble appears. That null is sent back in history on the next turn, where services.py builds `types.Part(text=None)`, which may be rejected and would break every later turn. A non-string (e.g. an object) would make react-markdown throw and, with no ErrorBoundary, unmount the app.

Evidence: api.js:15-16 `const data = await res.json(); return data.content`; services.py last line `return response.text`; react-markdown 9 `const children = options.children || ''` and it throws 'Unexpected value ... for `children` prop, expected `string`' for non-strings.

Fix: In api.js: `if (typeof data.content !== 'string' || !data.content.trim()) throw new Error('The Game Master had no response (possibly blocked). Try rephrasing.')`. On the backend, detect an empty or blocked response and return a structured error (finish_reason) instead of null.

Verifier: api.js returns data.content unchecked, and services.py returns response.text. In google-genai, response.text is None when a candidate has no text parts (for example a safety block). The null goes back as types.Part(text=None) on the next turn, which will likely be rejected as a Part with no data. That rejection is plausible but was not reproduced.

## [medium][frontend][confirmed] No streaming: long narrations appear only after the full generation finishes
File: src/frontend/src/lib/api.js:15
Category: missing-feature

The client waits for one JSON body and the backend uses non-streaming generate_content. GM narration is the core UX, and users see only 'Game Master is thinking...' for the whole generation. A RAG step (retrieval plus rerank) will add more latency. Moving to LangGraph is a natural point to add token streaming (graph.astream / astream_events with stream_mode='messages').

Evidence: api.js:15 `const data = await res.json()`; App.jsx:20-21 appends the reply only after the await; services.py uses `client.models.generate_content(...)` (not generate_content_stream).

Fix: Backend (FastAPI + LangGraph): a StreamingResponse or SSE endpoint yielding token chunks, plus a final event with metadata (e.g. retrieved sources). Frontend: POST with fetch and read `res.body.getReader()` with TextDecoder (EventSource cannot POST), appending to a placeholder assistant message by id with functional setState. Tie it to the AbortController from FE-12.

Verifier: The client awaits one res.json() and the backend uses non-streaming generate_content. This is a real missing feature, and the LangGraph migration is a good point to add streaming.

## [medium][frontend][confirmed] No responsive layout: fixed 224px sidebar leaves about 150px for the chat on phones
File: src/frontend/src/components/Sidebar.jsx:10
Category: ux

The sidebar is always `w-56 shrink-0` with no breakpoint, drawer or toggle. On a 375px-wide phone the main column gets about 151px, and inside it the header (title plus error pill, `justify-between`), `px-6` paddings and 75-85% max-width bubbles leave only a few words per line. `html, body, #root {height:100%}` with the input at the bottom also runs into mobile browser dynamic toolbars (100% vs dvh), which can hide the input. vite.config.js exposes the dev server on 0.0.0.0, so phone testing looks intended, but it will not work (see FE-18).

Evidence: Sidebar.jsx:10 `<aside className="w-56 shrink-0 flex flex-col ...">`; App.jsx:35 `h-full flex`; no `md:`/`sm:` classes anywhere in src; index.css:6-8 `height: 100%`.

Fix: `hidden md:flex` on the aside plus a hamburger button in the header that opens it as an overlay drawer; use `px-3 md:px-6`, `max-w-[90%] md:max-w-[75%]`; use `h-dvh` (Tailwind 3.4 supports it) for the root container.

Verifier: The sidebar is always w-56 shrink-0, and there are no sm:/md: classes anywhere in src. Root height is 100%. The numbers are right: 375 - 224 = 151px.

## [medium][frontend][confirmed] Accessibility gaps: icon-only send button without a name, textarea without a label, chat log not announced
File: src/frontend/src/components/MessageInput.jsx:38
Category: ux

The send button contains only an SVG icon (lucide renders aria-hidden='true' by default), so its accessible name is empty. The textarea has only a placeholder, no <label> or aria-label. New GM messages and the typing indicator are not in an aria-live region, so screen-reader users get no notice of a reply. The error has no role='alert' (FE-07). Active campaign buttons have no aria-current. Animations (fade-in, blinking dots, smooth scroll) ignore prefers-reduced-motion.

Evidence: MessageInput.jsx:38-45 `<button onClick={handleSend} ...><SendHorizonal size={18} /></button>` (no aria-label); MessageInput.jsx:26-37 textarea with `placeholder="Describe your action…"` only; ChatWindow.jsx:19 plain `<div className="flex-1 overflow-y-auto ...">`; TypingIndicator.jsx:14-16 `animate-blink`; ChatWindow.jsx:15 `behavior: 'smooth'`; Sidebar.jsx:29-35 no aria-current.

Fix: `aria-label="Send message"` on the button; `aria-label="Your action"` (or a visually hidden label) on the textarea; `role="log" aria-live="polite" aria-relevant="additions"` on the message list; `role="status"` on TypingIndicator; `aria-current="true"` on the active campaign; `motion-safe:animate-fade-in` and `behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'`.

Verifier: The send button is icon-only with no aria-label, the textarea has only a placeholder, there is no live region or role=alert, and there is no aria-current. One detail is off: package.json pins lucide-react ^0.383.0, which resolves to 0.383.x only, and the aria-hidden default came in later lucide releases. The SVG still has no title, so the button's accessible name is empty either way.

## [medium][frontend][confirmed] API base URL hard-coded to localhost:8000 with no Vite proxy: breaks from LAN or phone, and gets baked into any production build
File: docker-compose.yml:11
Category: devops

In compose, `VITE_API_URL=http://localhost:8000` is correct for a browser on the same machine. The fetch runs in the browser, not in the container, so `http://backend:8000` would be wrong. It fails once the app is opened from any other device (vite binds 0.0.0.0, vite.config.js:7): on a phone, `localhost:8000` is the phone itself, and the backend CORS allowlist has only localhost and 127.0.0.1 on port 5173. VITE_* values are also inlined at build time, so a `vite build` image would ship a hard-coded localhost URL. The value in compose is identical to the fallback in api.js:1, so the env var adds nothing. The frontend container runs the Vite dev server (Dockerfile CMD `npm run dev`); there is no production build or static serving stage.

Evidence: docker-compose.yml:11 `- VITE_API_URL=http://localhost:8000`; api.js:1 `import.meta.env.VITE_API_URL || 'http://localhost:8000'`; vite.config.js:6-9 `server: { host: '0.0.0.0', port: 5173 }` with no proxy; settings.py:134-137 CORS_ALLOWED_ORIGINS localhost/127.0.0.1:5173 only; Dockerfile:12 `CMD ["npm", "run", "dev", ...]`.

Fix: Use relative URLs (`fetch('/api/chat/')`) and add `server.proxy: { '/api': { target: process.env.API_PROXY_TARGET || 'http://localhost:8000', changeOrigin: true } }`, with API_PROXY_TARGET=http://backend:8000 in compose. The proxy runs inside the container, so the service hostname is right there. This removes CORS in dev and works from any device. For production, add a multi-stage Dockerfile (node build, then nginx serving dist and reverse-proxying /api).

Verifier: docker-compose.yml:11 sets VITE_API_URL to http://localhost:8000, which is the same as the api.js fallback. vite.config.js has no proxy, CORS_ALLOWED_ORIGINS lists only localhost and 127.0.0.1 on port 5173, and the Dockerfile CMD runs the dev server. Access from a LAN device or phone fails, as described.

## [medium][frontend][confirmed] No package-lock.json, npm install in Dockerfile, no .dockerignore: non-reproducible builds and host node_modules copied into the image
File: src/frontend/Dockerfile:6
Category: devops

No lockfile exists on disk or in git history (`git log --all -- '*package-lock.json'` is empty), so every `npm install` resolves the ^ ranges fresh (react-markdown ^9, vite ^5.4, tailwind ^3.4, etc.), and builds on different days or machines can differ. `COPY package*.json` suggests a lockfile was expected. With no .dockerignore, `COPY . .` sends the host's node_modules, if any, into the build context and overlays it on the Linux install. On this Windows host that means win32 native binaries (esbuild, rollup optional deps) mixed into an Alpine image, a common cause of 'installed esbuild for another platform' or rollup native-module errors. It also makes the build context very large. The image runs as root.

Evidence: Dockerfile:5-8 `COPY package*.json ./` / `RUN npm install` / `COPY . .`; `ls src/frontend` shows no package-lock.json and no .dockerignore; host platform is win32.

Fix: Run `npm install` once locally and commit package-lock.json. In the Dockerfile use `RUN npm ci`. Add src/frontend/.dockerignore with `node_modules`, `dist`, `.vite`, `.env*`. Add `USER node` for non-dev images.

Verifier: No package-lock.json exists on disk or in git history, the Dockerfile runs `npm install`, and there is no .dockerignore. Host node_modules is currently empty, so nothing contaminates the build today, but the first local npm install on Windows would copy win32 binaries through `COPY . .`.

## [medium][frontend][confirmed] UI does not match the thesis agent: hard-coded dark-fantasy branding, no scenario/system choice, no RAG source display, client-owned history
File: src/frontend/src/App.jsx:42
Category: architecture

The thesis describes a Keeper agent for systems like Call of Cthulhu that combines RAG over rules, lore and scenarios with LoRA fine-tuning, where retrieved context overrides parametric knowledge. The frontend hard-codes a dark-fantasy campaign ('The Ashen Crown', a Sword icon, 'Game Master is thinking'), the welcome text and prompts.py say 'dark fantasy ... Bloodborne meets Witcher', and there is no way to pick a ruleset or scenario, which is the key that selects the RAG corpus. Nothing in the message model can show retrieved chunks or citations, which would support the thesis's RAG-priority claim and the planned quantitative comparison (e.g. showing which rule or scenario passage grounded an answer, or toggling RAG, fine-tuned and base modes for evaluation). The client sends the full transcript each turn (App.jsx:20), so request size and tokens grow without limit, and this conflicts with LangGraph's server-side thread and checkpointer model.

Evidence: App.jsx:42-45 'The Ashen Crown' / 'Dark fantasy campaign'; ChatWindow.jsx:5-9 WELCOME; TypingIndicator.jsx:12 'Game Master is thinking'; api.js:7 `body: JSON.stringify({ messages })`; message shape is only `{role, content}` (App.jsx:13, 21).

Fix: Define the new API contract before the LangGraph/FastAPI migration: `POST /threads` with {system, scenario_id, mode: 'rag'|'ft'|'rag+ft'} returning thread_id; `POST /threads/{id}/messages` streaming tokens plus a final `{sources:[{doc, chunk_id, score}], mode}` event. Extend the message shape to `{id, role, content, status, sources?}`, render a collapsible 'Sources' section under GM replies, and make the system/scenario and branding data-driven.

Verifier: The code confirms hard-coded dark-fantasy branding (App.jsx, prompts.py 'Bloodborne meets Witcher'), no system or scenario selection, a message shape of only {role, content}, and the full transcript sent every turn. The thesis describes a RAG plus LoRA Keeper agent for Call of Cthulhu. This is a valid architectural gap.

## [medium][frontend][found-by-verifier] Anonymous node_modules volume goes stale after dependency changes
File: docker-compose.yml:9
Category: devops

The frontend service mounts an anonymous volume at /app/node_modules. When Compose recreates a container it keeps the old anonymous volumes. After a new dependency is added to package.json (for example remark-gfm, or a LangGraph SDK client) and `docker compose up --build` runs, the container still sees the old node_modules from the previous volume instead of the freshly installed image layer. Vite then fails with 'Failed to resolve import'. The fixes suggested in FE-09, FE-14 and FE-21 all add dependencies, so this will hit soon.

Evidence: docker-compose.yml:7-9 `volumes: - ./src/frontend:/app - /app/node_modules`; Dockerfile:5-6 `COPY package*.json ./` / `RUN npm install`.

Fix: Document `docker compose up --build -V` (--renew-anon-volumes), or use a named volume and run `npm ci` in an entrypoint when package-lock.json changes, or use Compose `develop.watch` with `action: rebuild` on package.json.

Verifier: 

## [medium][infra][confirmed] Django SECRET_KEY hardcoded and committed to a public GitHub repo
File: src/backend/app/config/settings.py:24
Category: security

settings.py is tracked, and the remote github.com/xgrsai/gm-assistant is public (GitHub API: visibility 'public'). The Django SECRET_KEY is a literal in that file, so it has leaked for good through git history. Anyone who deploys this settings file as-is has forgeable sessions, password-reset tokens and signed cookies. The GOOGLE_API_KEY is handled correctly: .env is ignored and never committed (git log --all -- .env is empty).

Evidence: settings.py:24 SECRET_KEY = 'django-insecure-&d5oh=cu)(24y12-...'; git ls-files includes src/backend/app/config/settings.py; api.github.com/repos/xgrsai/gm-assistant -> visibility: public.

Fix: SECRET_KEY from env (DJANGO_SECRET_KEY), failing fast when DEBUG is false. Add it to .env.example and generate a new key for any real deployment.

Verifier: settings.py:24 is the stock startproject key 'django-insecure-&d5oh=...'. The file is tracked, and the GitHub API returns private:false, visibility:public. The claim is accurate. I lowered severity because nothing is deployed, DEBUG is hard-coded True, and the app has no auth, sessions or users. The leaked key only matters if someone deploys this settings file, and that deployment would already be insecure for other reasons. It is still a real hygiene issue that must be fixed before any deployment.

## [medium][infra][confirmed] DEBUG/ALLOWED_HOSTS/CORS are hardcoded; compose 'DEBUG=true' is dead config
File: src/backend/app/config/settings.py:27
Category: devops

docker-compose passes DEBUG=true, but settings.py never reads it. DEBUG is the literal True and ALLOWED_HOSTS is []. Django accepts localhost/127.0.0.1 only because DEBUG is on. With DEBUG off, or when the backend is reached by service name (e.g. an nginx or Vite proxy calling http://backend:8000), every request returns 400 DisallowedHost. CORS_ALLOWED_ORIGINS is hardcoded to localhost:5173, so there is no production origin. The Gemini model name is also hardcoded in services.py:22, although it is exactly the kind of setting you would switch between for RAG vs fine-tuned experiments.

Evidence: docker-compose.yml:24 '- DEBUG=true'; settings.py:27 'DEBUG = True'; settings.py:29 'ALLOWED_HOSTS = []'; settings.py:134-137 CORS list hardcoded; grep shows no os.environ reads except GOOGLE_API_KEY (settings.py:132).

Fix: Same as proposed. Note that the model name is at services.py:27.

Verifier: docker-compose.yml:24 sets DEBUG=true, but settings.py never reads it (the only os.environ read is GOOGLE_API_KEY at :132). DEBUG=True is hard-coded at :27 and ALLOWED_HOSTS=[] at :29. With DEBUG on and ALLOWED_HOSTS empty, Django only allows .localhost, 127.0.0.1 and [::1], so a request with Host 'backend:8000' gets a 400 even in DEBUG. CORS is hard-coded at :134-137. One small error: the model name is at services.py:27 (`model="gemini-3.5-flash"`), not :22.

## [medium][infra][confirmed] .gitignore has no Node/Vite entries: node_modules and .env.local are not ignored
File: .gitignore:1
Category: devops

The .gitignore contains only the Python template plus .vscode/. node_modules/ is not ignored. The host dir is empty today only because the compose anonymous volume masks it. Once the author runs 'npm install' on the host (e.g. for editor IntelliSense), thousands of files will show up as untracked. Many packages' lib/ and dist/ subfolders would still be hidden by the Python rules, so the result is an inconsistent half-ignored tree. The Vite secret file convention .env.local / .env.*.local is also not ignored.

Evidence: git check-ignore -v --no-index src/frontend/node_modules/react/index.js -> no match; git check-ignore --no-index src/frontend/.env.local -> no match; src/frontend/node_modules/ exists on disk (empty, created by the '/app/node_modules' anonymous volume in docker-compose.yml:9).

Fix: Same as proposed: add Node/Vite ignore rules (node_modules/, *.local, .env.*, !.env.example, npm-debug.log*) and the planned ML artifact rules.

Verifier: `git check-ignore --no-index` gives no match for src/frontend/node_modules/react/index.js or src/frontend/.env.local. The .gitignore is the Python template plus .vscode/. src/frontend/node_modules exists on the host and is empty. The finding is correct, but high is too severe. The failure is noisy untracked files and a chance of an accidental commit, not breakage or a leak today, since no .env.local exists.

## [medium][infra][confirmed] No production build or runtime path (runserver + Vite dev server, source bind-mounted)
File: docker-compose.yml:1
Category: devops

The only images are dev images. The backend runs Django's runserver, and requirements.txt has no gunicorn or uvicorn. The frontend runs 'npm run dev', nothing ever runs 'vite build', and there is no nginx/static stage. Both services bind-mount source over the image, so the image contents are never what runs. VITE_API_URL is a runtime env var for the dev server. For a production build Vite inlines import.meta.env at build time, so it would need a build ARG, and the value 'http://localhost:8000' only works when the browser is on the Docker host. There is no compose override or prod profile.

Evidence: src/backend/Dockerfile:23 CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]; src/frontend/Dockerfile:12 CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]; docker-compose.yml:7-9 and 20-21 bind mounts; docker-compose.yml:11 VITE_API_URL=http://localhost:8000; requirements.txt has only django, djangorestframework, google-genai, django-cors-headers.

Fix: Same as proposed: a dev override file plus a prod-like compose using gunicorn/uvicorn and a multi-stage Vite build served by nginx with an /api reverse proxy.

Verifier: All quoted lines are accurate: backend CMD runserver, frontend CMD `npm run dev`, bind mounts, VITE_API_URL as a runtime env, and requirements.txt without gunicorn/uvicorn. Vite does pick VITE_* variables from process.env in dev, so dev works, and a production build would need them at build time. High is too severe for a local-only thesis prototype with no deployment target. It matters before any demo deployment.

## [medium][infra][confirmed] Python base image pin is stale, 3.10 reaches EOL in weeks, and Alpine/musl is a poor fit for the planned RAG/LoRA stack
File: src/backend/Dockerfile:1
Category: devops

The tag python:3.10.19-alpine3.23 does exist on Docker Hub (active, last updated 2026-02-07). But it is frozen at 3.10.19 while 3.10.20 and 3.10.21 have shipped (3.10.21-alpine3.23 was published 2026-09-02), so the image misses those security fixes. Python 3.10 reaches end of life on 2026-10-31 (PEP 619), about seven weeks from now. The thesis stack (embeddings, PEFT/LoRA, likely torch/transformers/sentence-transformers) is built around manylinux glibc wheels. PyTorch publishes no musllinux wheels, so on Alpine these installs either fail or compile from source for a very long time. LangGraph 1.2.11 still supports >=3.10, but the ecosystem will drop 3.10 soon after EOL.

Evidence: Dockerfile:1 'FROM python:3.10.19-alpine3.23'; Docker Hub tags API: '3.10.19-alpine3.23' last_updated 2026-02-07, '3.10.21-alpine3.23' last_updated 2026-09-02; PEP 619 / endoflife.date: 3.10 EOL Oct 2026; pypi.org/simple/torch lists no musllinux files.

Fix: Same as proposed: python:3.12/3.13-slim, pinned and bumped by Dependabot/Renovate, with training kept outside the API image.

Verifier: Checked on the web. Docker Hub has 3.10.19-alpine3.23 as active, last_updated 2026-02-07, and 3.10.20-alpine3.23 (2026-07-08) and 3.10.21-alpine3.23 (2026-09-02) also exist. endoflife.date lists Python 3.10 EOL as 2026-10-31 (PEP 619), with latest 3.10.21. PyPI's simple index for torch has zero musllinux files. LangGraph 1.2.11 requires_python >=3.10. Every factual claim holds. The torch/Alpine point only applies if ML libraries go into the API image. Ollama or Qdrant plus remote embeddings would not need torch there, but moving off 3.10/Alpine is still sound advice.

## [medium][infra][confirmed] Frontend image: Node 20 is EOL, no package-lock.json, and non-reproducible 'npm install'
File: src/frontend/Dockerfile:1
Category: devops

node:20-alpine has had no upstream security fixes since Node 20 reached EOL on 2026-04-30. No lockfile is committed or even exists on disk, so 'npm install' re-resolves every caret range (e.g. react-markdown ^9, vite ^5) on each build, and builds on different days can differ. The anonymous '/app/node_modules' volume in compose also keeps stale dependencies after package.json changes unless the author runs 'docker compose up --build -V'.

Evidence: src/frontend/Dockerfile:1 'FROM node:20-alpine'; Dockerfile:5-6 'COPY package*.json ./' + 'RUN npm install'; 'ls src/frontend/package-lock.json' -> No such file; docker-compose.yml:9 '- /app/node_modules'. Node.js 20 EOL: April 30, 2026.

Fix: Commit package-lock.json, use node:24-alpine (maintenance LTS until 2028-04-30) with `npm ci`, and use `docker compose up --build -V` after dependency changes.

Verifier: src/frontend/Dockerfile:1 is node:20-alpine. endoflife.date shows Node 20 EOL 2026-04-30, and the Docker Hub node:20-alpine tag was last updated 2026-04-15. There is no package-lock.json on disk, and the Dockerfile uses `npm install`. The anonymous /app/node_modules volume keeping stale deps is standard Compose behaviour. One correction to the fix: Node 24's active LTS ends 2026-10-20 and its maintenance/EOL is 2028-04-30, so 'active LTS until 2028-04' is wrong, though Node 24 is still the right target.

## [medium][infra][confirmed] Compose stack has none of the infrastructure the thesis method requires (vector DB, embedding model)
File: docker-compose.yml:1
Category: missing-feature

The thesis describes a chunked corpus embedded with nomic-embed-text, stored in a vector DB like Qdrant and retrieved by cosine similarity, plus a LoRA-tuned model. The compose file only has frontend and backend services with a direct Gemini call, and there are no persistent volumes at all. This is the main DevOps gap between the repo and what the thesis claims to implement.

Evidence: docker-compose.yml defines only 'frontend' and 'backend'; no volumes: section; requirements.txt has no qdrant-client/langchain/langgraph; tezy.txt lines 27-28, 33 describe chunking, embeddings, vector DB, cosine similarity.

Fix: Same as proposed.

Verifier: The compose file defines only frontend and backend, with no volumes section. requirements.txt has no vector-DB or LangChain/LangGraph packages. tezy.txt lines 27-28 and 33 describe chunking, embeddings, Qdrant plus nomic-embed-text, and cosine similarity. This is an accurate description of the gap between the thesis method and the repo.

## [medium][infra][confirmed] No CI, no linting/formatting config, no tests
File: README.md:1
Category: code-quality

The repo has no .github/workflows, no ruff/black/pyproject.toml, no eslint/prettier config, no .editorconfig, no pre-commit and no tests on either side. A minimal CI job running 'npm ci && npm run build' on a clean checkout would have caught D1, the missing api.js, immediately. For a thesis that plans a quantitative comparison of RAG vs fine-tuning vs hybrid, an automated evaluation or test harness in CI is also the natural place to make results reproducible.

Evidence: ls .github .editorconfig .pre-commit-config.yaml pyproject.toml ruff.toml src/frontend/eslint.config.js src/frontend/.prettierrc* -> all 'No such file'; package.json scripts only dev/build/preview (no lint/test); git ls-files contains no test files.

Fix: Same as proposed.

Verifier: The repo has no .github, .editorconfig, pyproject.toml or lint configs, and package.json scripts are only dev/build/preview. No test files are tracked. A clean-checkout `npm run build` in CI would have failed on the missing ./lib/api, so the claim that CI would have caught D1 holds.

## [medium][infra][confirmed] Public repo has no LICENSE; no guard against committing copyrighted RPG corpus or model weights
File: README.md:1
Category: devops

The GitHub repo is public with license: null, so by default nobody may legally reuse the code. README never mentions licensing. The thesis corpus will include the Call of Cthulhu rulebook and the module 'Clean Up, Aisle Four!' (tezy.txt line 26). These are third-party copyrighted materials and must not be pushed to a public repo, yet nothing in .gitignore prevents it. The same goes for LoRA adapters, datasets and vector DB storage.

Evidence: ls LICENSE* -> No such file; api.github.com/repos/xgrsai/gm-assistant -> visibility public, license null; tezy.txt:26 mentions the rulebook and 'Clean Up, Aisle Four!'; .gitignore has no data/corpus/weights rules.

Fix: Same as proposed.

Verifier: There is no LICENSE file, and the GitHub API returns license:null and visibility:public. tezy.txt:26 names the CoC rulebook and 'Clean Up, Aisle Four!'. .gitignore has no data, corpus or weights rules. No copyrighted material is committed yet, so the corpus part is preventive, but the corpus work is the next step in a public repo.

## [medium][infra][found-by-verifier] Unauthenticated, un-rate-limited Gemini proxy published on all host interfaces with DEBUG tracebacks
File: docker-compose.yml:19
Category: security

The compose ports '8000:8000' and '5173:5173' bind to 0.0.0.0 on the host. POST /api/chat/ has no auth, no throttling and no input size limit, and it forwards the whole client-supplied message history to Gemini using the author's GOOGLE_API_KEY. ALLOWED_HOSTS=[] under DEBUG only checks the Host header, which any LAN or VPN client can set to 'localhost' with curl. That lets them burn the author's Gemini quota or billing. DEBUG=True also returns technical 500/404 pages showing URL patterns and settings details.

Evidence: docker-compose.yml:18-19 'ports: - "8000:8000"'; core/views.py '@api_view(["POST"]) def chat(request)' with no permission_classes/throttle_classes; settings.py has no REST_FRAMEWORK throttling config; settings.py:27 DEBUG = True. Scenario: `curl -H 'Host: localhost' -X POST http://<author-LAN-IP>:8000/api/chat/ -d '{"messages":[{"role":"user","content":"..."}]}'` succeeds and consumes the key.

Fix: In dev, bind to loopback: '127.0.0.1:8000:8000' and '127.0.0.1:5173:5173'. Add DRF AnonRateThrottle (or slowapi after a FastAPI move), cap the message count and length, and read DEBUG from env (see D3).

Verifier: 

## [low][backend][confirmed] 'rest_framework' missing from INSTALLED_APPS: browsable API renderer raises TemplateDoesNotExist (500)
File: src/backend/app/config/settings.py:34
Category: bug

The views use DRF (@api_view, Response), but 'rest_framework' is not in INSTALLED_APPS. DRF's default renderers include BrowsableAPIRenderer, which loads the template 'rest_framework/api.html' through the APP_DIRS loader, and that loader only searches installed apps. Opening http://localhost:8000/api/chat/ in a browser (Accept: text/html; GET gives a 405 rendered by the browsable renderer) therefore returns a TemplateDoesNotExist 500 instead of a 405 page. There is also no REST_FRAMEWORK config (renderers, parsers, throttles, exception handler).

Evidence: settings.py:34-45 INSTALLED_APPS = ['django.contrib.admin', ..., 'core', 'corsheaders'] with no 'rest_framework'; views.py:1-2 import from rest_framework.

Fix: Add 'rest_framework' to INSTALLED_APPS, or set REST_FRAMEWORK = {'DEFAULT_RENDERER_CLASSES': ['rest_framework.renderers.JSONRenderer'], 'DEFAULT_PARSER_CLASSES': ['rest_framework.parsers.JSONParser'], ...} so the API is JSON-only (this also removes the form-encoded input path from BE-04).

Verifier: 'rest_framework' is absent from INSTALLED_APPS (settings.py:34-45). BrowsableAPIRenderer is in DRF's default renderers and loads 'rest_framework/api.html' through the app-directories loader, so a browser GET to /api/chat/ with Accept text/html fails with TemplateDoesNotExist. The real frontend fetch sends no Accept header (*/*), so JSONRenderer is chosen and the app works. Only manual browsing of the endpoint breaks, so low rather than medium.

## [low][backend][confirmed] CorsMiddleware placed below CommonMiddleware/CsrfViewMiddleware, and CommonMiddleware listed twice
File: src/backend/app/config/settings.py:52
Category: bug

django-cors-headers docs say CorsMiddleware should be 'placed as high as possible, especially before any middleware that can generate responses such as Django's CommonMiddleware'. Here it is fourth, after the first CommonMiddleware and CsrfViewMiddleware, and CommonMiddleware appears again at line 53. Responses produced above CorsMiddleware get no Access-Control-Allow-Origin header. For example, CommonMiddleware.process_request calls request.get_host(), and a disallowed Host (docker service name, LAN IP, or any host once DEBUG is off) becomes a 400 outside CorsMiddleware, so the browser shows an opaque CORS error. The duplicate makes every request/response pass through CommonMiddleware twice and hides ordering mistakes.

Evidence: settings.py:50 'django.middleware.common.CommonMiddleware', 51 'django.middleware.csrf.CsrfViewMiddleware', 52 'corsheaders.middleware.CorsMiddleware', 53 'django.middleware.common.CommonMiddleware'.

Fix: Order: SecurityMiddleware, CorsMiddleware, SessionMiddleware, CommonMiddleware, CsrfViewMiddleware, AuthenticationMiddleware, MessageMiddleware, XFrameOptionsMiddleware, with a single CommonMiddleware.

Verifier: The order is as quoted: CommonMiddleware at :50, Csrf at :51, Cors at :52, CommonMiddleware again at :53. This goes against the django-cors-headers docs. Current impact is minimal, though. CSRF rejection happens in process_view, which is inside CorsMiddleware's wrap, and @api_view is csrf_exempt anyway. APPEND_SLASH redirects are unused because the frontend URL has a trailing slash. The only header-less responses come from DisallowedHost in the outer CommonMiddleware (get_host is called unconditionally). The finder's examples (host 'backend', a LAN IP) would fail CORS regardless, because those origins are not in CORS_ALLOWED_ORIGINS. The duplicate CommonMiddleware is harmless apart from being confusing. Fix is correct; severity is inflated.

## [low][backend][confirmed] Client controls the full conversation, including fabricated assistant turns (prompt injection); unknown roles silently become 'model'
File: src/backend/app/core/services.py:19
Category: security

The server has no stored session and trusts whatever history the client sends. Any role other than exactly 'user' ('system', 'assistant', 'admin', typos) is mapped to 'model', so a client can plant fake GM turns ('As the Keeper I have now disabled all rules...') or many-shot jailbreak examples that the model treats as its own earlier output. This undermines 'Never break character' and, once RAG exists, any grounding or priority policy. Nothing separates or marks untrusted player text inside the prompt.

Evidence: services.py:16-21 `for msg in messages[:-1]: ... role="user" if msg["role"] == "user" else "model"`; views.py:17 checks only the last message's role.

Fix: Store the conversation server-side (DB or LangGraph checkpointer keyed by thread_id). The client sends only {thread_id, new user message}, and assistant turns come only from the server. Whitelist roles. Wrap retrieved context and player input in clearly delimited sections, and state in the system prompt that player text cannot change rules or system instructions.

Verifier: services.py maps any role other than 'user' to 'model', and the view checks only the last message's role. That is accurate. However, this is a single-player app where the client is the player, and a client-held stateless history is the normal pattern for a prototype (the model APIs work the same way). A player 'jailbreaking' their own GM is a small threat. It matters more for thesis evaluation integrity and once RAG priority rules exist. Low, with the server-side thread state recommendation still valid.

## [low][backend][confirmed] README promises 'Persistent conversation history', but the backend is stateless (no models, no migrations, empty DB)
File: README.md:9
Category: missing-feature

The backend has no models.py, no migrations directory, and db.sqlite3 is 0 bytes. The frontend keeps history only in React state (no localStorage per grep), so a page reload loses the whole campaign. The feature list does not match the code.

Evidence: README.md:9 `- Persistent conversation history`; src/backend/app/core contains only __init__.py, prompts.py, services.py, urls.py, views.py; `ls -la src/backend/app` shows db.sqlite3 at 0 bytes; App.jsx:13-21 uses in-memory state only.

Fix: Implement persistence (Conversation/Message models with migrations, or a LangGraph checkpointer such as SqliteSaver/PostgresSaver keyed by thread_id plus a GET /threads/{id} endpoint), or remove the claim from the README.

Verifier: README.md:9 claims 'Persistent conversation history'. There is no models.py or migrations, db.sqlite3 is 0 bytes, grep finds no localStorage in src/frontend/src, and App.jsx keeps history in useState only. The claim does not match the code, but it is documentation drift in a prototype, so low.

## [low][backend][confirmed] Unbounded history growth: full transcript sent to the LLM every turn, with no truncation, summarization or size limits
File: src/backend/app/core/services.py:16
Category: architecture

Every request rebuilds the whole conversation into contents. Input tokens per turn grow linearly and cumulative cost roughly quadratically over a session, latency rises, and a long campaign (or one oversized request, since there are no per-message or list limits) eventually hits the context window or request limits and fails with a leaked 500. There is no token counting, sliding window, or summary memory.

Evidence: services.py:16-22 iterates over all `messages[:-1]`; services.py:31-37 sends `[*history, last]`; App.jsx:20 `sendMessage(updated)` sends the entire array each time; views.py has no length checks.

Fix: Cap list length and characters per message at validation time. Keep a server-side window of the last N turns plus a rolling summary (LangGraph: a summarization node or trim_messages from langchain_core.messages with a token budget). Use client.models.count_tokens to enforce a budget.

Verifier: The full history is resent every turn (services.py:16-22, App.jsx:20) and there are no size limits. Hitting the context window is unrealistic in practice: the gemini-3.5-flash docs list 1,048,576 input tokens, and 150-word turns would take thousands of turns to fill it. Implicit prompt caching also softens the cost. The real effect is linear cost and latency growth per turn, which is a design point to handle in the LangGraph rewrite (trim_messages or a summary), not a medium defect.

## [low][backend][confirmed] Blocking, non-streaming LLM call on the Django dev server
File: src/backend/app/core/services.py:26
Category: architecture

generate_content is synchronous and returns only after the full answer (Gemini 3.5 Flash thinks at 'medium' by default, adding latency). The request thread is held for the whole generation, and the user sees only a typing indicator. There is no SSE/streaming endpoint, and the stack (sync DRF view under runserver/WSGI) cannot stream tokens cleanly.

Evidence: services.py:26 `response = client.models.generate_content(...)`; views.py:24-25 returns one JSON object; Dockerfile:23 `runserver`.

Fix: Use client.models.generate_content_stream (or client.aio.models.* in an async stack) and expose an SSE endpoint. With FastAPI + LangGraph this maps directly to `graph.astream(..., stream_mode='messages')` returned as a StreamingResponse.

Verifier: generate_content is synchronous and non-streaming. The Gemini docs confirm the default thinking_level is 'medium' for Gemini 3.5 Flash. The claim that the stack 'cannot stream tokens cleanly' is overstated: Django supports StreamingHttpResponse (SSE) under WSGI and runserver. This is a UX and architecture improvement rather than a defect, so low.

## [low][backend][confirmed] No logging: all backend errors swallowed without a traceback
File: src/backend/app/core/views.py:26
Category: devops

The broad except returns a response without logging, and settings.py has no LOGGING config. Failed Gemini calls, validation errors and safety blocks leave no server-side trace (runserver logs only the 500 request line), so debugging and later thesis evaluation (latency, token usage, failure rate) have no data.

Evidence: views.py:26-30 `except Exception as e: return Response({"error": str(e)}, ...)` with no logger call; no `import logging` in core/*; settings.py has no LOGGING.

Fix: Add module loggers (logger = logging.getLogger(__name__)) and logger.exception in the handler. Log model, latency, usage_metadata token counts and finish_reason per call. Configure LOGGING (JSON to stdout for Docker). Consider LangSmith/Langfuse tracing after moving to LangGraph.

Verifier: There is no logger and no LOGGING setting. One correction: Django's default logging does record returned 5xx responses via django.request (log_response logs 'Internal Server Error: /api/chat/' when DEBUG is on and output goes to the console), plus runserver's access line. So there is some trace, but no exception type, message or traceback, and no latency or token metrics. Low to medium; low for a prototype.

## [low][backend][confirmed] No tests at all
File: src/backend/app/core/views.py:8
Category: code-quality

There is no tests.py, tests/ package, pytest config or CI. The payload shape handling (BE-04), the None-text path (BE-06) and the role mapping (BE-10) have no coverage, and the planned LangGraph/FastAPI migration would have no regression safety net.

Evidence: `git ls-files` lists no test files; src/backend/app/core contains only __init__.py, prompts.py, services.py, urls.py, views.py; requirements.txt has no pytest.

Fix: Add pytest (+ pytest-django or FastAPI TestClient). Test with the Gemini client mocked (patch get_gm_response): valid request, each malformed payload → 400, upstream error → mapped status, empty/blocked response. Also add a small offline eval set of 'query → reference Keeper answer' pairs for the thesis comparison.

Verifier: git ls-files lists no tests, no pytest in requirements.txt, and no CI. Confirmed. For a thesis prototype this is a quality gap rather than a medium defect, though tests are worth adding before the FastAPI/LangGraph migration.

## [low][backend][confirmed] Dev server (runserver) as container CMD; image runs as a user that cannot write /app
File: src/backend/Dockerfile:23
Category: devops

The image's only entrypoint is Django's development server, which Django documents as not for production (no hardening, single process, autoreloader), combined with hardcoded DEBUG=True. No production server (gunicorn/uvicorn) is in requirements.txt. /app is copied as root and the process runs as django-user, so without the compose bind mount `manage.py migrate` cannot create or write db.sqlite3 (read-only directory). No HEALTHCHECK.

Evidence: Dockerfile:23 `CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]`; Dockerfile:6 `COPY ./app /app` (root-owned); Dockerfile:21 `USER django-user`; requirements.txt:1-4 has no gunicorn/uvicorn.

Fix: Keep runserver only in a compose dev override. Default CMD: `gunicorn config.wsgi -b 0.0.0.0:8000 --workers 2 --timeout 120`, or uvicorn for ASGI/FastAPI. Use COPY --chown=django-user or a writable data volume. Add a /healthz endpoint and HEALTHCHECK.

Verifier: Dockerfile:23 runs runserver. COPY ./app /app is root-owned and the process runs as django-user (--no-create-home), so without the compose bind mount /app is not writable for db.sqlite3. There is no gunicorn/uvicorn and no HEALTHCHECK. Compose is explicitly a dev setup with a bind mount, and the repo has no production deployment, so low to medium.

## [low][backend][confirmed] Python 3.10 on Alpine: EOL next month, and musl blocks the planned fine-tuning/ML stack
File: src/backend/Dockerfile:1
Category: architecture

Python 3.10 reaches end of life in October 2026 (PEP 619), about a month from today. Alpine/musl is a poor base for the thesis roadmap: PyTorch (needed by transformers/PEFT/LoRA and local embedding models) publishes no musllinux wheels, and other scientific or native packages often compile from source or fail. Local dev also uses 3.10 (__pycache__ cpython-310).

Evidence: Dockerfile:1 `FROM python:3.10.19-alpine3.23`; __pycache__/*.cpython-310.pyc in src/backend/app.

Fix: Move to python:3.12-slim or 3.13-slim (Debian/glibc). Keep the fine-tuning/training environment (GPU, torch, peft) in a separate image or notebook from the API service.

Verifier: PEP 619 schedules Python 3.10 EOL for October 2026, which is correct. PyTorch has no musllinux wheels. But the API service in the planned stack (langgraph, langchain-google-genai, qdrant-client, fastapi, and Ollama served separately for nomic-embed-text) is mostly pure Python or ships musllinux wheels (pydantic-core, numpy, orjson). torch/peft belong in a separate training environment anyway. Moving to a slim glibc base is sensible, but this is low, not medium.

## [low][backend][confirmed] `rm -rf /tmp` deletes the system temp directory instead of the copied requirements file
File: src/backend/Dockerfile:13
Category: devops

The intent is to remove /tmp/requirements.txt, but the whole /tmp directory is removed from the image. Anything run as django-user that uses tempfile (Django TemporaryUploadedFile for uploads over 2.5 MB, pip, HF/embedding caches, google-genai file uploads, test runners) loses the standard temp dir: Python falls back to /var/tmp or the cwd (/app), and tools that hardcode /tmp fail.

Evidence: Dockerfile:5 `COPY ./requirements.txt /tmp/requirements.txt`; Dockerfile:13 `rm -rf /tmp && \`.

Fix: Use `rm -f /tmp/requirements.txt`, or `pip install --no-cache-dir -r /tmp/requirements.txt` followed by `rm /tmp/requirements.txt`.

Verifier: Dockerfile:13 `rm -rf /tmp` removes the whole directory, and django-user cannot recreate it at / (root-owned). Python's tempfile falls back to /var/tmp or the cwd. The practical impact today is nil, since there are no uploads or temp usage, but it is a real latent bug. Low is correct.

## [low][backend][confirmed] Docker layer caching defeated and no .dockerignore (db.sqlite3/__pycache__ baked into image)
File: src/backend/Dockerfile:6
Category: devops

The app source is copied before the dependency install, so any code edit invalidates the pip layer and reinstalls everything. There is no .dockerignore in src/backend (or at the root), so the local db.sqlite3 and __pycache__/*.pyc from Windows Python 3.10 are copied into the image.

Evidence: Dockerfile:5-6 COPY requirements, then `COPY ./app /app` before Dockerfile:10-12 `pip install`; `ls src/backend/.dockerignore` gives 'No such file'; src/backend/app/db.sqlite3 and core/__pycache__ exist on disk.

Fix: Order: COPY requirements.txt, RUN pip install --no-cache-dir, then COPY app. Add src/backend/.dockerignore with __pycache__/, *.pyc, db.sqlite3, .env, .venv.

Verifier: COPY ./app (line 6) comes before pip install (line 10), so any code change invalidates the pip layer. There is no .dockerignore in the root or src/backend (ls confirms), and db.sqlite3 and __pycache__ exist in the build context. Compose's bind mount hides them at runtime, but they are baked into the image. Confirmed low.

## [low][backend][confirmed] Unused Django admin/auth/sessions/messages/sqlite stack; migrations never applied
File: src/backend/app/config/settings.py:35
Category: architecture

The API uses none of admin, auth, sessions, messages or the database, yet all are installed and /admin/ is routed. db.sqlite3 is 0 bytes, so runserver warns about unapplied migrations and any admin login fails with 'no such table'. This adds attack surface (admin plus the public SECRET_KEY) and startup weight with no benefit, and suggests Django is heavier than this service needs.

Evidence: settings.py:35-39 admin, auth, contenttypes, sessions, messages; settings.py:82-87 sqlite DATABASES; config/urls.py:21 `path('admin/', admin.site.urls)`; db.sqlite3 size 0.

Fix: If staying on Django, remove the unused apps, middleware and /admin/ route, or run migrations in the entrypoint once persistence is added. If moving to FastAPI + LangGraph, this whole layer goes away; use a LangGraph checkpointer (SQLite/Postgres) for conversation state.

Verifier: admin, auth, sessions and messages are installed, /admin/ is routed (config/urls.py), db.sqlite3 is 0 bytes and no migrate step exists, so runserver prints the unapplied-migrations warning. Confirmed low.

## [low][backend][confirmed] New genai.Client (httpx client + TLS setup) created on every request
File: src/backend/app/core/services.py:12
Category: code-quality

Each chat request builds a new Client, and with it a new httpx client and SSL context. Connection pooling and keep-alive are lost, so every call pays a fresh TLS handshake. It is closed only via __del__ when garbage-collected. Configuration (timeouts, retries) is also scattered per call.

Evidence: services.py:12 `client = genai.Client(api_key=settings.GOOGLE_API_KEY)` inside get_gm_response; google-genai 2.4.0 client.py provides close()/__enter__ for explicit lifecycle.

Fix: Create one module-level (lazy) client with http_options and reuse it, or with FastAPI create it in the lifespan handler and close it on shutdown.

Verifier: services.py:12 creates genai.Client inside the function. google-genai v2.4.0 BaseApiClient eagerly builds SyncHttpxClient/AsyncHttpxClient in __init__ and defines close()/aclose()/__del__. So each request pays client construction and a fresh TLS connection. Confirmed low.

## [low][backend][confirmed] Missing GOOGLE_API_KEY not detected at startup; failure surfaces per request as a leaked 500
File: src/backend/app/config/settings.py:132
Category: devops

GOOGLE_API_KEY defaults to an empty string. google-genai then falls back to env vars and raises ValueError('No API key was provided...') inside the request, which the view returns as a 500 with that text. Local (non-Docker) runs never load the root .env (no python-dotenv or similar), so `python manage.py runserver` fails until the variable is exported manually. None of this is documented.

Evidence: settings.py:132 `GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")`; _api_client.py (v2.4.0): `if not self.api_key: raise ValueError('No API key was provided...')`; views.py:26-30 returns str(e).

Fix: Validate required config at startup (pydantic-settings BaseSettings in FastAPI, or ImproperlyConfigured in Django when not DEBUG). Load .env for local dev (python-dotenv / django-environ) and document the setup in the README.

Verifier: settings.py:132 defaults to an empty string. The v2.4.0 _api_client.py raises ValueError('No API key was provided. Please pass a valid API key...'), and the finder's quote is accurate. genai.Client is constructed inside the try in services.py, so this returns a 500 with that text. There is no dotenv loading for local runs. Compose interpolates ${GOOGLE_API_KEY} from the root .env, so the Docker path works. Confirmed low.

## [low][backend][confirmed] Model id hardcoded and not configurable; dependency pins behind current releases
File: src/backend/app/core/services.py:27
Category: devops

Checked 2026-09-13: 'gemini-3.5-flash' is a valid GA model id (ai.google.dev model page, 1,048,576 input / 65,536 output tokens), and google-genai 2.4.0 is a real release (GitHub tag v2.4.0, 2026-05-18), so the call is not broken. However, the model is a literal and cannot be switched via env for experiments. The SDK is 19 minor versions behind (v2.23.0, 2026-09-10; v2.22.0 added Gemini 3.8 Flash). Django is pinned at 5.2.14 while the 5.2 line is at 5.2.18. There is no lock file or hashes, so transitive deps (httpx, pydantic, tenacity) float.

Evidence: services.py:27 `model="gemini-3.5-flash"`; requirements.txt:1 `django==5.2.14`, :3 `google-genai==2.4.0`; github.com/googleapis/python-genai/releases lists v2.23.0 (2026-09-10); docs.djangoproject.com/en/dev/releases lists 5.2.18.

Fix: Add GEMINI_MODEL to settings/env with the current id as default. Bump Django to the latest 5.2.x security release. Use pip-tools/uv to produce a locked requirements file, and add Dependabot/Renovate.

Verifier: Web checks agree. ai.google.dev lists gemini-3.5-flash as stable with 1,048,576/65,536 tokens, so the call is not broken. The raw v2.4.0 tag source is fetchable, so that release exists. The GitHub releases page shows v2.23.0 on 2026-09-10, and v2.22.0 on 2026-09-02 added Gemini 3.8 Flash. The Django releases index lists 5.2.18 as the latest 5.2.x, while the repo pins 5.2.14. The model literal is hardcoded and there is no lock file. Being four Django patch releases behind means missing security fixes, which is arguably the most actionable part.

## [low][backend][confirmed] Generation config ignores finish_reason, output limits and thinking level; history rebuilt without thought signatures
File: src/backend/app/core/services.py:28
Category: code-quality

GenerateContentConfig sets only system_instruction: no max_output_tokens (the 150-word limit exists only in the prompt), no thinking_level (Gemini 3.5 Flash defaults to 'medium' thinking, which adds latency and cost for short narration), and no safety settings. History is rebuilt from plain text, so thought signatures from earlier model turns are dropped. Google's Gemini 3.x guidance is to pass full history with signatures, and signatures become mandatory once tool/function calling (e.g. a retrieval tool in LangGraph) is added.

Evidence: services.py:28-30 `config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)`; services.py:17-21 `types.Content(role=..., parts=[types.Part(text=msg["content"])])` rebuilt from client strings.

Fix: Set max_output_tokens and an explicit thinking_config/thinking_level suited to narration. Check candidates[0].finish_reason. Keep model turns server-side (full Content objects or the LangChain AIMessage with its metadata) rather than rebuilding them from client-supplied text.

Verifier: The config sets only system_instruction. Google docs confirm Gemini 3.5 Flash defaults to thinking_level 'medium'. Model turns are rebuilt from plain strings, so thought signatures are lost. For plain text chat without function calling, Gemini does not strictly validate signatures (strict enforcement applies to function calling), so today this is a quality and latency concern, not a failure. Low is correct.

## [low][backend][confirmed] CORS origins hardcoded to Vite dev ports only
File: src/backend/app/config/settings.py:134
Category: devops

Only http://localhost:5173 and http://127.0.0.1:5173 are allowed, and they cannot be set via env. Opening the frontend from a LAN IP, `vite preview` (4173), or any deployed domain fails CORS with no config-only fix. The compose frontend works only because the host browser still reports origin localhost:5173.

Evidence: settings.py:134-137 CORS_ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]; docker-compose.yml:11 VITE_API_URL=http://localhost:8000.

Fix: Read CORS_ALLOWED_ORIGINS from an env var (comma-separated). Alternatively, avoid CORS in dev with a Vite proxy (server.proxy '/api' → http://backend:8000) and add 'backend' to ALLOWED_HOSTS.

Verifier: settings.py:134-137 hardcodes only the :5173 localhost origins and nothing is read from env. Compose works because the browser origin is still localhost:5173. Confirmed low.

## [low][backend][found-by-verifier] pip cache left in image layer; no --no-cache-dir
File: src/backend/Dockerfile:12
Category: devops

The pip install runs as root without --no-cache-dir. `rm -rf /tmp` removes /tmp, but the downloaded wheels in /root/.cache/pip stay in the layer and grow the image for no benefit (more so once LangChain/LangGraph/qdrant-client are added).

Evidence: Dockerfile:10-13 `/py/bin/pip install --upgrade pip && /py/bin/pip install -r /tmp/requirements.txt && rm -rf /tmp`, with no --no-cache-dir and no PIP_NO_CACHE_DIR env.

Fix: Use `pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt`, or set ENV PIP_NO_CACHE_DIR=1, and combine with the layer reordering from BE-19.

Verifier: 

## [low][backend][found-by-verifier] Empty or whitespace-only last message forwarded to Gemini, and empty assistant content causes a stuck conversation
File: src/backend/app/core/services.py:24
Category: bug

The view accepts `{"role":"user","content":""}` because only role is checked. services.py builds types.Part(text=""), which the Gemini API rejects as an empty part (400 INVALID_ARGUMENT). That returns a 500 with the upstream JSON (see BE-05). Combined with BE-06, one bad turn left in client history makes every later request fail until the user starts a new game, because the server resends the whole history each time.

Evidence: views.py:17 checks only `messages[-1].get("role")`; services.py:24 `last_message = messages[-1]["content"]` is used directly in types.Part(text=last_message); App.jsx:21 appends the reply to history unconditionally.

Fix: Validate content as a non-blank string (strip) for every message in the serializer or Pydantic model. Never append empty or None assistant content on the client or server. Return 400 for blank input.

Verifier: 

## [low][frontend][confirmed] The WELCOME narration is never sent to the model and disappears from the transcript after the first message
File: src/frontend/src/components/ChatWindow.jsx:21
Category: bug

WELCOME is rendered only while `messages.length === 0` and is never added to `messages`. Two problems follow. (1) The player's first action responds to text the model never saw ('The fire crackles...', 'threshold'), so the GM's first reply can contradict the opening scene. (2) Once the first message is sent, the condition is false and the welcome bubble vanishes, so the transcript starts with a user message that has no GM prompt before it. Also, the backend requires the last message to be from the user but has no rule for the first one, so a seeded assistant turn would be accepted. Gemini does accept a model turn first, but check this for the chosen SDK.

Evidence: ChatWindow.jsx:21-25 `{messages.length === 0 && (<MessageBubble message={{ role: 'assistant', content: WELCOME }} />)}`; App.jsx:14 `const updated = [...messages, userMsg]` has no welcome turn.

Fix: Seed the conversation with the welcome as a real assistant message (`useState([WELCOME_MSG])`, and reset to it in handleNewGame) so it stays on screen and is sent as history. Better: have the backend or graph generate the opening scene from the selected scenario (RAG over the scenario corpus), so it is grounded and not hard-coded.

Verifier: ChatWindow.jsx:21-25 renders WELCOME only when messages is empty, and it never goes into history. The bubble disappears after the first send and the model never sees the opening text. The impact is small: the welcome is generic ('threshold', fire), so contradictions are cosmetic. Whether Gemini accepts a model turn first was not verified, and the finder hedged on it too.

## [low][frontend][confirmed] Error display lives in the header: raw backend text, no dismiss or retry, not announced, can overflow the header
File: src/frontend/src/App.jsx:47
Category: ux

Errors are shown as a small pill on the right of the header, far from the input and the failed message. Nothing dismisses it: it clears only on the next send or New Campaign. There is no retry. The text is whatever the backend returned. views.py returns `str(e)` from the Gemini SDK, which can be a long JSON blob, and a network failure shows the browser's 'Failed to fetch' / 'NetworkError when attempting to fetch resource'. The <p> has no max-width, truncation or break-words, and the header is `justify-between` with a flex title, so a long message squeezes or overflows the header. It has no role='alert', so screen-reader users are never told the request failed.

Evidence: App.jsx:47-51 `{error && (<p className="text-sm text-red-400 ...">{error}</p>)}` inside the header; api.js:10-12 `throw new Error(err.error || `HTTP ${res.status}`)`; a fetch rejection passes through unchanged; backend views.py returns `{"error": str(e)}` with status 500.

Fix: Show the error inline under the failed message or above the input, with role='alert', a dismiss button and a Retry action (see FE-03). Map errors in api.js to friendly text (network down, 4xx, 5xx, timeout) and log the details to the console. On the backend, return a stable error code and message instead of str(e).

Verifier: The error is a <p> in the header (App.jsx:47-51) with no role=alert, no dismiss and no retry, and views.py returns str(e). One claim is slightly off: the title is a plain div, not a flex container. In a justify-between flex row a long error wraps and makes the header taller rather than overflowing, unless the string has long unbroken runs. The retry and alert parts overlap FE-03 and FE-16. Low is more accurate.

## [low][frontend][confirmed] Model markdown renders <img> and <a> unrestricted: external image loads (exfiltration/tracking vector) and links that navigate away from the SPA
File: src/frontend/src/components/MessageBubble.jsx:29
Category: security

react-markdown 9 is safe by default against XSS. Raw HTML is not rendered as HTML (it is turned into text), and defaultUrlTransform allows only http(s), irc(s), mailto and xmpp, so javascript: URLs are stripped. That part is correct and does not need changing. However, `![](https://host/x?q=...)` still renders an <img> that the browser loads automatically. This is the known markdown-image exfiltration channel for LLM apps: once RAG adds retrieved scenario/lore text (possibly untrusted PDFs or web content) to the prompt, an injected instruction can make the model emit an image URL carrying conversation data, and it is fetched with no click. Links render as plain <a href> with no target or rel, so clicking one replaces the app in the same tab and destroys the in-memory campaign (FE-04). Tailwind preflight also makes links look like body text (FE-09).

Evidence: MessageBubble.jsx:29 `<ReactMarkdown>{message.content}</ReactMarkdown>` with no `components`, `disallowedElements` or `urlTransform`. react-markdown 9.0.1 lib/index.js: `safeProtocol = /^(https?|ircs?|mailto|xmpp)$/i`, raw nodes converted to `{type: 'text', value: node.value}`; img has no special handling.

Fix: `<ReactMarkdown disallowedElements={['img']} unwrapDisallowed components={{ a: (p) => <a {...p} target="_blank" rel="noopener noreferrer nofollow" /> }}>`. Alternatively allow images only from an allowlist via urlTransform. Also add a CSP (img-src 'self') in production.

Verifier: Checked against the react-markdown 9.0.1 source. Raw HTML becomes text, safeProtocol is /^(https?|ircs?|mailto|xmpp)$/i, and img gets no special handling, so https image URLs render and load automatically. Links have no target, so clicking one navigates the SPA away and loses state. There is no untrusted input today (no RAG yet), and the only data that could leak is the user's own chat, so this is low now. It becomes medium once RAG over untrusted documents is added.

## [low][frontend][confirmed] GM markdown styling covers only p/strong/em; Tailwind preflight strips list bullets, heading sizes, link styling and code blocks, and <pre> can overflow the bubble
File: src/frontend/src/index.css:28
Category: ux

Tailwind v3 preflight resets `ul, ol {list-style:none; padding:0}`, `h1-h6 {font-size:inherit; font-weight:inherit}`, `a {color:inherit; text-decoration:inherit}`, and blockquote margins. .prose-gm restores only p margins, strong and em. LLM GMs often produce lists (inventory, options), headings and occasionally tables or code. Here lists render as unbulleted runs of lines, headings look like body text, links cannot be told apart from text, and a <pre> with long lines breaks out of the max-w-[85%] bubble (no overflow-x). Without remark-gfm, tables and strikethrough show as raw pipes and tildes. The last <p> also adds 0.75rem of extra bottom padding inside every bubble.

Evidence: index.css:28-41 defines only `.prose-gm p`, `.prose-gm strong`, `.prose-gm em`; tailwind.config.js:44 `plugins: []` (no @tailwindcss/typography); package.json has no remark-gfm; prompts.py asks for markdown.

Fix: Add @tailwindcss/typography with a custom dark 'gm' theme (`prose prose-invert`), or extend .prose-gm with ul/ol/li/h1-h3/a/blockquote/code/pre styles (`pre{overflow-x:auto}`, `p:last-child{margin-bottom:0}`). Add remark-gfm if tables are wanted.

Verifier: Tailwind v3 preflight removes list styles, heading sizes and link styling. .prose-gm covers only p, strong and em, and the last <p> keeps a 0.75rem bottom margin. prompts.py tells the model to use only **bold** and *italic* and to stay under 150 words, so lists, headings and code blocks will be uncommon. The bug is real but the impact is small.

## [low][frontend][confirmed] Input is disabled while the GM replies: focus is lost after every turn and the player cannot draft the next action
File: src/frontend/src/components/MessageInput.jsx:31
Category: ux

handleSend calls `textareaRef.current?.focus()` right after onSend, while the textarea is still enabled. The re-render then sets `disabled={loading}`, and the browser drops focus from a disabled control. When loading ends nothing restores focus, so after every GM reply the player has to click back into the box. Disabling the textarea (not just the send button) also blocks typing the next action during a multi-second LLM call.

Evidence: MessageInput.jsx:11-13 `onSend(text); setValue(''); textareaRef.current?.focus()`; MessageInput.jsx:31 `disabled={disabled}` on the textarea; App.jsx:55 `disabled={loading}`.

Fix: Keep the textarea enabled (or use readOnly/aria-busy) and disable only submission. Or re-focus in a `useEffect(() => { if (!disabled) textareaRef.current?.focus() }, [disabled])`.

Verifier: The textarea gets disabled={disabled}, and nothing restores focus when loading ends. Whether focus is actually dropped depends on the browser: Firefox blurs, and Chromium applies focus fixup in recent versions. Being unable to type the next action during generation is certain. This is a minor UX annoyance, so low.

## [low][frontend][confirmed] Low-contrast text: hint, placeholder and labels fail WCAG AA
File: src/frontend/src/components/MessageInput.jsx:47
Category: ux

ash-600 (#524938) on the ink or ash-900 background is about 2.1:1, and ash-500 (#6e6252) is about 3.1:1. Both are below the 4.5:1 AA minimum for normal text, and they are used on small text-xs copy.

Evidence: MessageInput.jsx:47 `text-xs text-ash-600` (keyboard hint); MessageInput.jsx:34 `placeholder:text-ash-600`; App.jsx:45 `text-xs text-ash-500` (subtitle); Sidebar.jsx:23 `text-xs ... text-ash-500` ('Campaigns'). Palette in tailwind.config.js:21-25.

Fix: Use ash-400 (#8f8370, about 4.9:1 on #1a1410) or lighter for secondary text and placeholders.

Verifier: Recomputed the contrast ratios. #524938 is 2.06:1 on #1a1410 and 2.15:1 on #130f08. #6e6252 is 3.07:1 and 3.21:1. #8f8370 on #1a1410 is 4.91:1. All match the finding.

## [low][frontend][confirmed] Vite HMR likely does not detect edits through a Windows bind mount (no polling configured)
File: src/frontend/vite.config.js:6
Category: devops

docker-compose bind-mounts ./src/frontend into the container, and the host is Windows. File change events from an NTFS path into a Docker Desktop (WSL2) Linux container are generally not delivered, and Vite's docs recommend `server.watch.usePolling` for this setup. Without it, edits on the host may not trigger HMR inside the container, so the dev container shows stale code. Not reproduced here, so this is plausible rather than confirmed.

Evidence: docker-compose.yml:7-9 `volumes: - ./src/frontend:/app - /app/node_modules`; vite.config.js:6-9 has no `watch` option; environment is Windows 11.

Fix: `server: { host: '0.0.0.0', port: 5173, watch: { usePolling: process.env.CHOKIDAR_USEPOLLING === 'true' } }` and set that env var in compose, or keep the repo inside the WSL filesystem.

Verifier: Docker Desktop's WSL2 best-practices docs say Linux containers receive inotify events only for files stored in the Linux filesystem. The bind mount is ./src/frontend on NTFS, and vite.config has no watch.usePolling. With the default WSL2 backend, HMR will likely miss host edits. Not reproduced.

## [low][frontend][confirmed] Unused dependencies (clsx, tailwind-merge; @types/* without TypeScript) and no lint or test tooling
File: src/frontend/package.json:16
Category: code-quality

clsx and tailwind-merge are declared but never imported. Sidebar.jsx:30-34 builds conditional classes with a template literal, which is exactly what they are for. @types/react and @types/react-dom are installed in a pure .jsx project with no tsconfig or jsconfig. There is no ESLint (eslint-plugin-react-hooks would flag stale-closure patterns like FE-02), no test runner, and no `lint` or `test` scripts.

Evidence: package.json:16-17 `"clsx": "^2.1.1", "tailwind-merge": "^2.5.2"`; grep for `clsx|twMerge|tailwind-merge` in src/frontend/src returns nothing; package.json:6-10 scripts are only dev/build/preview.

Fix: Either use them (`lib/cn.js`: `export const cn = (...a) => twMerge(clsx(a))`, and use it in Sidebar) or remove them. Add eslint with react-hooks and jsx-a11y, plus vitest and @testing-library/react with tests for handleSend error and cancel paths.

Verifier: Grep for clsx, twMerge and tailwind-merge in src finds nothing. @types/* are installed with no TypeScript config. The only scripts are dev, build and preview.

## [low][frontend][confirmed] Body background (#130f08, ash-900) differs from the app root bg-ink (#1a1410); theme colors duplicated as raw hex in CSS
File: src/frontend/src/index.css:11
Category: code-quality

body is painted ash-900 while the root container uses bg-ink, a slightly lighter brown. It shows during initial load before React mounts (a flash from #130f08 to #1a1410) and in overscroll/rubber-band areas on macOS and iOS. index.css also hard-codes palette hex values that duplicate tailwind.config.js, so changing the theme will not update them. The ::-webkit-scrollbar styling does not apply in Firefox.

Evidence: index.css:11 `background-color: #130f08;` vs App.jsx:35 `bg-ink` = tailwind.config.js:13 `ink: '#1a1410'`; index.css:23 `#38301f` (= ash-700), :34 `#c0522a` (= ember), :38 `#afa594` (= ash-300).

Fix: Use `@apply bg-ink text-parchment font-body` on body, and `theme('colors.ember')` etc. in .prose-gm and the scrollbar rules. Add `scrollbar-width: thin; scrollbar-color: theme('colors.ash.700') transparent` for Firefox.

Verifier: index.css:11 sets body to #130f08 (ash-900), while the root uses bg-ink #1a1410. The hex values duplicate the Tailwind palette, and the scrollbar styling is WebKit-only. All cosmetic.

## [low][frontend][confirmed] Auto-growing textarea depends on field-sizing: content; older browsers get a fixed one-row box
File: src/frontend/src/components/MessageInput.jsx:36
Category: ux

`style={{ fieldSizing: 'content' }}` is the only auto-grow mechanism. As of mid-2026 it is reported as Baseline 'newly available': Chromium 123+, Safari 26.2+, Firefox 152+ (June 2026). Older Firefox and Safari ignore it, and with `rows={1}` the player gets a single-line textarea that scrolls internally for multi-line actions. It degrades gracefully, but it is cramped. max-h-36 only has an effect where field-sizing works.

Evidence: MessageInput.jsx:33 `rows={1}`; MessageInput.jsx:36 `style={{ fieldSizing: 'content' }}`.

Fix: Keep field-sizing but add a JS fallback when `!CSS.supports('field-sizing', 'content')`: on input, set `el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 144) + 'px'`.

Verifier: MDN lists field-sizing as Baseline 2026 newly available (since June 2026). The exact per-browser versions were not verified, but older Safari and Firefox get a one-row scrolling textarea. It degrades gracefully.

## [low][frontend][confirmed] Auto-scroll always jumps to the bottom: long replies must be scrolled back up and reading older turns is interrupted
File: src/frontend/src/components/ChatWindow.jsx:14
Category: ux

Every change to messages or loading smooth-scrolls to the end marker. For a long GM narration the view lands at its last line, not its first. A player who scrolled up to reread earlier clues gets pulled to the bottom when the reply arrives. With streaming (FE-14) this effect would fire on every token and make scrolling up impossible.

Evidence: ChatWindow.jsx:14-16 `useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])`.

Fix: Track whether the user is near the bottom (onScroll threshold) and auto-scroll only then. When a new assistant message is added, scroll its top into view (`block: 'start'`). Show a 'New message' jump button otherwise.

Verifier: ChatWindow.jsx:14-16 scrollIntoView runs unconditionally on every change to messages or loading.

## [low][frontend][confirmed] Array-index keys for messages
File: src/frontend/src/components/ChatWindow.jsx:28
Category: code-quality

`key={i}` is harmless today because messages are append-only and cleared all at once. It becomes a real bug as soon as the planned features arrive (removing a failed message for FE-03, regenerate or edit, streaming placeholders, loading a different campaign): React would reuse bubble instances and the fade-in animation for the wrong content.

Evidence: ChatWindow.jsx:27-29 `{messages.map((msg, i) => (<MessageBubble key={i} message={msg} />))}`.

Fix: Assign `id: crypto.randomUUID()` when creating each message in App.jsx (or use server message ids from the LangGraph thread) and use `key={msg.id}`.

Verifier: key={i} at ChatWindow.jsx:28. It is harmless while the list is append-only, as the finder says.

## [low][frontend][confirmed] Enter sends during IME composition
File: src/frontend/src/components/MessageInput.jsx:17
Category: bug

The keydown handler does not check `isComposing`. For users of IME input methods (CJK, some mobile keyboards), pressing Enter to confirm a composition also submits the half-written message.

Evidence: MessageInput.jsx:16-21 `if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }`.

Fix: `if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing)`.

Verifier: MessageInput.jsx:17 checks only key and shiftKey, not isComposing. Note that Safari may report isComposing=false on the confirming Enter, so a keyCode 229 check helps there too.

## [low][frontend][confirmed] index.html: no favicon, fonts loaded from an external CDN with no local fallback
File: src/frontend/index.html:9
Category: ux

There is no <link rel="icon">, so every load requests /favicon.ico and gets a 404 plus the default browser icon. All three fonts come from fonts.googleapis.com. Offline or firewalled (a common setup at thesis demos), Cinzel and Crimson Pro fall back to generic serif, and the page leaks visitor IPs to Google, which matters under GDPR in the EU. No theme-color or meta description.

Evidence: index.html:4-9 has only charset, viewport, title and Google Fonts preconnect/stylesheet links; no icon link.

Fix: Add a favicon (e.g. public/favicon.svg with <link rel="icon">) and `<meta name="theme-color" content="#1a1410">`. Self-host fonts via @fontsource/cinzel and @fontsource/crimson-pro.

Verifier: index.html has no icon link and loads fonts only from Google Fonts. Minor.

## [low][frontend][found-by-verifier] Display font Cinzel has no Cyrillic glyphs, so Ukrainian campaign or scenario titles fall back to generic serif
File: src/frontend/tailwind.config.js:7
Category: ux

The author is Ukrainian and the thesis is in Ukrainian. Once campaign names and headers become data-driven (FE-06, FE-27), any Cyrillic title in font-display (the header h1 and the sidebar logo) renders in the browser's fallback serif, mixed with Cinzel's Latin letters. Fontsource lists Cinzel with only the Latin subset. The page also declares lang="en", so screen readers will mispronounce Ukrainian GM text.

Evidence: tailwind.config.js:7 `display: ['"Cinzel"', 'serif']`; App.jsx:42 `font-display` on the campaign title; index.html:2 `<html lang="en">`.

Fix: Choose a display face that includes Cyrillic (for example Cormorant SC or Philosopher), or give font-display a Cyrillic-capable fallback with a matching style. Set lang per UI locale.

Verifier: 

## [low][frontend][found-by-verifier] .env.example does not document VITE_API_URL or any frontend configuration
File: .env.example:1
Category: devops

.env.example has only GOOGLE_API_KEY. The frontend URL is configured only inline in docker-compose.yml and through the api.js fallback. Someone running `npm run dev` outside Docker, or pointing at a remote backend, cannot find the variable.

Evidence: .env.example content: `GOOGLE_API_KEY=your-api-key`; api.js:1 `import.meta.env.VITE_API_URL || 'http://localhost:8000'`.

Fix: Add `VITE_API_URL=` (or API_PROXY_TARGET if the proxy approach from FE-18 is used) with a comment, possibly in a src/frontend/.env.example.

Verifier: 

## [low][infra][confirmed] Non-root user cannot write /app: SQLite DB and migrations break without the bind mount; migrate never run
File: src/backend/Dockerfile:6
Category: bug

The app is copied into /app as root, with no chown, and the container then runs as django-user. SQLite is at BASE_DIR/'db.sqlite3' = /app/db.sqlite3, so in the built image (no bind mount) 'manage.py migrate' fails with 'unable to open database file'. No Dockerfile or compose step runs migrate, and the host db.sqlite3 is 0 bytes, so the tables have never been created. /admin/ is routed (config/urls.py) and would fail with 'no such table'. The DB also lives inside the source bind mount instead of a named volume. On Linux hosts the bind-mounted dir's UID will not match django-user either.

Evidence: Dockerfile:6 'COPY ./app /app' (root-owned), Dockerfile:21 'USER django-user'; settings.py:82-87 sqlite at BASE_DIR / 'db.sqlite3'; 'ls -la src/backend/app' -> 'db.sqlite3' size 0; docker-compose.yml has no command/entrypoint running migrate.

Fix: Same as proposed. Dropping admin, auth, sessions and SQLite is the simplest option while the service is a stateless LLM proxy.

Verifier: Dockerfile:6 copies as root and :21 switches to USER django-user with no chown, so in the image alone /app is not writable and SQLite writes fail. db.sqlite3 is 0 bytes and nothing runs migrate. In practice the impact is small. In the author's compose flow the Windows bind mount is writable: the 0-byte db.sqlite3 was created by the container's runserver migration check. The only endpoint, POST /api/chat/, never touches the DB, because DRF's anonymous SessionAuthentication path makes no session query when there is no session cookie. Only the unused /admin/ would fail. Low, not medium.

## [low][infra][confirmed] No .dockerignore in either build context
File: src/backend/Dockerfile:6
Category: devops

Without a .dockerignore, everything in the build context goes into the image. The backend 'COPY ./app /app' copies the host's __pycache__ (cpython-310 .pyc files exist on disk) and the local db.sqlite3. The frontend 'COPY . .' copies host node_modules once it exists, and after a Windows 'npm install' those contain Windows-native binaries (esbuild/rollup) that overwrite the Linux ones installed in the previous layer and break Vite in the container. It would also copy any future src/frontend/.env.local. The root .env is outside both contexts, which is good.

Evidence: 'ls .dockerignore src/backend/.dockerignore src/frontend/.dockerignore' -> No such file; on disk: src/backend/app/config/__pycache__/*.cpython-310.pyc, src/backend/app/core/__pycache__/*, src/backend/app/db.sqlite3; src/frontend/Dockerfile:8 'COPY . .'.

Fix: Same as proposed: add a .dockerignore to each build context.

Verifier: There is no .dockerignore in the root, src/backend or src/frontend. __pycache__/*.cpython-310.pyc and db.sqlite3 exist under src/backend/app. src/frontend/Dockerfile:8 is `COPY . .`. Two things lower the impact. In the only runtime path, compose, both contexts are masked by bind mounts and the node_modules anonymous volume, so the copied image contents never run. The claim that host Windows node_modules would definitely 'break Vite' is also overstated: platform packages such as @esbuild/linux-x64 and @rollup/rollup-linux-x64-musl from the install layer survive a merge-COPY, so breakage depends on version skew. Worth adding, but low.

## [low][infra][confirmed] Backend Dockerfile layering and cleanup anti-patterns (cache busting, pip cache, 'rm -rf /tmp')
File: src/backend/Dockerfile:13
Category: devops

The app source is copied before the dependency install. Any code edit therefore invalidates the layer cache and triggers a full pip reinstall on every 'docker compose build', which will get painful once langgraph, qdrant-client and similar packages are added. pip runs without --no-cache-dir, so the pip cache is baked into the layer. 'rm -rf /tmp' deletes the /tmp directory itself rather than the requirements file. Tools that assume /tmp exists (tempfile-heavy libraries, gunicorn worker tmp files, HF downloads) then either fall back unexpectedly or fail. There is also no PYTHONDONTWRITEBYTECODE, so .pyc files get written back into the bind-mounted host source.

Evidence: Dockerfile:5-6 'COPY ./requirements.txt /tmp/requirements.txt' then 'COPY ./app /app' before Dockerfile:10-12 'RUN python -m venv /py && /py/bin/pip install --upgrade pip && /py/bin/pip install -r /tmp/requirements.txt'; Dockerfile:13 'rm -rf /tmp && \'.

Fix: Same as proposed.

Verifier: Dockerfile:6 `COPY ./app /app` comes before the pip RUN at :10-12, so any code change busts the pip layer. pip runs as root without --no-cache-dir, which bakes /root/.cache/pip into the image. :13 has `rm -rf /tmp`. There is no PYTHONDONTWRITEBYTECODE, and cpython-310 .pyc files exist in the bind-mounted source. The /tmp deletion is milder than stated: Python's tempfile falls back to /var/tmp, then cwd. Low is correct.

## [low][infra][confirmed] '--no-create-home' user has a nonexistent HOME: model/tool caches will fail once RAG deps land
File: src/backend/Dockerfile:16
Category: devops

Running as non-root with a root-owned venv at /py is good: the app cannot modify its own packages. But because of --no-create-home, django-user's HOME (/home/django-user) does not exist. Libraries that write to ~/.cache will crash with PermissionError/FileNotFoundError at runtime: huggingface_hub and sentence-transformers (for downloading nomic-embed-text), fastembed (qdrant-client), tiktoken, and langchain caches. The only writable location is currently nothing at all, since /app is root-owned (see D6).

Evidence: Dockerfile:14-17 'adduser --disabled-password --no-create-home django-user'; Dockerfile:21 'USER django-user'; no HOME/HF_HOME/XDG_CACHE_HOME set.

Fix: Same as proposed.

Verifier: Dockerfile:14-17 uses `--no-create-home` and no HOME, HF_HOME or XDG_CACHE_HOME is set. /home/django-user does not exist, and /home is root-owned. Any library writing to ~/.cache would fail. Nothing does so today, so this is a future-facing low issue. It stays moot if embeddings are served by a separate Ollama container.

## [low][infra][confirmed] No healthchecks, restart policy or readiness ordering in compose
File: docker-compose.yml:12
Category: devops

depends_on only orders container start. There is no healthcheck on either service and no 'condition: service_healthy'. The backend has no health endpoint either: the only routes are POST /api/chat/ and /admin/. The first thing that needs readiness gating will be a Qdrant/Ollama service (the embedding model pull takes minutes). No restart policy is set.

Evidence: docker-compose.yml:12-13 'depends_on:\n  - backend' with no condition; no 'healthcheck:' or 'restart:' keys anywhere; core/urls.py only has path('chat/', chat).

Fix: Same as proposed.

Verifier: docker-compose.yml:12-13 has a plain depends_on with no healthcheck or restart keys. core/urls.py only has path('chat/', chat), and config/urls.py adds admin/. The claim is accurate. The frontend doesn't really need backend readiness, because the browser calls the backend directly. Low.

## [low][infra][confirmed] README claims features that do not exist and lacks setup instructions
File: README.md:9
Category: ux

README lists 'Persistent conversation history', but chat state is only React useState: there is no localStorage, no DB model and no history endpoint, and it is lost on reload or 'New Campaign'. The sidebar campaigns are a hardcoded array. The tech stack omits the LLM provider (Google Gemini). There are no setup/run instructions (cp .env.example .env, docker compose up), no required env vars, and no ports. Nothing mentions the thesis scope: RAG, Qdrant, fine-tuning, the Call of Cthulhu Keeper. The current system prompt is actually generic dark fantasy ('Bloodborne meets Witcher'). The project name also differs across places: 'AI GameMaster' in README, 'gm-assistant' in the package, 'GM Assistant' in the UI.

Evidence: README.md:9 '- Persistent conversation history'; App.jsx:8 'const [messages, setMessages] = useState([])' and App.jsx:29-32 handleNewGame clears it; grep for localStorage in src/frontend/src -> no matches; Sidebar.jsx:3-6 'const CAMPAIGNS = [ { id: 1, name: 'The Ashen Crown', active: true }, ...'; prompts.py:1 'Game Master for a dark fantasy tabletop RPG'.

Fix: Same as proposed.

Verifier: README.md:9 claims 'Persistent conversation history', but App.jsx:8 keeps history in useState, handleNewGame clears it, and grep finds no localStorage. Sidebar.jsx:3-6 has the hard-coded CAMPAIGNS array. prompts.py:1 says dark fantasy, not CoC. The README has no setup instructions, and the names differ ('AI GameMaster' in the README, 'GM Assistant' in Sidebar.jsx:16). The header in App.jsx:43 is also hard-coded 'The Ashen Crown'. All accurate, but this is a documentation issue, so low is more appropriate than medium.

## [low][infra][confirmed] .env.example is incomplete and Django does not load .env outside Docker
File: .env.example:1
Category: devops

.env.example documents a single variable. compose uses no env_file. GOOGLE_API_KEY reaches the container only through ${GOOGLE_API_KEY} interpolation, which works only when compose runs from the repo root, where it picks up .env for substitution. settings.py uses os.environ with no dotenv loading, so 'python manage.py runserver' on the host (the author's .vscode points at a conda env) silently gets an empty key. A missing key is not caught at startup. It only surfaces as a 500 whose body contains the raw exception string (views.py:26-29).

Evidence: .env.example: 'GOOGLE_API_KEY=your-api-key' (only line); docker-compose.yml:23 '- GOOGLE_API_KEY=${GOOGLE_API_KEY}'; settings.py:132 'GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")'; .vscode/settings.json 'python-envs.defaultEnvManager': 'ms-python.python:conda'.

Fix: Same as proposed.

Verifier: .env.example has a single line. Compose uses ${GOOGLE_API_KEY} interpolation with no env_file. settings.py:132 uses os.environ.get with no dotenv loading. views.py:26-29 returns str(e) with a 500. One small inaccuracy: Compose v2 reads .env from the project directory, which defaults to the directory of the compose file, not strictly the cwd. So `docker compose -f path/docker-compose.yml` from elsewhere still interpolates it (unless --project-directory overrides that).

## [low][infra][confirmed] Vite hot reload will not trigger in Docker Desktop on Windows with the repo on NTFS
File: src/frontend/vite.config.js:6
Category: devops

The repo lives at C:\Users\1\Documents\GitHub, on the Windows filesystem, and is bind-mounted into Linux containers. Docker's WSL 2 guidance says containers only receive inotify file-change events when the source files are on the Linux filesystem. Vite's watcher relies on those events, so edits on Windows will not trigger HMR unless polling is enabled. Django runserver's StatReloader polls, so the backend is unaffected.

Evidence: vite.config.js server: { host: '0.0.0.0', port: 5173 } with no watch.usePolling; docker-compose.yml:8 '- ./src/frontend:/app'; working dir C:\Users\1\Documents\GitHub\gm-assistant.

Fix: Set server.watch.usePolling (gated by env), or set CHOKIDAR_USEPOLLING=true in the frontend compose environment, or move the checkout into the WSL2 filesystem.

Verifier: Docker's WSL best-practices page states: "Linux containers only receive file change events, 'inotify events', if the original files are stored in the Linux filesystem." The repo is under C:\Users\1\Documents and bind-mounted, and vite.config.js has no watch.usePolling. Vite 5 (chokidar 3) also honours CHOKIDAR_USEPOLLING=true as an env-only fix. Django's StatReloader polls, so the backend is unaffected, as stated.

## [low][infra][confirmed] Coarse commits and no .gitattributes for line endings
File: .gitignore:202
Category: code-quality

The history has four commits, two of them large grab-bags ('backend + base' with 17 files, 'frontend + fixes' with 18 files mixing backend fixes and the whole frontend). This makes the thesis development trail hard to follow and hid D1: the api.js omission shipped unnoticed inside the 'frontend + fixes' commit. core.autocrlf=true and there is no .gitattributes, so the working tree mixes CRLF (Dockerfiles, compose, services.py, prompts.py, requirements.txt) and LF files. The index is LF, so the repo itself is fine. But future shell entrypoint scripts copied into Linux images from a CRLF checkout will fail with '/bin/sh^M: not found', and contributors on other OSes will get noisy diffs.

Evidence: git log --stat: 99f2e69 '17 files changed', 2799aaf '18 files changed' (includes settings.py, services.py, Dockerfile alongside all frontend files); git ls-files --eol: 'i/lf w/crlf' for src/backend/Dockerfile, docker-compose.yml, src/backend/app/core/services.py; 'git config core.autocrlf' -> true; no .gitattributes.

Fix: Same as proposed.

Verifier: git log --stat shows 99f2e69 with 17 files and 2799aaf with 18 files mixing backend and frontend. `git ls-files --eol` shows i/lf w/crlf for Dockerfiles, docker-compose.yml, services.py, prompts.py, core/urls.py and requirements.txt. core.autocrlf=true and there is no .gitattributes. The index is LF. The CRLF shell-script risk for future entrypoints is real, because build contexts use the working tree.

## [low][infra][found-by-verifier] CorsMiddleware placed after CommonMiddleware, and CommonMiddleware is listed twice
File: src/backend/app/config/settings.py:50
Category: bug

django-cors-headers says to put CorsMiddleware as high as possible, before CommonMiddleware or any middleware that can generate a response. Here the first CommonMiddleware (line 50) runs before CorsMiddleware (line 52), and CommonMiddleware appears again at line 53. Responses generated in CommonMiddleware.process_request, such as DisallowedHost from request.get_host() or the APPEND_SLASH handling of '/api/chat' without a slash (a RuntimeError in DEBUG for POST), are built before CORS processing. The browser then reports an opaque 'CORS error' instead of the real 400/500. This makes the D3 ALLOWED_HOSTS failures much harder to diagnose from the frontend.

Evidence: settings.py:47-57 MIDDLEWARE = ['...SecurityMiddleware', '...SessionMiddleware', 'django.middleware.common.CommonMiddleware', '...CsrfViewMiddleware', 'corsheaders.middleware.CorsMiddleware', 'django.middleware.common.CommonMiddleware', ...]

Fix: Remove the first CommonMiddleware entry and move 'corsheaders.middleware.CorsMiddleware' above CommonMiddleware (directly after SecurityMiddleware). Better still, remove CORS entirely with a same-origin /api proxy (Vite server.proxy in dev, nginx in prod).

Verifier: 

## [low][infra][found-by-verifier] Frontend container runs as root
File: src/frontend/Dockerfile:1
Category: devops

The backend image drops to a non-root user, but the frontend image never sets USER, so the Vite dev server (exposed on 0.0.0.0:5173) runs as root inside the container even though node images ship a 'node' user. With the ./src/frontend bind mount, any file the container writes (e.g. .vite cache, or a lockfile after `npm install` inside the container) is root-owned on Linux hosts.

Evidence: src/frontend/Dockerfile has no USER instruction; CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"].

Fix: Add `USER node` (and `COPY --chown=node:node`) or run with `user: "node"` in compose. For production, serve the static build from nginx-unprivileged.

Verifier: 


# Refuted

