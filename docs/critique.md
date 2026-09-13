# Critique of `final-architecture.md` (completeness and fact-check)

Date: 2026-09-13. Reviewed in full: `final-architecture.md` (§0–§7), `issues.md` (83 issues; the 36 critical/high/medium ones checked one by one), `research.md` (the relevant sections and all "Verified facts"), and `tezy.txt`.

Doubtful APIs were checked against these sources:
- **Local doc snapshots** in `scratchpad/docs/`: LangGraph streaming, checkpointers and use-graph-api; the Ollama integration.
- **Local LangGraph source:** `scratchpad/pregel_main.py`.
- **Library source fetched from GitHub `main` today:**
  - `langchain-google-genai`: `chat_models.py` and `_common.py`
  - `langgraph-checkpoint-sqlite`: `aio.py`
  - `langchain-ollama`: `chat_models.py` and `embeddings.py`
  - `langgraph/_internal/_retry.py`
  - `langchain_core/messages/utils.py`
- **Official documentation pages:**
  - FastAPI SSE tutorial
  - TRL 1.13 `SFTTrainer` docs
  - Unsloth Gemma 4 training docs
  - vLLM Gemma 4 recipe and supported-models page
  - Google Gemini API pricing page
- **Issue trackers:** vLLM GitHub issues #39246 and #41754.
- **Release registries:** PyPI JSON and the Docker Hub tag API.
- **CoC 7e community rule summaries:** philgamer "Let's Study CoC 7e", morganhua, rpgstack.

Confidence tags: **[verified]** means checked in source or official docs today. **[medium]** means secondary sources or inference.

---

## 1. Verdict summary

**Overall:** The plan is sound and unusually well-grounded. Most API claims hold up:
- The choice of FastAPI + LangGraph `StateGraph` + deterministic dice is right for a one-person thesis.
- All pins and images exist (checked list below).
- The key API claims match current docs and source:
  - `fastapi.sse` with POST
  - `version="v2"` StreamPart shapes
  - `add_node(timeout=…)`
  - `Runtime` / `context_schema`
  - the `nostream` tag
  - `AsyncSqliteSaver.from_conn_string` as an async context manager, plus `adelete_thread`
  - the `ChatOllama` fields
  - `qdrant-client` `query_points`
- The `OllamaEmbeddings` double-prefix bug the plan found is real. In `langchain-ollama` source, `embed_query` returns `self.embed_documents([text])[0]`.

**Pins and images confirmed on PyPI and Docker Hub today:**
- PyPI: `langgraph 1.2.11`, `langgraph-checkpoint-sqlite 3.1.1`, `fastapi 0.141.1`, `qdrant-client 1.19.0`, `trl 1.13.0`, `unsloth 2026.9.4`, `peft 0.20.0`, `langchain 1.4.0`, `langchain-core 1.6.3`, `langchain-google-genai 4.4.0` (requires `google-genai>=2.20,<3`), `uvicorn 0.52.4`, `vllm 0.29.0`.
- Docker Hub: `ollama/ollama:0.34.0`, `vllm/vllm-openai:v0.29.0`, `qdrant/qdrant:v1.19.1`.

**But it is not ready to implement as written.** Five defects would silently break the thesis results or the Day-10 demo. Each is fixed in §2.

1. **Crash-resume can re-roll dice (C1).** Default `durability="async"` does not guarantee the post-`mechanics` checkpoint is on disk when the process is killed. The Day-10 demo ("kill the API mid-turn → Retry without re-rolling") can fail.
2. **Timeouts are never retried (C2).** LangGraph's `default_retry_on` returns `False` for `OSError`, and `TimeoutError` (so `NodeTimeoutError`) is an `OSError` subclass. The same default *does* retry Gemini 4xx `ClientError`s.
3. **The SFT recipe would silently train on truncated or empty targets (C3).** TRL 1.13 `SFTConfig.max_length` defaults to **1024** with `truncation_mode="keep_start"`. RAFT examples (system prompt + sheet + 5 chunks + target) are far longer. Gemma 4 also uses `<|turn>user` / `<|turn>model` turn tags, not Gemma 3's `<start_of_turn>`.
4. **The FT arm may be a no-op on vLLM (C4).**
   - vLLM issue #41754 ("Unsloth LoRA adapters are ignored during inference despite successful loading", Gemma 4, opened 2026-05-05) is still open.
   - The vLLM Gemma 4 recipe recommends a 24 GB+ GPU for E4B at bf16.
   - The plan has no adapter-effect test and no fallback.
5. **The "paired design" is not actually paired (C7, C8).**
   - `adjudicate` receives `rules_hits` only in RAG arms, and the aux Gemini model is sampled at its fixed default (Gemini 3.6+ ignores `temperature`). So adjudication, and therefore which dice are rolled, differs between arms for the same item and seed.
   - The FT model never interprets mechanics in the main matrix. That removes one of the three FT benefits the thesis claims ("типові шаблони інтерпретації ігрових механік").

**Also wrong or incomplete, but less severe:**
- **The CoC rules code does not implement the flags the plan says it has (C5):**
  - no fumble-threshold flag;
  - bonus/penalty dice are capped before cancelling;
  - the INT roll for temporary insanity is flagged but never rolled;
  - indefinite insanity is not tracked;
  - a push can reference a roll from any earlier turn.
- **Duplicate `client_turn_id` corrupts history (C6).** `add_messages` replaces the message by id; the plan claims idempotency.
- **Context overflow (C10).** Ollama `num_ctx=8192` versus 6000 history tokens + RAG + sheet + output; `count_tokens_approximately` assumes 4 chars per token, which undercounts Cyrillic.
- **No client-side rate limiting or API budget (C11).**
- **Retrieval covers only `source_type="rules"` (C12).** Lore, scenarios and narrative examples from the thesis corpus never reach the main pipeline.
- **Explicit `google_api_key=None` disables the env-var fallback (C9).**

**Thesis coverage:** 8 of 11 components are fully covered, 3 partially (§3.1). The "no formal verification" limitation is only partly addressed. `[R#]` marker validation checks that a cited id exists, not that the chunk supports the claim. §5 proposes a verifier node that turns this limitation into a measurable contribution.

**MVP scope:** Still slightly too heavy for a Windows developer. Run the API and Vite natively and use Docker only for Qdrant/Ollama. Cut the Postgres branch, the `@tool` wrapper, Studio, the arm picker UI, the semaphore and `API_TOKEN` (§4).

---

## 2. Concrete corrections

Each entry gives: **Location** (section or code block in `final-architecture.md`) → **Problem** → **Evidence** → **Fix**. Ordered by impact.

### C1. Use `durability="sync"` for play turns so crash-resume never re-rolls dice [critical]

**Location:**
- §6.2 `run_turn` (`graph.astream(...)`)
- §6.2 `create_campaign` (`graph.ainvoke(...)`)
- §5 Day-10 demo ("kill the API mid-turn → Retry finishes the turn without re-rolling the dice")
- §7 smoke test 3
- §1.1 L3

**Problem:** Resume-from-checkpoint only skips `mechanics` if the checkpoint written after `mechanics` is durable. LangGraph's default durability is `"async"`, which writes checkpoints while the next step runs. A process kill during `narrate` can lose that write, so Retry re-runs `mechanics` with a new RNG draw whenever `seed=None` (the default in play). §7 test 3 only raises an exception inside `narrate`. That path flushes pending writes on exit, so the test passes while the demo's kill scenario can still fail.

**Evidence [verified]:**
- `docs/oss_python_langgraph_checkpointers.md`, "Durability modes": `"async"` has "a small risk that LangGraph does not write checkpoints if the process crashes during execution"; `"sync"` "persists changes synchronously before the next step starts".
- `pregel_main.py:2603`: `durability = config.get(CONF, {}).get(CONFIG_KEY_DURABILITY, "async")`.

**Fix:** Pass `durability="sync"` to every play-graph execution. The cost is about one SQLite write per step, which is negligible. Also make §7 test 3 kill a subprocess, not just raise.

```python
# server/routes.py — run_turn
async for part in graph.astream(
    inputs, config,
    context=KeeperContext(spec=spec, campaign_id=campaign.id),
    stream_mode=["updates", "messages", "custom"],
    version="v2",
    durability="sync",            # checkpoint after `mechanics` is on disk before `narrate` starts
):
    ...

# create_campaign
await st.graph.ainvoke({"bootstrap": {...}}, {"configurable": {"thread_id": campaign.id}},
                       context=KeeperContext(spec=spec, campaign_id=campaign.id), durability="sync")
```

```python
# tests/test_resume_crash.py (sketch): real crash, not an exception
# 1) start uvicorn in a subprocess with a fake generator model that sleeps 5 s inside narrate
# 2) POST /turns, wait for the `roll` SSE event, then proc.kill()
# 3) restart, POST /turns/retry, assert len(state.values["roll_log"]) == 1 and the roll equals the one seen before
```

---

### C2. The `RetryPolicy` never retries timeouts but does retry non-retryable 4xx [high]

**Location:**
- §6.1 `graph.py`: `LLM_RETRY = RetryPolicy(max_attempts=3)` on `analyze`, `retrieve`, `adjudicate`
- §4.3 row #11 ("graph-level `RetryPolicy` + `timeout=` on aux nodes")
- §1.1 L13

**Problem 1: timeouts.** A `RetryPolicy` without `retry_on` uses `default_retry_on`:
- It returns `False` for `ValueError, TypeError, …, RuntimeError, …, OSError`.
- `TimeoutError` is a subclass of `OSError`, so `NodeTimeoutError(TimeoutError)` is **never retried**, even though issue #11 is exactly about stalls.

**Problem 2: 4xx errors.** `google.genai.errors.ClientError` (400 invalid schema, 403 key, 429) is a plain `Exception` subclass. It falls through to `return True`, so bad requests are retried 3× on top of the SDK's own retries.

**Evidence [verified]:**
- `langgraph/_internal/_retry.py::default_retry_on` (fetched today):

  ```python
  if isinstance(exc, (ValueError, TypeError, ArithmeticError, ImportError, LookupError, NameError,
                      SyntaxError, RuntimeError, ReferenceError, StopIteration, StopAsyncIteration, OSError)):
      return False
  return True
  ```

- The use-graph-api doc: timed-out attempts are retried only "if the node has a `retry_policy` that retries `TimeoutError` or `NodeTimeoutError`".

**Fix:**
- Let the SDKs handle 429/5xx: `max_retries` exists on `ChatGoogleGenerativeAI` and `ChatOpenAI`. Note that for Google, `max_retries=1` means no retry.
- Let the graph retry only timeouts and transport errors.
- For the streamed `narrate` node, consider `TimeoutPolicy`.

```python
# keeper/graph.py
import httpx
from langgraph.errors import NodeTimeoutError
from langgraph.types import RetryPolicy, TimeoutPolicy

# SDK-level retries (max_retries on the chat model) already cover 429/5xx before any token is produced.
LLM_RETRY = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0,
                        retry_on=(NodeTimeoutError, ConnectionError, httpx.TransportError))

g.add_node("analyze",    n.analyze,    retry_policy=LLM_RETRY, timeout=30)
g.add_node("retrieve",   n.retrieve,   retry_policy=LLM_RETRY, timeout=20)
g.add_node("adjudicate", n.adjudicate, retry_policy=LLM_RETRY, timeout=45)
# narrate: no retry (tokens already streamed). A run cap plus an idle cap is kinder to long narrations;
# check the fault-tolerance doc for which events refresh idle_timeout before relying on it.
g.add_node("narrate", n.narrate, timeout=TimeoutPolicy(run_timeout=180, idle_timeout=45))
```

`server/errors.py::map_error` should map these codes (per issue #9):
- `google.genai.errors.APIError.code == 429` → `rate_limited` (retryable)
- `>= 500` → `upstream_unavailable` (retryable)
- other 4xx → `internal_error` (not retryable)

---

### C3. The Phase 4 SFT recipe would silently truncate examples; Gemma 4 template tags and thinking parity are missing [critical for the FT arm]

**Location:** §5 Phase 4 bullets ("Unsloth QLoRA on Gemma 4 E4B … `train_on_responses_only`"), §3.2 `training/train_unsloth.py` (no code given), §6.3 (train/serve parity claim).

**Problems:**
1. **Truncation.** TRL 1.13 `SFTConfig(max_length=1024, truncation_mode="keep_start")` is the default, and `keep_start` is the only supported mode. A RAFT example has the `keeper-v1` system prompt, the investigator JSON, 5 × ~1000-character chunks, history and the target. That is typically 2.5–4k tokens, so the **completion is cut off**. With response-only loss this can leave examples with zero supervised tokens. Training "succeeds" and learns nothing.
2. **Wrong turn tags.** `train_on_responses_only` needs Gemma 4's tags. Unsloth's Gemma 4 docs use `instruction_part="<|turn>user\n"` and `response_part="<|turn>model\n"`. Gemma 3 strings silently mask everything.
3. **Thinking-mode mismatch.** Unsloth's Gemma 4 guide uses the `gemma-4-thinking` chat template, and thinking is toggled with `enable_thinking` in `apply_chat_template`. If training and serving differ (vLLM's default template, Ollama `reasoning=False`), the "train/serve parity" claim in §6.3 breaks at the chat-template level even though the Jinja system prompt matches.

**Evidence [verified]:**
- TRL docs v1.13.0, `SFTConfig` signature: `max_length: int | None = 1024`, `truncation_mode: str = 'keep_start'`, `completion_only_loss: bool | None = None` (defaults to completion-only for prompt-completion datasets).
- Unsloth Gemma 4 training docs: `FastModel.from_pretrained`, `get_peft_model(finetune_vision_layers=False, …)`, `get_chat_template(tokenizer, chat_template="gemma-4-thinking")`, the response-masking strings above, and "E4B LoRA requires 17GB VRAM".

**Fix:** Commit a concrete `training/train_unsloth.py` with a length gate.

```python
# training/train_unsloth.py  (Colab/Kaggle/WSL2 CUDA env; pins from requirements-train.txt)
import json, statistics, sys
from datasets import load_dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer

SEED, MAX_LEN = 3407, 4096
model, tokenizer = FastModel.from_pretrained("unsloth/gemma-4-E4B-it", max_seq_length=MAX_LEN,
                                             load_in_4bit=True, full_finetuning=False)
model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                 finetune_attention_modules=True, finetune_mlp_modules=True,
                                 r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=SEED)
# Use the SAME template/thinking mode you will serve with. Check the exact name in
# unsloth.chat_templates.CHAT_TEMPLATES; the docs show "gemma-4-thinking".
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4-thinking")

def render(ex):   # ex["messages"] = [system, user, assistant] rendered by build_sft.py with keeper/prompts
    text = tokenizer.apply_chat_template(ex["messages"], tokenize=False, enable_thinking=False)
    return {"text": text.removeprefix("<bos>")}      # the trainer adds BOS itself (Unsloth notebook convention)

ds = load_dataset("json", data_files={"train": "sft/train.jsonl", "validation": "sft/val.jsonl"}).map(render)

# LENGTH GATE: refuse to train if anything would be truncated
lens = [len(tokenizer(t)["input_ids"]) for t in ds["train"]["text"]]
print("tokens p50/p95/max:", statistics.median(lens), sorted(lens)[int(.95 * len(lens))], max(lens))
if max(lens) > MAX_LEN:
    sys.exit(f"{sum(l > MAX_LEN for l in lens)} examples exceed {MAX_LEN} tokens; shorten context or raise MAX_LEN")

trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=ds["train"], eval_dataset=ds["validation"],
                     args=SFTConfig(dataset_text_field="text", max_length=MAX_LEN, per_device_train_batch_size=1,
                                    gradient_accumulation_steps=8, learning_rate=1e-4, num_train_epochs=2,
                                    lr_scheduler_type="cosine", warmup_steps=10, optim="adamw_8bit",
                                    logging_steps=10, eval_strategy="steps", eval_steps=50,
                                    seed=SEED, report_to="none", output_dir="out/keeper-lora"))
trainer = train_on_responses_only(trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n")

# Masking gate: at least one supervised token per example
lab = trainer.train_dataset[0]["labels"]
assert any(t != -100 for t in lab), "response masking removed every label: check turn tags"
trainer.train()
model.save_pretrained("adapters/keeper-lora"); tokenizer.save_pretrained("adapters/keeper-lora")
json.dump({"seed": SEED, "max_len": MAX_LEN, "data_sha256": "...", "template": "gemma-4-thinking/enable_thinking=False"},
          open("adapters/keeper-lora/train_meta.json", "w", encoding="utf-8"))
```

**Serving parity:** tell vLLM the same thinking mode on every request, and log it in the run record.

```python
ChatOpenAI(model=p.model, base_url=self.vllm_url, api_key="EMPTY", max_tokens=p.max_tokens, timeout=self.timeout,
           max_retries=2, extra_body={"chat_template_kwargs": {"enable_thinking": False}}, **opt)
```

---

### C4. vLLM + LoRA on Gemma 4: add an adapter-effect gate, a fallback, and realistic GPU flags [high]

**Location:**
- §3.5 compose `vllm` service
- §6.6 "Serving the fine-tuned model"
- §3.6 profiles (`gemma4-*-vllm`, "same bf16 precision")
- §5 Phase 4 export bullet
- §7 (missing test)

**Problems:**
1. **Adapters may be ignored.** vLLM added Gemma 4 LoRA (issue #39246, closed via PR #39291), and the supported-models page marks `Gemma4ForCausalLM` LoRA ✅. But issue **#41754** reports that Unsloth-trained Gemma 4 adapters load without error and **have no effect on outputs**. It is open with no fix linked. If this hits the author, the "FT" and "RAG+FT" arms equal the base arms and the thesis measures nothing.
2. **VRAM.** The vLLM Gemma 4 recipe recommends "1× GPU with 24 GB+ VRAM" for E4B at bf16 and shows `--max-model-len` and a text-only `--limit-mm-per-prompt image=0,audio=0` flag. The compose command sets neither. A typical Windows gaming GPU (8–16 GB) will OOM or reserve most of its memory for audio/vision encoders the thesis never uses.
3. **Precision mismatch.** An adapter trained against a 4-bit NF4 base (QLoRA) and served on a bf16 base is a small train/serve mismatch. It is acceptable, but it has to be stated; the plan's "same precision" wording is only true for base vs FT at serve time.

**Evidence:** GitHub vllm #41754 (open, 2026-05-05) and #39246 (closed) **[verified]**; vLLM Gemma 4 recipe and supported-models table **[verified]**.

**Fix:** Adapter-effect smoke test, a merged-weights fallback, and explicit flags.

```yaml
# compose.yaml — vllm (Linux or WSL2 only)
command: ["--model", "google/gemma-4-E4B-it", "--max-model-len", "12288",
          "--limit-mm-per-prompt", "image=0,audio=0",            # text-only (recipe flag; syntax varies by vLLM version)
          "--gpu-memory-utilization", "0.90", "--seed", "0",
          "--enable-lora", "--lora-modules", "keeper=/adapters/keeper-lora",
          "--max-lora-rank", "16", "--max-loras", "1"]
```

```python
# training/check_adapter_effect.py — must pass before any FT arm is run
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")
PROBES = [l.strip() for l in open("training/probes.txt", encoding="utf-8") if l.strip()]   # 20 held-out prompts

def gen(model, p):
    r = c.chat.completions.create(model=model, messages=[{"role": "user", "content": p}], temperature=0,
                                  max_tokens=200, seed=0,
                                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return r.choices[0].message.content

base = [gen("google/gemma-4-E4B-it", p) for p in PROBES]
ft   = [gen("keeper", p) for p in PROBES]
differs = sum(a != b for a, b in zip(base, ft)) / len(PROBES)
style   = sum("Mechanics:" in t for t in ft) / len(PROBES) - sum("Mechanics:" in t for t in base) / len(PROBES)
print(f"differs={differs:.0%} style_marker_gain={style:+.0%}")
assert differs >= 0.8, "adapter looks IGNORED (vllm #41754): use merged fallback"
```

**Fallback that keeps precision parity:**
- Export `model.save_pretrained_merged("models/keeper-merged-16bit", tokenizer, save_method="merged_16bit")`.
- Run **two** vLLM servers sequentially with identical flags: `google/gemma-4-E4B-it`, then `/models/keeper-merged-16bit`.
- Profiles differ only in `model`/`base_url`.
- Record `serving: lora|merged16` in the run record.

**If there is no 24 GB GPU:** run `evals/run.py` on a rented or Colab L4/A100 where vLLM runs locally. Evals need no streaming or UI, and the Gemini aux calls go over the internet. Do not force vLLM onto an 8–12 GB Windows card.

---

### C5. CoC 7e rules code: flag missing, cap order wrong, INT roll and indefinite insanity missing, push window unbounded [high for "code decides outcomes"]

**Location:** §6.4 `dice.py` (`RulesConfig`, `roll_d100`, `success_level`), `rules.py` (`percentile_check`, `can_push`, `sanity_check`, `resolve`), §5 Phase 1c ("config flags for the KRB ambiguities (fumble threshold base…)"), §6.4 tests.

**Findings:**

| # | What the plan's code does | Rule [confidence] | Problem |
|---|---|---|---|
| a | `success_level(roll, skill)` checks fumble with `skill < 50` only; `RulesConfig` has **no** fumble flag even though Phase 1c and §7.7 say there is one | KRB wording (quoted by secondary sources): a fumble is 96–100 "if the dice roll required for success is below 50". philgamer's "Let's Study" reads this as the *difficulty-adjusted target*. morganhua and rpgstack paraphrase it as base skill. [medium] | The ambiguity the plan identifies is not implemented, and `percentile_check` never passes difficulty, so a Hard roll at skill 60 (target 30) can never fumble on 96–99. |
| b | `roll_d100` clamps bonus and penalty **separately** before netting (`b=min(bonus,2)`, `p=min(penalty,2)`, `net=b-p`) | One bonus die cancels one penalty die (QSR) [verified]; at most 2 extra dice (KRB p.91) [medium]. research.md's reference implementation nets first, then caps. | With bonus 3 and penalty 1 the plan rolls with net +1, while netting first gives +2. The plan's code deviates from research.md without saying so. |
| c | `sanity_check` sets `int_roll_for_temporary_insanity: lost >= 5` but never rolls INT | Losing ≥5 SAN from one roll triggers an INT roll; **success** means temporary insanity for 1D10 hours (QSR/CWiki) [verified] | The outcome is left to the LLM narrator, which violates Principle 1 ("Code decides outcomes"). |
| d | No tracking of SAN lost per game day | Indefinite insanity when ≥ 1/5 of the day-start SAN is lost within a day (KRB via TRPGLine) [medium]; SAN 0 means permanent insanity [medium] | Silent omission. At least record `san_day_start` and `san_lost_today`, and let the Keeper prompt see the flag. |
| e | `can_push(prev)` uses `roll_log[-1]` from any earlier turn | A push is a re-roll of *that* failed roll, justified by in-fiction action right away (QSR) [verified] | A player can push a roll from 20 turns ago. |
| f | `sanity_loss` notation is chosen by the aux LLM | Scenario or KRB tables define SAN loss per horror [verified QSR notation] | This is outcome magnitude decided by the LLM. Prefer `scenario.yaml` values (`sanity_events: {ghoul_sighting: "0/1D6"}`) and fall back to the LLM only when absent (and log that). |

**Fix (drop-in replacements):**

```python
# keeper/game/coc7e/dice.py
@dataclass(frozen=True)
class RulesConfig:
    max_bonus_penalty: int = 2
    fumble_threshold_on: Literal["target", "base_skill"] = "target"   # KRB "roll required for success" (literal reading)
    fumbled_san_is_max_loss: bool = True

TARGET = {"regular": lambda s: s, "hard": lambda s: s // 2, "extreme": lambda s: s // 5}

def roll_d100(bonus: int = 0, penalty: int = 0, rng: random.Random | None = None, cfg: RulesConfig = DEFAULT_RULES) -> D100:
    rng = rng or random.SystemRandom()
    net = max(-cfg.max_bonus_penalty, min(max(0, bonus) - max(0, penalty), cfg.max_bonus_penalty))  # cancel, THEN cap
    units = rng.randint(0, 9)
    tens = tuple(rng.randint(0, 9) for _ in range(1 + abs(net)))
    totals = [100 if (t == 0 and units == 0) else t * 10 + units for t in tens]
    return D100(min(totals) if net > 0 else max(totals) if net < 0 else totals[0], tens, units)

def success_level(roll: int, skill: int, difficulty: Difficulty = "regular", cfg: RulesConfig = DEFAULT_RULES) -> Level:
    fumble_base = TARGET[difficulty](skill) if cfg.fumble_threshold_on == "target" else skill
    if roll == 1:
        return Level.CRITICAL
    if roll == 100 or (fumble_base < 50 and roll >= 96):
        return Level.FUMBLE
    if roll <= skill // 5: return Level.EXTREME
    if roll <= skill // 2: return Level.HARD
    if roll <= skill:      return Level.REGULAR
    return Level.FAILURE
```

```python
# keeper/game/coc7e/rules.py (changed parts)
def percentile_check(inv, kind, target, difficulty, bonus, penalty, rng, cfg=DEFAULT_RULES, pushed=False, turn_no=0):
    name, value = lookup(inv, kind, target)
    d = roll_d100(bonus, penalty, rng, cfg)
    level = success_level(d.result, value, difficulty, cfg)
    return {"kind": kind, "target": name, "value": value, "difficulty": difficulty, "bonus": bonus, "penalty": penalty,
            "tens": list(d.tens), "units": d.units, "roll": d.result, "level": level.name.lower(),
            "passed": level >= REQUIRED[difficulty], "pushed": pushed, "turn_no": turn_no}

def can_push(prev: dict | None, turn_no: int) -> bool:
    return (bool(prev) and prev["kind"] in PUSHABLE and not prev["passed"] and prev["level"] != "fumble"
            and not prev["pushed"] and prev.get("turn_no") == turn_no - 1)          # only the immediately previous turn

def sanity_check(inv, notation, rng, cfg=DEFAULT_RULES, turn_no=0) -> tuple[list[dict], dict]:
    on_success, on_failure = _split_san(notation)
    san = int(inv["san"])
    d = roll_d100(rng=rng, cfg=cfg)
    level = success_level(d.result, san, "regular", cfg)
    success = level >= Level.REGULAR
    lost = max_expr(on_failure) if (level == Level.FUMBLE and cfg.fumbled_san_is_max_loss) \
        else roll_expr(on_success if success else on_failure, rng)
    lost = max(0, min(lost, san))
    after = san - lost
    day_start = int(inv.get("san_day_start", san))
    lost_today = int(inv.get("san_lost_today", 0)) + lost
    rec = {"kind": "sanity", "target": "SAN", "value": san, "difficulty": "regular", "bonus": 0, "penalty": 0,
           "tens": list(d.tens), "units": d.units, "roll": d.result, "level": level.name.lower(), "passed": success,
           "pushed": False, "turn_no": turn_no, "san_loss": lost, "san_after": after,
           "involuntary_action": not success,
           "indefinite_insanity": lost_today >= max(1, day_start // 5),       # KRB via secondary sources [medium]
           "permanent_insanity": after <= 0}
    records = [rec]
    if lost >= 5 and after > 0:                                               # QSR: INT roll decides temporary insanity
        int_rec = percentile_check(inv, "characteristic", "INT", "regular", 0, 0, rng, cfg, turn_no=turn_no)
        int_rec["purpose"] = "temporary_insanity_check"
        rec["temporary_insanity"] = int_rec["passed"]                         # success = understands the horror -> insane 1D10 h
        records.append(int_rec)
    return records, {**inv, "san": after, "san_day_start": day_start, "san_lost_today": lost_today}
```

**Tests to add:**

```python
def test_fumble_flag():
    tgt, base = RulesConfig(fumble_threshold_on="target"), RulesConfig(fumble_threshold_on="base_skill")
    assert success_level(97, 60, "hard", tgt) is Level.FUMBLE      # target 30 < 50
    assert success_level(97, 60, "hard", base) is Level.FAILURE    # skill 60 >= 50
def test_cancel_then_cap():
    assert len(roll_d100(3, 1, make_rng(1, "x")).tens) == 3        # net +2 -> 1 + 2 tens dice
@given(st.integers(0, 10_000))
def test_int_roll_present_when_loss_ge_5(seed):
    recs, _ = sanity_check({"san": 60, "characteristics": {"INT": 70}, "skills": {}}, "5/5", make_rng(seed, "s"))
    assert any(r.get("purpose") == "temporary_insanity_check" for r in recs)
```

Document the chosen default (`target`) and its source in the thesis methods chapter. Report a sensitivity check on the adjudication dataset: how many outcomes flip between `target` and `base_skill`.

---

### C6. Duplicate `client_turn_id` rewrites history instead of being rejected [medium]

**Location:** §6.2 `post_turn` (`HumanMessage(content=body.text, id=f"h-{body.client_turn_id}")`); design A's idempotency claim, carried into §1 ("idempotency").

**Problem:** `add_messages` merges by id. A message whose id already exists **replaces the old message in place** rather than appending. A double-submit, or a client resend after a network blip following a completed turn:
- overwrites the earlier player message (same text, but its position stays old);
- starts a new turn whose history ends with an `AIMessage`. With `trim_messages(start_on="human")` and Gemini 3.6+ "no model prefill", that request can fail with `ValueError` (see C9).

This cannot be handled inside the SSE generator: by then `http.response.start` has been sent and status codes can't change.

**Evidence [verified]:** the `add_messages` reducer replaces by id (LangGraph graph-api docs). §6.2 itself explains that pre-stream checks must live in a dependency.

**Fix:** a dependency that checks the id before streaming.

```python
async def new_turn(body: TurnIn, request: Request, campaign=Depends(idle_campaign)):
    snap = await request.app.state.graph.aget_state({"configurable": {"thread_id": campaign.id}})
    if any(getattr(m, "id", None) == f"h-{body.client_turn_id}" for m in snap.values.get("messages", [])):
        raise HTTPException(409, detail={"code": "duplicate_turn", "retryable": False})
    return campaign

@router.post("/campaigns/{campaign_id}/turns", response_class=EventSourceResponse)
async def post_turn(body: TurnIn, request: Request, campaign=Depends(new_turn)) -> AsyncIterable[ServerSentEvent]:
    ...
```

---

### C7. The paired design is broken by arm-dependent adjudication; add cached aux outputs or a two-stage eval [high for thesis validity]

**Location:**
- §3.1 Principle 2 ("Only the RAG block in the prompt and the generator model differ between variants")
- §3.3 aux-calls paragraph ("do not confound the comparison")
- §3.6 reproducibility rule 3 ("The same item and seed get the same dice in every arm")
- §6.1 `adjudicate` (renders `use_rag` and `rules_hits`)

**Problems:**
1. **Different inputs per arm.** `adjudicate` sees `rules_hits` only when `use_rag=True`, so RAG and non-RAG arms send *different* adjudicator prompts. Its outputs (needs_roll, target, difficulty, bonus/penalty) differ. The RNG stream is paired, but the dice are consumed by different checks, so outcomes are not paired.
2. **Aux calls are not deterministic.** For current Gemini models the SDK drops `temperature`/`top_k`/`top_p` ("fixed sampling"; see C9), and `seed` is "best effort". Even across two non-RAG arms, adjudication and `analyze` queries vary run to run.
3. **Result.** Differences in narration quality are confounded with adjudication and retrieval-query variance. McNemar/Wilcoxon "paired" tests then overstate what the pairing controls for.

**Evidence:** plan code §6.1 **[verified by reading]**. `langchain_google_genai/chat_models.py`: `_FIXED_SAMPLING_AND_NO_PREFILL_MODELS = {"gemini-3.5-flash-lite", "gemini-3.6-flash"}`, with the comment "Per Google's docs these rules apply to these models and all future Gemini model releases" **[verified]**. `_common.py` seed docstring: "Reproducibility is 'best effort'" **[verified]**.

**Fix:** Split evaluation into two stages and report both.

- **Stage A, "generator-isolated" (the primary 2×2 table).**
  - For each (item, seed), run `analyze → retrieve → adjudicate → mechanics` **once**, with RAG on, and freeze the result as `turn_context.jsonl`.
  - Every arm then receives `forced_search_query`, `forced_hits`, `forced_adjudication` and `forced_rolls`.
  - Non-RAG arms simply do not render RULES CONTEXT.
  - Only `narrate` varies (generator × RULES CONTEXT present), which is exactly the thesis's RAG × FT question.
- **Stage B, "end-to-end system".** The current pipeline with live aux calls, reported as a secondary table with variance across 3 seeds.

```python
# keeper/nodes.py — generalise the existing forced_* pattern (topology unchanged)
async def adjudicate(self, state, runtime):
    if (fa := state.get("forced_adjudication")) is not None:
        return {"adjudication": fa, "forced_adjudication": None}
    ...
async def mechanics(self, state, runtime):
    if (fr := state.get("forced_rolls")) is not None:          # replay frozen dice records verbatim
        return {"turn_rolls": fr, "roll_log": fr, "forced_rolls": None}
    ...
```

```python
# evals/freeze_context.py — run once per (item, seed) with the RAG spec, before any arm
ctx = await graph.ainvoke(inputs, cfg, context=KeeperContext(spec=rag_spec, campaign_id=tid),
                          interrupt_before=["narrate"])     # stop before generation; values hold the frozen context
frozen = {k: ctx[k] for k in ("search_query", "rules_hits", "adjudication", "turn_rolls")}
```

Before relying on `ctx[...]`, check that `interrupt_before` returns values with an `InMemorySaver` (read them with `aget_state`). Alternatively, compile a sub-graph without `narrate` (`build_graph(deps, stop_before_narrate=True)`).

---

### C8. The FT factor never touches mechanics interpretation; make `adjudicator: generator` a reported condition [medium–high]

**Location:** §3.6 `thesis-2x2-gemma4.yaml` (`fixed: adjudicator: aux`); §5 Phase 5 (ablation "adjudicator: generator", optional); design B's rationale quoted in §1 ("B's FT arm cannot affect adjudication at all; that is defensible but not discussed").

**Problem:** The thesis (tezy, lines 36 and 38) claims three FT effects:
1. terminology;
2. Keeper narration style;
3. "типові шаблони інтерпретації ігрових механік" (typical patterns of interpreting mechanics), plus better use of retrieved rules.

In the main matrix, effect 3 is structurally excluded: a fixed Gemini decides every roll. The study would then support only effects 1 and 2, and the write-up would silently narrow the thesis claim.

Also note that the "local Gemma" arms still depend on Gemini for 2–3 calls per turn. That weakens the "limited compute" motivation (tezy line 37).

**Fix:**
- Keep `adjudicator: aux` for Stage A (C7).
- Add one **primary** condition in Stage B: `adjudicator: generator` for the four Gemma arms. Evaluate it on `adjudication.v1.jsonl` (accuracy, parse-failure rate, injection flip rate).
- Measure structured-output parse failures separately. Otherwise a "worse FT adjudication" result may just be JSON breakage.

```yaml
# experiments/thesis-2x2-gemma4.yaml (addition)
conditions:
  - {name: aux_fixed,      adjudicator: aux}         # Stage A/B primary for narration quality
  - {name: self_adjudicate, adjudicator: generator}  # Stage B: tests the thesis claim about mechanics interpretation
```

---

### C9. Gemini specifics: fixed sampling and no-prefill on 3.6+ models, the `seed` audit, and `google_api_key=None` [medium]

**Location:**
- §1.1 L2 and L15
- §3.6 reproducibility rule 6 ("Gemini 3+ is left at its default of 1.0 … and logged")
- §6.6 `_build` (`google_api_key=self.key.get_secret_value() if self.key else None`)
- §3.6 profile comment

**Findings [all verified in `langchain-google-genai` source on `main`]:**
1. **Fixed sampling.** `_FIXED_SAMPLING_AND_NO_PREFILL_MODELS = {"gemini-3.5-flash-lite", "gemini-3.6-flash"}`. For these, `temperature`, `top_k` and `top_p` are **dropped with a `UserWarning`**. The comment says Google applies this to "all future Gemini model releases", and the allowlist is explicit and must be updated as models ship. `gemini-3.8-flash` (the plan's choice) is newer than 3.6 and should be assumed fixed-sampling. The model is GA and priced on the official page **[verified]**, but it is not yet in the SDK allowlist.
   - **Consequence 1:** rule 6's "logged temperature 1.0" is misleading. Log `sampling: provider-fixed` instead.
   - **Consequence 2:** "no model prefill". The SDK raises `ValueError` if the last history turn has role `model` (checked for allowlisted models; assume the API enforces it for 3.8). Every Gemini call must end with a `HumanMessage`. The plan does satisfy this today, but C6 (duplicate ids) and any future "continue narration" feature would break it. Add an assertion.
2. **Seed.** `seed` is a model field **and** a call-time kwarg (`"seed": kwargs.get("seed", self.seed)`), so `.bind(seed=…)` would also have worked. L2's "doubtful" verdict is too cautious. The constructor approach is fine; just correct the audit note.
3. **API key.** The field is `google_api_key: SecretStr | None = Field(alias="api_key", default_factory=secret_from_env(["GOOGLE_API_KEY","GEMINI_API_KEY"], default=None))`. Passing `None` explicitly overrides the default factory. `validate_environment` then raises `"API key required for Gemini Developer API…"` even if `GOOGLE_API_KEY` is set in the container env.
4. **Retries.** `max_retries` semantics: `max_retries=1` means no retry, and `0` means "use the Google default". So `max_retries=2` is one retry.

**Fix:**

```python
case "google_genai":
    from langchain_google_genai import ChatGoogleGenerativeAI
    kw: dict = {}
    if self.key:                                   # only pass when set; otherwise the env default_factory applies
        kw["api_key"] = self.key.get_secret_value()
    if p.thinking_level:
        kw["thinking_level"] = p.thinking_level
    if seed is not None:
        kw["seed"] = seed
    # no temperature for Gemini >= 3.6: provider-fixed sampling (SDK drops it with a warning)
    return ChatGoogleGenerativeAI(model=p.model, max_tokens=p.max_tokens, timeout=self.timeout,
                                  max_retries=3, rate_limiter=self.limiters.get("google_genai"), **kw)
```

```python
# keeper/nodes.py — guard shared by analyze/adjudicate/narrate
def assert_ends_with_human(msgs):
    if not msgs or msgs[-1].type != "human":
        raise ValueError("last message must be a player turn (Gemini >= 3.6 rejects model prefill)")
```

---

### C10. Context budget: Ollama truncation and Cyrillic token undercount [medium]

**Location:** §3.6 `models.yaml` (`num_ctx: 8192` default in §6.6 `ModelProfile`), §6.1 `narrate` (`trim_messages(..., token_counter=count_tokens_approximately, max_tokens=spec.history_max_tokens)`), §3.6 `history_max_tokens: 6000`.

**Problem:**
- **Budget arithmetic.** Narrate sends:
  - system prompt ≈ 600–900 tokens;
  - investigator JSON ≈ 300–500;
  - RULES CONTEXT of 5 × 1000 characters ≈ 1,300–1,600 English tokens;
  - history up to 6,000 tokens;
  - plus `num_predict` (1024+).

  That is over 9,000 tokens against `num_ctx=8192`. Ollama truncates overlong input to fit the context window, which can drop the system prompt and RULES CONTEXT. That silently invalidates RAG arms **[medium: documented Ollama behaviour, confirm with `ollama ps`/logs in the smoke test]**.
- **Cyrillic undercount.** `count_tokens_approximately(chars_per_token=4.0)` **[verified signature in `langchain_core/messages/utils.py`]** assumes English. Ukrainian text usually tokenizes into more tokens per character, so real history is larger than counted.

**Fix:** Derive the history budget from the model context and calibrate the counter from real `usage_metadata`.

```python
# keeper/nodes.py — narrate
from functools import partial
CPT = {"uk": 2.8, "en": 4.0}      # starting values; recalibrate from runlog: chars / usage_metadata.input_tokens

def history_budget(profile, system_tokens: int, spec) -> int:
    ctx = profile.num_ctx if profile.provider == "ollama" else profile.context_window   # add context_window to ModelProfile
    return max(1_000, min(spec.history_max_tokens, ctx - system_tokens - profile.max_tokens - 256))

counter = partial(count_tokens_approximately, chars_per_token=CPT.get(state["scenario"].get("language", "uk"), 3.0))
system_tokens = counter([SystemMessage(system)])
history = trim_messages(state["messages"], strategy="last", token_counter=counter,
                        max_tokens=history_budget(self.d.llms.profiles[spec.generator], system_tokens, spec),
                        start_on="human")
```

**Profile settings:**
- Ollama Gemma profiles: `num_ctx: 16384`. Check the model's maximum with `ollama show gemma4:e4b`.
- vLLM: `--max-model-len ≥ 12288` (C4).
- Log `input_tokens` per call and fail an eval item if `input_tokens > 0.95 × ctx`.

---

### C11. Add client-side rate limiting and an API budget guard [medium; issue #7 remainder]

**Location:** §6.6 `ModelFactory`; §3.4 "Concurrency" (a semaphore is not a rate limit); §5 Phase 3 `evals/run.py`.

**Problem:**
- An eval matrix issues thousands of Gemini requests. Stage B alone is 230 items × 6 arms × 3 seeds × 2–3 calls ≈ 8–12k requests, plus judge calls.
- Free-tier limits for 3.8 Flash are reported by third parties at 15 RPM / 1,500 RPD [medium; confirm in AI Studio], so a single run would take more than a week. SDK retries on 429 just burn the daily quota.
- Nothing caps spend in the app either.

**Evidence [verified]:** the official pricing page for `gemini-3.8-flash`:
- Paid: $0.75 input / $3.75 output per 1M tokens through 2026-12-31, doubling from 2027-01-01.
- Batch: half price.
- Free tier: "Content used to improve our products: Yes"; paid tier: "No".

**Fix:**

```python
# keeper/llm.py
from langchain_core.rate_limiters import InMemoryRateLimiter

class ModelFactory:
    def __init__(self, profiles, *, google_api_key, ollama_url, vllm_url, timeout=120.0, google_rpm: int = 60):
        ...
        # ONE limiter per provider: quotas are per project/model, not per chat-model instance
        self.limiters = {"google_genai": InMemoryRateLimiter(requests_per_second=google_rpm / 60,
                                                             check_every_n_seconds=0.1, max_bucket_size=1)}
```

```python
# server/budget.py — daily token budget from the run log (issue #7: "spends the Gemini key")
class Budget:
    def __init__(self, max_output_tokens_per_day: int): ...
    def check(self): ...   # raise BudgetExceeded -> SSE error {code: "budget_exceeded", retryable: false}
    def add(self, usage_metadata: dict): ...
```

**Cost estimate [medium; token counts assumed, prices verified]:**

| Workload | Assumption | Estimate (2026 prices) |
|---|---|---|
| One RAG play turn | analyze 1.5k in + adjudicate 2.5k in + narrate 6k in; about 1.2k output incl. thinking | ≈ $0.012 per turn; a 100-turn session ≈ $1.2 |
| One Stage B matrix | 4,140 item-runs × aux ≈ $0.004 + 1,380 Gemini-reference narrations × ≈ $0.008 + 2,340 judge calls × ≈ $0.0034 | ≈ $35–40 per full run; budget 3–5 reruns ≈ $120–200 |
| Stage A (C7) | aux once per (item, seed) instead of per arm | cuts aux cost about 6× |
| SFT candidates | 1,500 × (3k in, 0.6k out) with a Flash-class teacher | ≈ $7 (≈10× with a Pro-class teacher) |

**Recommendation:**
- Use the **paid tier** for anything containing Keeper Rulebook text or player chats, because free-tier content is used for product improvement.
- Use the Batch API for SFT candidate generation and judge calls; neither needs streaming.
- Finish the runs before 2027-01-01, or budget double.

---

### C12. Retrieval covers only rules; lore, scenarios and narrative examples from the thesis corpus never reach generation [medium–high for thesis alignment]

**Location:** §6.5 `Retriever.search(..., source_types=("rules",))` plus the fixed `visibility="public"` filter; §3.6 `retrieval.collection: coc_rules__…`; §5 Phase 2 (only rules are ingested); Phase 3b (scenarios are optional); nothing covers narrative examples.

**Problem:** The thesis corpus (tezy line 25) has four data types:
1. normative rules;
2. structured world descriptions (monsters, locations, artifacts);
3. scenario materials;
4. narrative examples / Keeper-style dialogues.

In the plan, only type 1 is retrieved. Type 2 is committed (`data/open/` Lovecraft) but filtered out. Type 3 is deferred to optional Phase 3b. Type 4 appears only implicitly as SFT targets. Tezy line 26 also says scenarios can be a "source for generating new plots"; that is not addressed. A defence reviewer will compare Figure 1 of the thesis against the implementation.

**Fix (small, keeps the graph unchanged):**
- One collection per game system, with `source_type` in the payload.
- Per-type quotas in `RetrievalSpec`.
- A non-citable STYLE EXAMPLES block, and a secret KEEPER NOTES block for the self-written demo scenario. Because `demo-haunt` is the author's own text, there is no copyright or spoiler-test burden for the MVP.

```python
class RetrievalSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    collection: str = "coc7e__nomic15__c1000o200"
    embedder: str = "nomic15"
    quotas: dict[str, int] = {"rules": 4, "lore": 1}           # Phase 2; add "scenario": 2, "narrative_example": 1 later
    exact: bool = True

# Retriever.search — one exact query per source type (tiny corpus → cheap), then stable ordering
async def search(self, query: str, r: RetrievalSpec, visibility: str = "public") -> list[dict]:
    vector = await self.embedders[r.embedder].aembed_query(query)
    hits = []
    for source_type, limit in r.quotas.items():
        res = await self.client.query_points(
            collection_name=r.collection, query=vector, limit=limit, with_payload=True,
            search_params=models.SearchParams(exact=r.exact),
            query_filter=models.Filter(must=[
                models.FieldCondition(key="metadata.game_system", match=models.MatchValue(value=self.game_system)),
                models.FieldCondition(key="metadata.source_type", match=models.MatchValue(value=source_type)),
                models.FieldCondition(key="metadata.visibility", match=models.MatchValue(value=visibility)),
            ]))
        hits += [(source_type, p) for p in res.points]
    return [{"marker": i + 1, "source_type": st, "chunk_id": str(p.id), "score": float(p.score),
             "source": p.payload["metadata"].get("source_title"), "section": p.payload["metadata"].get("section_path"),
             "page": p.payload["metadata"].get("page"), "text": p.payload["page_content"]}
            for i, (st, p) in enumerate(hits)]
```

**Prompt changes in `keeper-v1.md.j2`:**
- `rules`/`lore` hits go under RULES & WORLD CONTEXT and are citable as `[R#]`.
- `narrative_example` hits go under STYLE EXAMPLES ("imitate tone, never facts, never cite").
- `scenario` hits (visibility `keeper`) go under KEEPER NOTES ("secret; reveal only through play").

**Thesis write-up:** state that narrative examples are used both as retrieval exemplars and as SFT targets. Plot generation from scenarios is future work.

---

### C13. SQLite checkpointer path and location on Windows [medium]

**Location:** §6.2 `open_checkpointer` (`AsyncSqliteSaver.from_conn_string(url.removeprefix("sqlite:///"))`); §3.5 "Without Docker on Windows"; repo path `C:\Users\1\Documents\GitHub\…`.

**Problems:**
1. **Missing directory.** `from_conn_string` is `aiosqlite.connect(conn_string)` **[verified source]**. SQLite does not create parent directories, so a native host run from `src/backend` fails with "unable to open database file" until `var/` exists.
2. **WAL on synced or network storage.** `setup()` runs `PRAGMA journal_mode=WAL` **[verified source]**. WAL needs a shared-memory `-shm` file and is unreliable on network or synced filesystems:
   - a Docker Desktop **bind mount** from NTFS (the plan correctly uses a named volume `api_var`);
   - a **OneDrive-redirected `Documents` folder**, which is common on Windows 11 and is where this repo lives.
3. **`setup()` is internal.** Its docstring says it "is called automatically when needed and should not be called directly by the user". Calling it is harmless (idempotent under the saver's lock), but the plan's comment implies it is required.

**Fix:**

```python
from pathlib import Path

@asynccontextmanager
async def open_checkpointer(url: str):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    path = Path(url.removeprefix("sqlite:///")).resolve()
    if any(part.lower().startswith("onedrive") for part in path.parts):
        raise RuntimeError(f"{path}: SQLite WAL on a OneDrive-synced folder is unsafe; set CHECKPOINT_URL elsewhere")
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield saver                                   # setup() runs lazily on first use
```

Remove the Postgres branch from the MVP (see §4). Checkpoint growth: the SQLite saver stores whole checkpoints per step, so long campaigns grow the file super-linearly. Fine for a thesis, but add a `scripts/vacuum_threads.py` later if `var/checkpoints.db` exceeds a few hundred MB.

---

### C14. Citations always show `page=None`; chunk ids ignore the source document [low–medium]

**Location:** §6.5 `ingest.py` (`chunk_markdown`, `metas`, `uuid5(sha256(t))`); §3.4 SSE `done.citations[].page`; §6.3 prompt (`p.{{ h.page }}`).

**Problems:**
1. **No page numbers.** Markdown carries no page numbers, and `chunk_markdown` never sets `page`. The UI and thesis citations promise pages but always get `None`.
2. **Id collisions across documents.** Ids hash only the chunk text (section path + content). Identical boilerplate in two documents collides, and `upsert` then overwrites the other document's metadata. QSR and Keeper Rulebook sections with the same heading and identical paragraphs are plausible.

**Fix:** Keep page anchors during the offline PDF→Markdown conversion, and hash `(source_title, text)`.

```python
PAGE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")       # emitted by the offline converter, one per PDF page

def split_pages(md: str) -> list[tuple[int | None, str]]:
    parts, page, out = PAGE.split(md), None, []
    out.append((None, parts[0]))
    for i in range(1, len(parts), 2):
        out.append((int(parts[i]), parts[i + 1]))
    return [(p, t) for p, t in out if t.strip()]

# in ingest(): chunk each page's text separately (overlap never crosses pages) OR record the first page seen in the chunk
point_id = str(uuid.uuid5(NAMESPACE, hashlib.sha256(f"{meta['source_title']}\x1f{t}".encode()).hexdigest()))
```

Also normalise text before hashing (`unicodedata.normalize("NFC", t)`, collapse whitespace), so re-conversion on another OS gives identical ids and `gold_chunk_ids` stay valid.

---

### C15. Tests: `GenericFakeChatModel` cannot back `with_structured_output` [medium for the Week-1 plan]

**Location:** §4.3 row #44 ("graph test with `GenericFakeChatModel`"); §5 Phase 1a tests; §6.6 `ModelFactory.structured`.

**Problem:**
- `BaseChatModel.with_structured_output` needs `bind_tools` or a provider-native response format.
- Fake chat models implement neither, so `llms.structured(...)` raises `NotImplementedError` in tests. That breaks the analyze and adjudicate paths **[medium: base-class behaviour; confirm on first run]**.

**Fix:** inject a test `ModelFactory` whose `structured()` returns a `RunnableLambda`. This keeps `chat()` on `GenericFakeChatModel`, so token streaming is still exercised.

```python
# tests/fakes.py
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

class FakeFactory:
    def __init__(self, narrations: list[str], adjudications: list[dict], queries: list[str] | None = None):
        self._narr = iter([AIMessage(content=t) for t in narrations])
        self._adj, self._q = iter(adjudications), iter(queries or [])
    def chat(self, name, *, seed=None):
        return GenericFakeChatModel(messages=self._narr)          # streams chunks -> exercises the SSE token path
    def structured(self, name, schema, *, seed=None):
        src = self._q if schema.__name__ == "SearchQuery" else self._adj
        def _call(_msgs):
            item = next(src)
            return schema.model_validate({"query_en": item} if isinstance(item, str) else item)
        return RunnableLambda(_call)
```

**More tests the plan does not list:**
- `test_prompts.py`: render every template branch (`use_rag`, rules question, errors) with `StrictUndefined`, and assert the template hash is stable across OSes (LF normalisation).
- `test_retriever.py` with `AsyncQdrantClient(location=":memory:")`. Local mode supports `query_points` with an exact filter, so no Docker is needed in CI.
- `test_idempotency.py` (C6), `test_resume_crash.py` (C1), `test_budget.py` (C11).
- `test_eval_smoke.py`: 3 items × 2 arms through `evals/run.py` with `FakeFactory`. Assert the JSONL schema and `spec_hash`.

---

### C16. Frontend: `crypto.randomUUID()` is only available in secure contexts [low]

**Location:** §6.7 `sse.js` usage comment (`client_turn_id: crypto.randomUUID()`); issue #23 (LAN/phone access).

**Problem:** `crypto.randomUUID` exists only in secure contexts (HTTPS or `localhost`) [verified MDN behaviour]. Opening Vite at `http://192.168.x.x:5173` from a phone, which issue #23 wants to allow, throws `TypeError` on the first turn. `crypto.getRandomValues` is available in insecure contexts.

**Fix:**

```js
export const uuid = () =>
  globalThis.crypto?.randomUUID?.() ??
  '10000000-1000-4000-8000-100000000000'.replace(/[018]/g, c =>
    (c ^ (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))).toString(16))
```

---

### C17. Gaps in `run_turn` robustness [low]

**Location:** §6.2 `run_turn`, §4.3 row #10/#56.

1. **Blocked prompts loop forever.** A deterministic safety block (`finish_reason=SAFETY`) makes `narrate` fail the same way on every Retry. After 2 consecutive `empty_generation` errors on the same pending checkpoint, return `retryable: false` with the message "rephrase your action". Clearing the pending turn needs a `Command`/`update_state` to drop the last `HumanMessage` (`RemoveMessage(id=…)`).
2. **Undefined dependencies.** `st.deps.citations(...)` and `st.deps.experiments.resolve(...)` are called, but `Deps` in §3.2 lists only `(llms, prompts, scenarios, retriever|None, runlog)`. Add `experiments: ExperimentRegistry` and a pure `citations(values) -> (valid, invalid)` in `keeper/citations.py`, so evals can reuse it.
3. **Client disconnect.** Whether an ASGI server cancels the generator promptly depends on the ASGI spec version the server advertises. §7 test 3 must assert both outcomes: the turn is either cancelled and resumable, or completed server-side and visible on reload.
4. **Missing git sha in Docker.** `GET /api/meta` git sha: `.dockerignore` excludes `.git`, so pass `ARG GIT_SHA` at build time.

---

### C18. Smaller factual fixes in the audit table and prose

| Location | Statement | Correction |
|---|---|---|
| §1.1 L2 | "Doubtful for `ChatGoogleGenerativeAI` (seed is a model field, and call-time passthrough is not documented)" | Call-time works: `_prepare_params` reads `kwargs.get("seed", self.seed)` [verified]. Keep the constructor approach anyway. |
| §1.1 L15 / §3.6 rule 6 | "Gemini 3+ auto-sets temperature 1.0 when unset" | The SDK sets `temperature=None` for Gemini ≥ 3.0 when unset (server default). For 3.5-flash-lite, 3.6-flash and, per Google, all later models, custom sampling parameters are dropped entirely [verified source]. Log "provider-fixed sampling". |
| §6.1 note, §4.3 #11 | Timeouts are retried by `RetryPolicy` | Only if `retry_on` includes `TimeoutError`/`NodeTimeoutError` (C2). |
| §6.3 | "Load it with `jinja2.Environment(undefined=StrictUndefined, keep_trailing_newline=True)`" | Also set `autoescape=False` explicitly and hash the **LF-normalised** source, or Windows checkouts without `.gitattributes` produce different prompt hashes. |
| §3.5 `vllm` | "base and LoRA adapter on one server at the same precision" | True at serve time only; training was QLoRA (NF4 base). State this in the limitations (C4). |
| §3.6 matrix, FT × Gemini | "Developer API has no tuning" | Fine. Add that Gemini references are **not** paired with Gemma arms on sampling (fixed sampling vs temperature 0.7), so compare Gemini only descriptively. |
| §6.4 docstring | "fumble 100, or 96-100 when skill < 50" | Say "when the roll required for success (difficulty-adjusted, configurable) is < 50" (C5). |
| §3.2 `training/requirements-train.txt` | `unsloth==2026.9.4, trl==1.13.0, peft==0.20.0` | Versions exist on PyPI [verified]. Also pin `transformers` and `bitsandbytes` from the same Colab image, and record `torch.__version__` / CUDA in `train_meta.json`. |
| §5 Phase 4 | `r=16, lora_alpha=16, lr about 1e-4` | Reasonable. Unsloth's Gemma 4 example uses `r=8, lora_alpha=8, learning_rate=2e-4` [verified docs]; TRL recommends ≈1e-4 for adapters. Treat r ∈ {8,16} as a small ablation only if time allows. |

---

## 3. Missing items and coverage gaps

### 3.1 Thesis components → plan

| Thesis component (tezy line) | Where in the plan | Status | Gap / action |
|---|---|---|---|
| Corpus: rules (25–26) | Phase 2, `data/private` QSR/KRB, `data/open` BRP ORC | Covered | n/a |
| Corpus: world descriptions / lore (25) | `data/open` Lovecraft; `source_type=lore` in the ingest docstring | **Partial** | Filtered out of retrieval → C12 |
| Corpus: scenarios, used directly or to generate plots (25–26) | Phase 3b (optional) | **Partial** | Minimal KEEPER NOTES in Phase 2 with the self-written pack (C12); plot generation declared future work |
| Corpus: narrative examples / Keeper dialogues (25) | Implicit in SFT targets | **Partial** | Explicit `narrative_example` type + STYLE EXAMPLES block (C12) |
| Fixed-size chunks with overlap (27) | `RecursiveCharacterTextSplitter(1000, 200)` after the header split | Covered | Describe it in the thesis as "≤1000 characters, 200 overlap, structure-aware" (it is not strictly fixed); chunk size ablation in Phase 5 |
| Embeddings (nomic-embed-text) (28) | `PrefixedEmbeddings` wrapper, nomic v1.5 via Ollama | Covered | The double-prefix bug is correctly fixed. nomic v1.5 is English-only → cross-lingual design (§5 idea 6) |
| Vector DB (Qdrant) (28) | Qdrant 1.19.1, COSINE, payload indexes | Covered | n/a |
| Semantic cosine search (33) | `query_points`, `exact=True` | Covered | n/a |
| Context injection (34) | `keeper-v1` RULES CONTEXT with `[R#]` | Covered | Add lore/style/keeper blocks (C12) |
| FT dataset "query → reference Keeper answer" + PEFT/LoRA (35) | Phase 4 `build_sft.py`, RAFT mix, review CLI, Unsloth QLoRA | Covered, but the recipe is unsafe | C3 (truncation, template), C4 (serving) |
| FT learns terminology, style, mechanics interpretation (36) | FT is the narrator only | **Partial** | C8: `adjudicator: generator` condition |
| Base-model choice as future research (37) | Gemma 4 E4B picked | Covered | Justify with the "limited resources" criterion from the thesis (VRAM, QLoRA 17 GB per Unsloth, vLLM 24 GB bf16) |
| RAG priority on conflict (38) | Mandatory retrieve, prompt rule 5, conflict probes, context-adherence metric | Covered (strong) | Stratify probes by perturbation type (§5 idea 9) |
| Generalisation via knowledge-base swap (23, 42) | `game_system` payload, `keeper/game/<system>`, prompts per system | **Designed, not demonstrated** | Hard-coded `coc7e` defaults and CoC-only `Adjudication` (`sanity`). Add a `GameSystem` protocol and a small second-system demo (§5 idea 8) |
| Limitation: retrieval errors lead to wrong answers (40) | Error attribution (retrieval miss vs generation error) | Covered | n/a |
| Limitation: no formal verification (41) | `[R#]` marker validity, deterministic dice, typed adjudication | **Partially improved** | Marker validity ≠ support. Add a verifier node with claim-level entailment and mechanics consistency (§5 idea 1) |
| Quantitative comparison as future work (17, 42) | Phases 3–5 | Covered (goes beyond the thesis) | Fix pairing (C7) |

### 3.2 Critical / high / medium verified issues (#1–#36) → plan

The status reflects the plan after the corrections above. "Deferred" is accepted only where noted.

| # | Sev. | Issue | Plan location | Status |
|---|---|---|---|---|
| 1, 2, 3 | crit/crit/high | `lib/` gitignore drops `api.js` | §4.1 Phase 0 | Covered. Add a CI clean-checkout build on day 1, not day 10 (see #34). |
| 4 | high | No persistence; README claim | §4.3, §4.4, `AsyncSqliteSaver` | Covered (+C1, C13) |
| 5, 27 | med | Hard-coded SECRET_KEY | §4.1 (Django deleted) | Covered by deletion |
| 6, 28 | med | DEBUG/ALLOWED_HOSTS | §4.2 | Vanishes |
| 7, 36 | med | Unauthenticated, **unthrottled** key spend; 0.0.0.0 | §4.1 localhost bind; §4.3 lock/semaphore/`API_TOKEN` | **Partial.** No rate limit or budget → C11. `API_TOKEN` can go if bound to localhost (§4). |
| 8 | med | Payload validation | §4.3 `TurnIn` | Covered |
| 9 | med | Raw exception text | `server/errors.py` | Covered. Map 429/5xx codes (C2). |
| 10 | med | `response.text` None | `EmptyGeneration` | Covered (+C17 infinite-retry guard) |
| 11 | med | No timeout/retries | model `timeout`/`max_retries` + node timeouts | **Partial:** timeouts are not retried (C2) |
| 12 | med | Prompt contradicts thesis | `keeper-v1` | Covered |
| 13 | med | No LLM/RAG abstraction | `ModelFactory`, spec | Covered |
| 14 | med | New Campaign race | `useTurnStream` AbortController | Covered |
| 15 | med | Orphan turn, no retry | `/turns/retry` resume | Covered (+C1 durability, C6 duplicates) |
| 16 | med | Hard-coded campaigns | `/api/campaigns` | Covered |
| 17 | med | Newlines collapse | `whitespace-pre-wrap` | Covered |
| 18 | med | No timeout/cancel | AbortController | Covered. Note that aborting does not stop server generation (C17.3). |
| 19 | med | Null content bubble | token-built bubble | Covered |
| 20 | med | No streaming | SSE | Covered |
| 21 | med | Not responsive | §4.5 Phase 5 | **Deferred.** Acceptable only if the defence demo is on desktop. A `md:` sidebar drawer is about 1 hour, so do it in 1b. |
| 22 | med | Accessibility | §4.4 minimum | Partial (acceptable) |
| 23 | med | Hard-coded API URL | Vite proxy | Covered (+C16 for LAN) |
| 24, 32 | med | No lockfile / Node 20 | §4.4 | Covered |
| 25 | med | UI ≠ thesis agent | §4.4 + Citations in Phase 2 | Covered |
| 26 | med | Stale node_modules volume | `-V` docs | Covered (moot if Vite runs natively, §4) |
| 29 | med | No Node ignores | §4.1 | Covered |
| 30 | med | No production path | §4.5 deferred | Deferred (acceptable; see missing item M1) |
| 31 | med | Python 3.10 Alpine | `python:3.12-slim` | Covered |
| 33 | med | No vector DB in compose | `--profile rag` | Covered |
| 34 | med | No CI/lint/tests | Day 10 | **Partial.** Move one CI job to day 1 (it would have caught #1). Add the tests from C15. |
| 35 | med | No LICENSE; no corpus guard | §4.1 LICENSE + ignores | **Partial.** `.gitignore` does not stop `git add -f` → add a pre-commit/CI guard (M5). |

All low-severity issues are covered or explicitly deferred. Checked against §4.2–§4.5; no omissions found.

### 3.3 Windows 11 + Docker Desktop pitfalls

The plan already handles file watching (`WATCHFILES_FORCE_POLLING`, `VITE_USE_POLLING`), a SQLite named volume, `.gitattributes`, and avoiding psycopg on the Proactor loop. These are still missing:

| # | Pitfall | Symptom | Fix |
|---|---|---|---|
| W1 | Repo under `C:\Users\1\Documents\…`, which is often OneDrive-redirected | SQLite WAL `-shm`/`-wal` locking errors; OneDrive syncs `node_modules`, `.venv`, `var/`, `artifacts/` (thousands of files, file locks during `uv sync`/`npm ci`) | Check with `Get-Item $env:USERPROFILE\Documents \| Select Target`. Move the repo to `C:\dev\gm-assistant`, or better, into the WSL2 filesystem (`\\wsl$\Ubuntu\home\…`) when using Docker. Add the OneDrive guard (C13). |
| W2 | Bind mounts from NTFS into Linux containers | Slow Python imports and `npm` in containers; polling-based reload uses CPU | **MVP default: run `uvicorn` and `vite` natively on Windows**; Docker only for `qdrant`/`ollama` (§4). If you prefer containers, clone inside WSL2 so inotify works and polling is unnecessary. |
| W3 | Text-mode encoding defaults to the ANSI code page (cp1251/cp1252) on Python 3.12/3.13 | `UnicodeEncodeError` or mojibake when writing Ukrainian JSONL, logs, `open()` without `encoding=` | Set `PYTHONUTF8=1` in `.env.example`, the VS Code launch config and CI. Also always pass `encoding="utf-8"` and `ensure_ascii=False` in `runlog.py`, `build_sft.py`, `evals/*`. Add ruff rule `PLW1514` (unspecified-encoding). |
| W4 | CRLF in hashed artefacts | Prompt sha256 and chunk ids differ between a Windows checkout and Docker/CI, breaking `spec_hash` comparability and `gold_chunk_ids` | `.gitattributes`: `* text=auto eol=lf`, `*.j2 text eol=lf`, `*.sh text eol=lf`, `*.ps1 text eol=crlf`, `*.pdf *.png *.gguf *.safetensors binary`. Hash **LF-normalised** template text and NFC-normalised chunk text (C14, C18). |
| W5 | vLLM has no native Windows build | `pip install vllm` fails; the `vllm` compose profile needs WSL2 + NVIDIA | Docker Desktop with the WSL2 backend and a current NVIDIA Windows driver (GPU paravirtualisation). Raise WSL memory in `%UserProfile%\.wslconfig` (`[wsl2] memory=24GB`, `swap=16GB`), because the default cap is a fraction of host RAM. E4B at bf16 wants a 24 GB+ GPU (C4); otherwise run evals on a cloud GPU. |
| W6 | Ollama running twice (native Windows app plus the `ollama` container) | Port 11434 conflict, or models pulled into the wrong store | Pick one. On Windows, the native Ollama app is simplest and uses the GPU directly. Containers reach it via `http://host.docker.internal:11434`. Drop the `ollama` compose service from the `rag` profile or rename its port. |
| W7 | Hugging Face cache symlinks | Without Developer Mode, `huggingface_hub` falls back to copies and warns; the cache doubles | Run training/export in WSL2 or Colab (the plan already isolates `training/`). On Windows set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` or enable Developer Mode yourself. |
| W8 | `MAX_PATH` (260 characters) | `npm ci` or HF cache paths fail deep in `node_modules` / `.cache` | Short repo path (W1). Windows long-path support is a system setting the user may enable; the agent should not change it. |
| W9 | `uvicorn --reload` on Windows | Ctrl+C sometimes leaves the reloader child holding port 8000 | `Get-NetTCPConnection -LocalPort 8000 \| Select OwningProcess` → stop that PID. Document it in the README. `uvloop` is not used on Windows (`uvicorn[standard]` skips it). |
| W10 | Event-loop policy | Plan is fine for SQLite. If the Postgres branch is ever enabled on the host, psycopg async needs `WindowsSelectorEventLoopPolicy` (deprecated in 3.14) | Keep Postgres out of the MVP (§4). If needed later, run the API in Docker/WSL for the Postgres variant. |
| W11 | PowerShell vs bash in docs | README commands like `export VAR=…`, `rm -rf`, `\` line continuations fail in PowerShell | Give both PowerShell and bash snippets, or a cross-platform `justfile`/`uv run` script (`uv run python -m keeper.rag.ingest …`). |
| W12 | Frontend `crypto.randomUUID` over LAN HTTP | TypeError on a phone | C16 |
| W13 | Windows Defender scanning `node_modules`, `.venv`, `var/` | Very slow installs/tests | Ask the user to consider their own antivirus exclusions for dev folders. This is a security setting, so do not automate it. |

### 3.4 Missing items

**M1. Auth, multi-user and demo hosting.**
- Out of scope in the plan (§5 "Explicitly out of scope"). That is correct for a single-user thesis, but the **defence demo** needs a plan.
- Recommended: keep the app on `127.0.0.1` and demo locally.
- If a remote examiner must access it, put it behind an identity-aware tunnel (e.g. Cloudflare Tunnel + Access, or Tailscale), not home-made auth. The per-campaign `asyncio.Lock` and SQLite are single-process: document "one uvicorn worker" as a hard constraint.
- Drop `API_TOKEN` from the MVP. A static bearer token in a Vite-proxied SPA ends up in the browser anyway.

**M2. Rate limiting and API cost.** See C11. Also add `GET /api/meta.usage_today`, so the author sees spend during development.

**M3. Ukrainian language handling.** The plan has the `analyze` translation, prompt rule 4, the Cyrillic font and bilingual gold sets. Missing:
- **Retrieval language path is a hidden variable.**
  - With nomic v1.5 (English-only, [verified research]), a Ukrainian query depends on the aux model's translation.
  - Log `search_query` and the detected language.
  - Add a fallback when `analyze` fails: embed the raw query with a multilingual embedder, or return no hits with `errors=["query_translation_failed"]`. Never embed Ukrainian text with the English-only model silently.
- **Answer-language adherence metric.** Detect the reply language (e.g. `lingua-language-detector`, restricted to uk/en/ru) and report the % of turns answered in the player's language, per arm. Include Russian in detection: small models often drift uk→ru, and it is a visible failure for Ukrainian users.
- **Terminology glossary.** `data/glossary/coc7e.uk-en.yaml`, author-defined because the plan should not assume an official Ukrainian 7e terminology. Use it in three places:
  1. the prompt (rule 4 examples);
  2. `build_sft.py` (consistent targets);
  3. a metric: % of glossary terms rendered with the canonical Ukrainian form plus the English term on first use.
- **Token economics.** Cyrillic costs more tokens (C10). Report tokens per turn separately for uk and en in the latency/cost table.
- **Gold-set translation quality.** `question_uk` written or reviewed by a native speaker (the author), not machine-translated, or the cross-lingual study measures MT artefacts.
- **UI strings.** The existing UI is English/dark-fantasy; add a minimal `i18n` dictionary (uk default, en).

**M4. Copyright and data protection.** The plan is good (`data/private`, open corpus, LICENSE for code only). Add:
- **Gemini free tier:** "Content used to improve our products: Yes" [verified pricing page]. Rulebook chunks and player chats sent on the free tier may be used by Google. Use the **paid tier** for KRB-derived prompts, eval and SFT generation, and say so in the thesis.
- **CoCai is AGPL-3.0** [verified research]. Cite it and learn from it, but do not copy code into an MIT/Apache repo.
- **"Clean Up, Aisle Four!"** is used by CoCai "with permission" of the author [verified research]. That permission does not extend to this project. Treat it as private corpus unless you obtain permission.
- **Derived artefacts:** SFT data, gold sets built from KRB text, LoRA adapters and Qdrant snapshots are derived from copyrighted text. Keep them private (the plan ignores `/adapters/` and `/models/`; also `/artifacts/corpus/` chunk dumps). Do not upload adapters to the Hugging Face Hub publicly.
- **Thesis document and demo video:** short quotations only; show retrieved chunks truncated in screenshots.
- **Human evaluation participants** (§5 idea 10): informed consent, anonymised transcripts, no personal data in `artifacts/runs`. Chat text may contain personal details players type.

**M5. A corpus and secret guard that actually blocks commits.** `.gitignore` does not stop `git add -f data/private/krb.md` or a PDF dropped elsewhere.

```python
# scripts/guard_commit.py — pre-commit hook and CI step
import subprocess, sys, pathlib
BLOCK_EXT = {".pdf", ".gguf", ".safetensors", ".bin", ".pt", ".epub"}
BLOCK_DIRS = ("data/private/", "adapters/", "models/", "artifacts/")
MAX_BYTES = 2_000_000
staged = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=AM"],
                        capture_output=True, text=True, encoding="utf-8").stdout.split()
bad = [f for f in staged if f.startswith(BLOCK_DIRS) or pathlib.Path(f).suffix.lower() in BLOCK_EXT
       or (pathlib.Path(f).is_file() and pathlib.Path(f).stat().st_size > MAX_BYTES)]
if bad:
    sys.exit("blocked (corpus/weights/large files): " + ", ".join(bad))
```

In CI, run the same checks over `git ls-files`, plus a secret scanner such as gitleaks. The old Django `SECRET_KEY` is already public, so allowlist that commit.

**M6. CI from day 1.** The plan puts CI on Day 10. Minimum GitHub Actions job on `ubuntu-latest`:
- `pip install uv`
- `uv sync --frozen --group dev`
- `uv run ruff check`
- `uv run pytest -m "not live"`
- `npm ci && npm run build` in `src/frontend`
- `python scripts/guard_commit.py --all`

Set `PYTHONUTF8=1`. Optionally run a second job on `windows-latest` for only `tests/test_dice.py tests/test_prompts.py`, which catches encoding and CRLF hash drift (W3/W4). Live Gemini tests are `@pytest.mark.live` and never run in CI (no secrets).

**M7. Observability for the eval harness.** JSONL run logs are enough. Also record:
- `langgraph`, `langchain-*` and `google-genai` versions;
- vLLM/Ollama server versions (`/version` endpoints);
- GPU name;
- the Qdrant collection point count and payload-schema hash;
- the RulesConfig flags.

Without these, reruns months later cannot be explained.

**M8. Backups.** `var/checkpoints.db`, `var/app.db`, `data/private/eval/*.v1.jsonl` and SFT review decisions are unique research data. Add a weekly `scripts/backup.ps1`, copying to a private, non-OneDrive location. Note that the SQLite backup API or `VACUUM INTO` is safer than a file copy while WAL is active.

---

## 4. MVP cuts (still over-engineered for 10 days on Windows)

| Cut or simplify | Where | Why | Replacement |
|---|---|---|---|
| **api/web containers for dev on Windows** | §3.5 compose `api`, `web`; Frontend Dockerfile; `WATCHFILES_FORCE_POLLING`, `VITE_USE_POLLING`, `web_node_modules` volume | Bind-mount performance and polling on NTFS (W2), stale-volume docs (#26) and CRLF surprises cost days; SQLite needs no services | `uv run uvicorn server.main:app --reload` + `npm run dev` natively. Compose keeps only `qdrant` (and optionally `vllm` for WSL2). Keep the Dockerfiles, built in CI only, for #31/#32/#83 hygiene. |
| Postgres branch in `open_checkpointer` + `postgres` dependency group | §6.2, §3.5 | YAGNI; untestable on the Windows host (W10) | Delete. Mention the swap in one README sentence. |
| `@tool` `skill_check` + `ToolRuntime` "agentic demo mode" | §6.4 `tools.py` | Not used by any thesis arm; a second code path to maintain | Delete. |
| `langgraph.json` + `langgraph dev`/Studio | §3.2, §5 Day 10, §6.1 | Pulls in the Agent Server CLI just for a diagram | `print(build_graph(deps).compile().get_graph().draw_mermaid())` in `scripts/diagram.py`. |
| `EXPERIMENTS_ENABLED` arm picker in the New Campaign dialog, `GET /api/debug/search` | §3.4, §5 Phase 2 | UI and API surface for things only the author uses | `python -m keeper.rag.search "query" --collection …` and `python -m evals.run`. The app always uses `DEFAULT_ARM`. |
| `max_concurrent_turns` semaphore, `API_TOKEN` | §3.4, §6.2 | Single user, localhost | Per-campaign lock only. Token only with M1 remote demo, via tunnel auth instead. |
| `progress` event after every node + stage indicator | §3.4, §4.4 #20 | Nice but not needed for Day 10 | Emit `progress` only for `retrieve` and `mechanics`. Keep `roll` + `token` + `done` + `error`. |
| Push rules, bonus/penalty in the Day-10 adjudicator | §5 Phase 1c | Push validation plus hypothesis invariants in the same week as SSE/frontend is risky | Day 10: regular/hard/extreme checks + SAN (with C5 fixes). Push and bonus/penalty dice in the first Phase 2 week (the code is ready, only enable it). |
| Ollama merged-GGUF FT profile (`gemma4-keeper-ollama`) | §3.6, §5 Phase 4, §6.6 | Quantization confound, export time, and it is not a thesis arm | Only if a live offline demo of the FT model is required. |
| Gemini reference arms × 3 seeds | §3.6 `seeds` × `ref_gemini*` | Descriptive only (not paired, C18); costs about a third of the matrix | 1 seed for the two reference arms. |
| `adjudicator` ablation as an optional Phase 5 item | §5 | Wrong place: it is a core claim (C8) | Promote to a Stage B condition; drop embedder ablations to 2 embedders if time is short. |
| Phase 3b in full (plan node, clue-by-id, summariser, interrupt, leak metric) | §5 | Already optional, but it tempts scope creep | Replace with the minimal KEEPER NOTES block from C12. Keep summarisation only as research idea 7. |
| Separate `var/app.db` + `store.py` table | §3.4 | Small, but a second SQLite file to back up and migrate | Acceptable to keep. Alternatively a `campaigns.json` written atomically, which is fine for fewer than 100 campaigns. Low priority. |

**Keep despite the cost** (these carry the thesis):
- `ExperimentSpec` + `spec_hash`
- prompt hashes
- deterministic dice with seeded RNG
- `forced_*` input keys
- `[R#]` citations
- conflict probes
- JSONL run log
- `EmptyGeneration`
- resume (with C1)

---

## 5. Research and novelty ideas for the master's thesis

Each idea names the thesis claim it strengthens, the design, and the measurement. All reuse the existing graph, `forced_*` keys and JSONL records. Ideas 1, 3 and 5 are the strongest for a defence.

### Idea 1. A rules-grounding verifier node turns the "no formal verification" limitation into a measured contribution

- **Claim addressed:** tezy line 41 ("відсутність формальної перевірки коректності").
- **Design:** add `verify` after `narrate`, running on the out-of-character part and rules answers.
  1. **Deterministic checks (formal, cheap):**
     - every number in the "Mechanics:" line and every success/failure word must match `turn_rolls` (roll, level, passed, san_loss);
     - no dice results appear when `turn_rolls` is empty;
     - every `[R#]` ∈ retrieved markers (already planned);
     - rule constants (e.g. "Hard = half", "max 2 bonus dice") are checked against a small typed rules table (`keeper/game/coc7e/facts.yaml`).
  2. **Claim-level grounding:** an aux model extracts atomic rule claims; each claim is checked for entailment against its cited chunk by an NLI or LLM judge. Claims without a supporting chunk are flagged `unsupported`.
  3. **Policy variants:** `off`, `flag` (show a warning chip in the UI), `revise` (one regeneration with the verifier feedback; only in evals, since streaming UX conflicts with revision).
- **Metrics:**
  - mechanics-consistency violation rate per arm (should be about 0 with deterministic checks);
  - unsupported-claim rate;
  - verifier precision/recall against ~150 human-labelled claims (Cohen's κ for the labels);
  - the change in rules-QA correctness and context-adherence with `revise` vs `off`;
  - added latency and tokens.
- **Why it is novel here:** it converts the stated limitation into a partial, measurable formal-verification layer. It is also system-agnostic, since the rules table swaps with the knowledge base.

### Idea 2. Citation faithfulness, not just citation validity

- **Claim addressed:** "RAG is responsible for factual accuracy" (tezy line 38).
- **Design:** ALCE-style citation evaluation per answer sentence:
  - **citation recall:** is each rules sentence supported by the union of its cited chunks?
  - **citation precision:** is each cited chunk actually needed?
- **Metrics:** citation recall and precision per arm; correlation with correctness; FT vs base difference (does RAFT-style SFT improve faithful citing?).
- **Cost:** reuses the idea-1 entailment judge.

### Idea 3. A proper factorial test of "RAG and FT compensate each other"

- **Claim addressed:** tezy line 42 ("їх комбінування взаємно компенсує слабкі сторони"). This is an *interaction* claim, and two separate pairwise tests cannot show it.
- **Design:**
  - Stage A (C7) makes narrate the only varying step.
  - Fit a mixed-effects logistic regression: `correct ~ RAG * FT + language + category + (1|item) + (1|seed)`.
  - Continuous style scores use a linear mixed model.
  - Fixed seeds [11, 12, 13] and frozen turn contexts.
- **Metrics:**
  - main effects and the **RAG×FT interaction** coefficient with 95% CI;
  - marginal-means plot;
  - Holm-corrected pairwise contrasts as secondary results;
  - a pre-registered hypothesis list (write it into `experiments/HYPOTHESES.md` before running) to avoid garden-of-forking-paths criticism.

### Idea 4. Hybrid lexical+dense retrieval vs dense only, for a terminology-heavy domain

- **Rationale:** CoC queries are full of exact tokens ("Spot Hidden", "1D6", "POW×5", "Major Wound") where BM25 excels; nomic handles paraphrase.
- **Design:** 4 retrievers over the same chunks:
  1. dense (nomic);
  2. BM25 / Qdrant sparse;
  3. hybrid with RRF fusion;
  4. hybrid + cross-encoder rerank (the plan already has a `reranker` hook).

  Queries in en and uk (via `analyze`).
- **Metrics:**
  - Hit@5, Recall@5/20, MRR against `gold_chunk_ids`, broken down by query type (term lookup / paraphrase / multi-hop);
  - downstream rules-QA correctness;
  - error attribution (retrieval miss vs generation error);
  - latency.
- **Keep deterministic:** `exact=True`; record sparse model and version.

### Idea 5. Deterministic mechanics engine vs LLM adjudication under rhetorical injection (Ukrainian CoC-Seduce)

- **Claim addressed:** reliability of mechanics interpretation; prior art: CoC-Seduce (arXiv 2607.02802, 5,376 samples, 16 skill categories; it shows scale and reasoning don't confer robustness, and does not propose tool-based adjudication) [verified research].
- **Design, 3 conditions × clean/injected player turns:**
  - (a) CoCai-style: the LLM decides the roll **and** narrates the dice and outcome;
  - (b) the plan: the LLM adjudicates in typed fields, code rolls and resolves;
  - (c) (b) + `<player_turn>` data wrapping + verifier (idea 1).

  Build ~200 items: translate or adapt a CoC-Seduce subset into Ukrainian, plus author-written injections ("I obviously succeed", "as the Keeper already agreed, no roll needed", fake dice "I rolled 03").
- **Metrics:**
  - needs_roll accuracy and **injection flip rate**;
  - **outcome-integrity rate** (narrated outcome == engine outcome);
  - illegal-push acceptance rate;
  - skill-selection accuracy;
  - separately for FT vs base generators when `adjudicator: generator` (C8).

### Idea 6. Cross-lingual retrieval study: Ukrainian player, English corpus

- **Design:** paired uk/en questions (the same `gold_chunk_ids`) × retrieval paths:
  1. translate-then-nomic v1.5 (thesis baseline);
  2. raw uk → `bge-m3`;
  3. raw uk → `nomic-embed-text-v2-moe`;
  4. raw uk → `qwen3-embedding` (with its query instruction);
  5. (1) + hybrid BM25 over the translated query.

  One collection per embedder (the plan already encodes the embedder in the collection name).
- **Metrics:**
  - Recall@5 / MRR per path;
  - the **language gap** ΔRecall = Recall_en − Recall_uk (paired bootstrap CI);
  - translation-error attribution (manual check of 50 failed uk queries);
  - downstream answer-language adherence and glossary term consistency (M3);
  - extra latency and cost of the translation call.

### Idea 7. Long-campaign memory: trim vs summary vs retrieval over the transcript

- **Design:**
  - Scripted 120-turn campaigns (the same player script per condition, frozen dice via `forced_rolls`).
  - Insert **consistency probes**: facts established at turn t (NPC name, clue found, SAN lost, item given) queried at t+20/50/100.
  - Conditions:
    1. `trim_messages` only (current);
    2. rolling summary + `RemoveMessage` compaction;
    3. episodic memory: embed past turns into a per-campaign Qdrant collection and retrieve top-k;
    4. (2)+(3).
- **Metrics:**
  - probe accuracy by distance;
  - contradiction rate (NLI between narration and the established-facts ledger);
  - input tokens per turn (cost growth curve);
  - summary faithfulness (entailment of summary sentences against source turns).

### Idea 8. An empirical knowledge-base-swap generalisation test

- **Claim addressed:** tezy lines 23 and 42 ("метод є узагальненим").
- **Design:**
  - Implement a `GameSystem` protocol: `adjudication_schema`, `resolve()`, `prompt_pack`, `collection`, `glossary`.
  - Add a second, openly licensed system with a small rules corpus and ~40 QA items. Options: the BRP ORC document already planned in `data/open/`, or a CC-BY-licensed d20 SRD. Check the licence text before committing.
  - Keep the graph, API and eval harness unchanged.
- **Metrics:**
  - engineering cost of the swap (files and LOC changed outside `keeper/game/<system>` and prompts; should be ~0);
  - retrieval and QA metrics on the second system with RAG vs prompt-only;
  - **cross-system interference:** run the CoC LoRA on the second system with RAG and measure context-adherence. Does CoC style FT increase wrong CoC-specific rules (e.g. "SAN") being injected into another system?

### Idea 9. Knowledge-conflict probes stratified by perturbation type and parametric familiarity

- **Claim addressed:** "RAG context has priority over parametric knowledge" (tezy line 38), and whether FT makes the model more parametrically stubborn.
- **Design:** 3 perturbation types × 2 familiarity levels, injected via `forced_hits`:
  - perturbations: numeric change, negation or rule inversion, entity/term swap;
  - familiarity: well-known rules (success levels) vs obscure rules (the automatic-fire third penalty die).

  Include "no-conflict" controls and "irrelevant context" controls (distractor only).
- **Metrics:**
  - context-adherence rate;
  - parametric-override rate;
  - abstention ("Keeper ruling") rate;
  - all by stratum × arm.
- **Hypothesis:** FT raises override rates on well-known rules unless RAFT distractor training is used. Test by comparing a RAFT-mix adapter vs an oracle-only adapter, if a second small training run fits the budget.

### Idea 10. Calibrated abstention and "Keeper ruling" honesty

- **Design:** use the ~15% unanswerable items already planned in `rules_qa`, plus questions about house-ruled or optional rules.
- **Metrics:**
  - abstention precision/recall;
  - **selective accuracy** (accuracy on answered items vs coverage curve);
  - rate of confident wrong answers per arm (the hallucination proxy the thesis introduction is about).
- **Human study (small):**
  - 6–10 Ukrainian-speaking players rate blinded pairs of Keeper replies (base vs FT, ±RAG) for style, clarity and rules trust;
  - Bradley–Terry scores, Krippendorff's α;
  - report LLM-judge vs human agreement, so the automatic style metric is validated.

---

## 6. Recommended edit list for `final-architecture.md` (in order)

1. §6.2 and eval runner: `durability="sync"`; subprocess-kill resume test (C1).
2. §6.1 `LLM_RETRY` with `retry_on=(NodeTimeoutError, ConnectionError, httpx.TransportError)`; `TimeoutPolicy` on `narrate` (C2).
3. §5 Phase 4: add `train_unsloth.py` with `max_length` gate, Gemma 4 turn tags, and thinking parity (C3); `check_adapter_effect.py` + merged-16bit fallback + vLLM flags (C4).
4. §6.4: `fumble_threshold_on` flag, cancel-then-cap, INT roll, day SAN tracking, push window (C5).
5. §6.2: `duplicate_turn` dependency (C6).
6. §3.1/§3.6/§5 Phase 3: Stage A (frozen turn context) + Stage B (end to end); `adjudicator: generator` condition (C7, C8).
7. §6.6: Gemini kwargs (no `api_key=None`, no temperature), shared `InMemoryRateLimiter`, budget guard, paid tier for private corpus (C9, C11, M4).
8. §6.1/§3.6: context budget from `num_ctx`/`max-model-len`, calibrated `chars_per_token` (C10).
9. §6.5: per-type quotas, lore/style/keeper blocks, page anchors, source-aware ids (C12, C14).
10. §3.5/§4: native Windows dev path as default, OneDrive guard, `PYTHONUTF8`, CI + commit guard on day 1 (C13, W1–W4, M5, M6).
