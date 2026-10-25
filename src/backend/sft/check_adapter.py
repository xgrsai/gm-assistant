"""Adapter-effect check and model card for the Phase 4 fine-tuned Keeper (docs/critique.md C4, plan step S8).

    cd src/backend
    uv run python -m sft.check_adapter                 # run the check in Ollama, write the results and the model card
    uv run python -m sft.check_adapter --card-only     # rebuild only the card from the saved results (no Ollama)

Serving (C4 fallback): this Ollama rejects `ADAPTER`, so the LoRA is merged into the full-precision
Qwen/Qwen2.5-3B-Instruct and quantized to Q4_K_M (`training/merge_and_quantize.py`, one llama.cpp version), and the
base arm is the same Qwen2.5-3B-Instruct converted and quantized the same way. The two GGUFs, `gguf_meta.json` and the
training output live in the adapter directory; `python -m sft.ollama_models modelfiles` writes the two Modelfiles
(official `qwen2.5:3b` Modelfile with only the FROM line replaced) and prints the two `ollama create` commands.

A fine-tuned model can load without an error and still behave like the base (C4), and then the "FT" arm silently
equals the base arm. So before any FT arm runs, ~20 prompts from `artifacts/sft/coc7e-v1/val.jsonl` (the chat WITHOUT
the final Keeper reply; never gold evaluation items) go through the base model and the fine-tuned model with
`temperature: 0` and a fixed seed, straight through Ollama's `/api/chat` (the app's sampling options and LangChain are
not involved).

Gates (a failed gate exits with 1; the results and the card are written all the same):
  determinism  the first `--repeat` prompts are sent to the base model a second time, after the whole first pass (so
               nothing is served from the prompt cache). Their replies must be identical (`--min-stable`, default
               1.0): if the base model is not deterministic, "FT differs from base" proves nothing.
  effect       the share of prompts whose FT reply differs from the base reply is at least `--min-differ` (0.8, C4).
               A pair counts only if both replies are non-empty: an empty reply is a serving problem, not an effect.
  template     `/api/show` returns the same chat template for the base model, the FT model and the official
               `--official` model (`qwen2.5:3b`): the arms differ only in the weights.
  weights      the weights blob behind each tag (the `FROM` line of the `/api/show` modelfile) is the sha256 of the
               matching GGUF file in the adapter directory, the two blobs differ, and no model has an ADAPTER layer.
  parity       the two GGUF headers have the same tensor count, and `model_info` (+ family / size / quantization) of
               base and FT agree except for `general.*` (`--parity-ignore PREFIX` adds more): a difference in the
               conversion settings (rope base, context length, EOS id, ...) would confound the pair the thesis
               compares. The same comparison of base vs the official model is INFORMATIONAL only
               (`parity.official_differences`, printed as a warning): its blob was converted long ago by another
               llama.cpp, so harmless metadata differences are expected and never fail the gate or lower the status.
A check that cannot be read from the API (older Ollama, no `modelfile` / `model_info` field, no official model pulled)
is reported as NOT VERIFIED and does not fail the run; only a definite mismatch does. An empty template is a value
(it differs from a real one), not an unknown.

The card's `status` is "passed" only when every gate passed AND was verified; "passed_unverified" when some check
was NOT VERIFIED or a consistency check of the merge was waived (`gguf_meta.json` `waived_mismatches` from
`merge_and_quantize.py --allow-mismatch`, or an adapter that is not the one `train_meta.json` records; both are listed
in the card's top-level `waived_mismatches`); evals/model_cards.py accepts only "passed" unless --allow-failed-card;
"failed" otherwise. If Ollama fails midway, the replies received so far are written to
`check_adapter.incomplete.json` marked `"incomplete": true` (an existing complete results file, which the committed
card's `results_file.sha256` refers to, is left untouched; no card is written) and the exit code is 2.

Informational (they go to the card, never fail the run): marker deltas base vs FT vs the reference target - the
share of 🎲-line replies among prompts with a roll (and the spurious ones among prompts without), the Cyrillic share
for Ukrainian prompts, the share of replies with an `[R#]` citation, the reply length - and whether the PARAMETER
(stop tokens) and SYSTEM settings of the three models agree.

Files:
  <adapter-dir>/check_adapter.json                  full results incl. every reply (git-ignored, with the adapter)
  training/model_cards/<adapter-dir name>.json      the model card (small, committed; read by evals/model_cards.py)
Exit codes: 0 all gates passed, 1 a gate failed, 2 the check could not run (missing files/models, Ollama down, data
that differs from what the model was trained on).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import struct
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from keeper import citations
from server.settings import REPO_ROOT, get_settings
from sft import gates

DEFAULT_OFFICIAL = "qwen2.5:3b"  # the library model whose Modelfile (template, stop tokens) both arms reuse
DEFAULT_BASE = "qwen2.5-3b-selfq4:v1"  # Qwen2.5-3B-Instruct converted and quantized by us, without the LoRA
DEFAULT_FT = "keeper-qwen25-3b:sft-v1"  # the same, with the LoRA merged in first
DEFAULT_ADAPTER_DIR = REPO_ROOT / "adapters" / "keeper-qwen25-3b-sft-v1"
DEFAULT_VAL = REPO_ROOT / "artifacts" / "sft" / "coc7e-v1" / "val.jsonl"
DEFAULT_SPLIT = REPO_ROOT / "data" / "sft" / "coc7e" / "section_split.v1.json"
CARDS_DIR = REPO_ROOT / "training" / "model_cards"

# Output of training/merge_and_quantize.py, expected in the adapter directory.
FT_GGUF = "keeper-qwen25-3b-sft-v1.Q4_K_M.gguf"
BASE_GGUF = "qwen2.5-3b-instruct.Q4_K_M.gguf"
GGUF_META = "gguf_meta.json"
# The first serving attempt (LoRA converted with llama.cpp and attached with ADAPTER); Ollama refuses ADAPTER now.
SUPERSEDED_LORA_GGUF = {
    "file": "adapter.gguf",
    "llama_cpp_commit": "71255fefc092b0344f2acddb891388814ae961be",
    "command": "convert_lora_to_gguf.py --base-model-id Qwen/Qwen2.5-3B-Instruct --outtype f16 "
               "--outfile adapter.gguf adapter/",
    "status": "superseded and not used for any result: Ollama no longer supports ADAPTER",
}  # fmt: skip

RESULTS_SCHEMA = 2  # 2: `status` (passed | passed_unverified | failed) was added
CARD_SCHEMA = 1
SELECTION_SEED = 1927  # fixed: the chosen prompts do not depend on --seed (decoding) or on the order of val.jsonl
DICE = "🎲"
CYRILLIC = re.compile("[" + chr(0x0400) + "-" + chr(0x04FF) + "]")
LATIN = re.compile(r"[A-Za-z]")
WORD = re.compile(r"\w+")
MODELFILE_LINE = re.compile(r"^(FROM|ADAPTER)[ \t]+(.+?)[ \t]*$", re.M)  # '# FROM qwen2.5:3b' comments do not match
BLOB_DIGEST = re.compile(r"sha256[-:]([0-9a-f]{64})")
SHA256_IN_TEXT = re.compile(r"[0-9a-f]{64}")
PARITY_IGNORE = ("general.",)  # model_info key prefixes allowed to differ (file name, quantization version, ...)
DETAIL_KEYS = ("family", "parameter_size", "quantization_level")  # `details` of /api/show compared next to model_info
UK_MIN_CYRILLIC_SHARE = 0.5  # language of a reference reply; the dataset lints guarantee >= 0.7 (uk) and <= 0.05 (en)
MARKER_KEYS = (
    "dice_line_rate_with_roll", "dice_line_rate_without_roll", "cyrillic_share_uk", "cyrillic_share_en",
    "citation_rate", "citation_rate_when_reference_cites", "words_mean", "words_median", "chars_mean",
    "truncated", "empty",
)  # fmt: skip


class CheckError(RuntimeError):
    """The check cannot run or its inputs are inconsistent (exit code 2)."""


class OllamaError(RuntimeError):
    """Ollama is unreachable or answered with an error."""


# ---------------------------------------------------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reply:
    content: str
    done_reason: str | None = None
    eval_count: int | None = None


class OllamaApi(Protocol):
    """What the check needs from Ollama; tests pass a fake."""

    def chat(self, model: str, messages: list[dict[str, str]], options: dict[str, Any]) -> Reply: ...

    def show(self, model: str) -> dict[str, Any]: ...

    def tags(self) -> list[dict[str, Any]]: ...

    def version(self) -> str | None: ...


class OllamaClient:
    """Plain httpx client for the four Ollama endpoints used here (`/api/chat|show|tags|version`)."""

    def __init__(self, base_url: str, timeout: float = 900.0, transport: httpx.BaseTransport | None = None):
        # the first request loads the model into memory; a long prompt on a small GPU takes minutes, hence the timeout
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout, connect=10.0), transport=transport
        )

    def close(self) -> None:
        self._http.close()

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = self._http.request(method, path, json=body)
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"cannot connect to Ollama at {self._http.base_url} ({exc}); start the Ollama app or pass --ollama-url"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaError(f"Ollama did not answer {path} in time ({type(exc).__name__}); see --timeout") from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"request to Ollama {path} failed: {type(exc).__name__}: {exc}") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("error") or response.text
            except ValueError:
                detail = response.text
            raise OllamaError(f"Ollama {path} returned {response.status_code}: {str(detail)[:300]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError(f"Ollama {path} returned something that is not JSON: {response.text[:200]!r}") from exc
        if not isinstance(data, dict):
            raise OllamaError(f"Ollama {path} returned {type(data).__name__}, expected a JSON object")
        return data

    def chat(self, model: str, messages: list[dict[str, str]], options: dict[str, Any]) -> Reply:
        body = {"model": model, "messages": messages, "stream": False, "options": options}
        data = self._call("POST", "/api/chat", body)
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaError(f"Ollama /api/chat answer has no message.content: {str(data)[:200]}")
        return Reply(message["content"], data.get("done_reason"), data.get("eval_count"))

    def show(self, model: str) -> dict[str, Any]:
        return self._call("POST", "/api/show", {"model": model})

    def tags(self) -> list[dict[str, Any]]:
        models = self._call("GET", "/api/tags").get("models")
        return [m for m in models if isinstance(m, dict)] if isinstance(models, list) else []

    def version(self) -> str | None:
        try:
            version = self._call("GET", "/api/version").get("version")
        except OllamaError:
            return None
        return version if isinstance(version, str) else None


# ---------------------------------------------------------------------------------------------------------------------
# Prompts and reply markers
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    id: str
    prompt: list[dict[str, str]]  # system, history, user: the chat without the final Keeper reply
    reference: str  # the target reply of the dataset (what the model was trained to write)
    language: str  # "uk" | "en", read from the reference
    has_roll: bool  # the reference has a 🎲 line; the dataset lint makes that equal to "the example has rolls"


def cyrillic_share(text: str) -> float | None:
    """Cyrillic letters over Cyrillic + Latin letters (None when the text has neither)."""
    cyr, lat = len(CYRILLIC.findall(text)), len(LATIN.findall(text))
    return cyr / (cyr + lat) if cyr + lat else None


def dice_lines(text: str) -> int:
    """Lines that start with 🎲 (the keeper-v2 FORMAT puts the roll on its own line)."""
    return sum(1 for line in text.splitlines() if line.lstrip().startswith(DICE))


def reply_metrics(text: str) -> dict[str, Any]:
    share = cyrillic_share(text)
    return {
        "dice_lines": dice_lines(text),
        "citations": len(citations.cited_markers(text)),
        "cyrillic_share": None if share is None else round(share, 4),
        "words": len(WORD.findall(text)),
        "chars": len(text),
    }


def load_cases(val_path: Path, manifest: dict[str, Any]) -> list[Case]:
    """The val rows as cases; the ids come from `manifest.ids.val` (same order as the file)."""
    rows = gates.load_jsonl(val_path)
    gates.check_messages(rows, "val")
    ids = (manifest.get("ids") or {}).get("val")
    if not isinstance(ids, list) or len(ids) != len(rows):
        raise CheckError(
            f"manifest ids.val has {len(ids) if isinstance(ids, list) else 'no'} entries, {val_path.name} has "
            f"{len(rows)} rows; rebuild the dataset or pass the manifest that belongs to this val file"
        )
    cases = []
    for example_id, row in zip(ids, rows, strict=True):
        messages = row["messages"]
        reference = messages[-1]["content"]
        share = cyrillic_share(reference)
        cases.append(
            Case(
                id=str(example_id),
                prompt=[{"role": m["role"], "content": m["content"]} for m in messages[:-1]],
                reference=reference,
                language="uk" if share is not None and share >= UK_MIN_CYRILLIC_SHARE else "en",
                has_roll=dice_lines(reference) > 0,
            )
        )
    return cases


def select_cases(cases: list[Case], n: int) -> list[Case]:
    """The first `n` cases by a seeded hash of the id: deterministic, mixes scenes and kinds, ignores file order."""

    def order(case: Case) -> tuple[str, str]:
        return hashlib.sha256(f"{SELECTION_SEED}:{case.id}".encode()).hexdigest(), case.id

    return sorted(cases, key=order)[: max(n, 0)]


# ---------------------------------------------------------------------------------------------------------------------
# Running the models
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckConfig:
    base: str = DEFAULT_BASE
    ft: str = DEFAULT_FT
    official: str = DEFAULT_OFFICIAL
    n: int = 20
    repeat: int = 5
    seed: int = 0
    num_ctx: int = 16384  # must hold the longest prompt (estimate <= 7000 tokens) or Ollama silently truncates it
    num_predict: int = 512  # the longest reference is ~360 tokens; the margin keeps the 🎲 line of a long reply
    min_differ: float = 0.8
    min_stable: float = 1.0
    parity_ignore: tuple[str, ...] = PARITY_IGNORE

    def options(self) -> dict[str, Any]:
        return {"temperature": 0, "seed": self.seed, "num_ctx": self.num_ctx, "num_predict": self.num_predict}


def reply_record(reply: Reply) -> dict[str, Any]:
    return {
        "text": reply.content,
        "done_reason": reply.done_reason,
        "eval_count": reply.eval_count,
        "metrics": reply_metrics(reply.content),
    }


def new_records(cases: list[Case]) -> list[dict[str, Any]]:
    """One record per prompt, filled in place by `generate_all` (so a run that dies midway still has what it got)."""
    return [
        {
            "id": c.id,
            "language": c.language,
            "has_roll": c.has_roll,
            "reference": {"text_sha256": gates.sha256_text(c.reference), "metrics": reply_metrics(c.reference)},
        }
        for c in cases
    ]


def generate_all(
    api: OllamaApi, cases: list[Case], cfg: CheckConfig, log: Callable[[str], None],
    records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:  # fmt: skip
    """base on every prompt, base again on the first `repeat` ones, then FT on every prompt.

    One model at a time (Ollama on a 6 GB GPU would swap two models back and forth otherwise), and the repeat comes
    after the whole first pass so that the second answer is computed from scratch rather than from the prompt cache.
    An OllamaError propagates; the replies received so far stay in `records` (pass the list to keep them).
    """
    records = new_records(cases) if records is None else records
    passes = [
        ("base", cfg.base, "base", list(range(len(cases)))),
        ("base again", cfg.base, "base_repeat", list(range(min(cfg.repeat, len(cases))))),
        ("ft", cfg.ft, "ft", list(range(len(cases)))),
    ]
    for label, model, key, indices in passes:
        for done, index in enumerate(indices, start=1):
            started = time.monotonic()
            reply = api.chat(model, cases[index].prompt, cfg.options())
            records[index][key] = reply_record(reply)
            note = "" if reply.done_reason in (None, "stop") else f" [{reply.done_reason}]"
            log(
                f"[{label} {done}/{len(indices)}] {cases[index].id}: {len(reply.content)} chars, "
                f"{time.monotonic() - started:.1f}s{note}"
            )
    for record in records:
        record["differs"] = replies_differ(record)
        again = record.get("base_repeat")
        record["base_stable"] = None if again is None else again["text"] == record["base"]["text"]
    return records


def replies_differ(record: dict[str, Any]) -> bool:
    base, ft = record["base"]["text"], record["ft"]["text"]
    return bool(base.strip()) and bool(ft.strip()) and base != ft


# ---------------------------------------------------------------------------------------------------------------------
# Serving facts from Ollama
# ---------------------------------------------------------------------------------------------------------------------


def normalize_name(name: str) -> str:
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def find_tag(tags: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    wanted = normalize_name(name)
    for tag in tags:
        if wanted in (normalize_name(str(tag.get("name", ""))), normalize_name(str(tag.get("model", "")))):
            return tag
    return None


def parse_modelfile(text: str) -> dict[str, Any]:
    """Blob digests of the `FROM` and `ADAPTER` lines of `ollama show --modelfile` (a path ending in sha256-<hex>)."""
    from_blob = None
    adapters: list[str] = []
    for kind, value in MODELFILE_LINE.findall(text or ""):
        found = BLOB_DIGEST.search(value)
        if kind == "FROM" and found:
            from_blob = found.group(1)
        elif kind == "ADAPTER" and found:
            adapters.append(found.group(1))
    return {"from_blob": from_blob, "adapter_blobs": adapters}


def describe_model(
    api: OllamaApi, name: str, tags: list[dict[str, Any]], *, required: bool = True
) -> dict[str, Any] | None:
    tag = find_tag(tags, name)
    if tag is None:
        if not required:
            return None
        raise CheckError(
            f"Ollama has no model {name!r}. Write the Modelfiles with `uv run python -m sft.ollama_models modelfiles` "
            "and run the two `ollama create` commands it prints (training/README.md, 'Перевірка адаптера і model card')"
        )
    show = api.show(name)
    modelfile = show.get("modelfile") if isinstance(show.get("modelfile"), str) else ""
    blobs = parse_modelfile(modelfile)
    template = show.get("template")
    model_info = show.get("model_info")
    return {
        "name": name,
        "digest": tag.get("digest"),
        "size_bytes": tag.get("size"),
        "details": show.get("details") or tag.get("details"),
        "from_blob": blobs["from_blob"],
        "adapter_blobs": blobs["adapter_blobs"],
        "has_adapter": bool(blobs["adapter_blobs"]) if modelfile else None,
        # "" is a value: an empty template differs from a real one, only a missing/non-string field is unknown
        "template_sha256": gates.sha256_text(template) if isinstance(template, str) else None,
        "parameters_sha256": gates.sha256_text(show.get("parameters") or ""),
        "system_sha256": gates.sha256_text(show.get("system") or ""),
        "model_info": model_info if isinstance(model_info, dict) else None,
    }


def _same(*values: Any) -> bool | None:
    """True/False when every value is known, None when one of them could not be read."""
    return None if any(v is None for v in values) else all(v == values[0] for v in values)


MISSING = "<missing>"


def _short(value: Any, limit: int = 80) -> Any:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value if len(text) <= limit else text[:limit] + f"... (sha256 {gates.sha256_text(text)[:12]})"


def comparable_info(model: dict[str, Any] | None, ignore: tuple[str, ...]) -> dict[str, Any] | None:
    """`model_info` of `/api/show` without the ignored key prefixes, plus the family / size / quantization details."""
    info = None if model is None else model.get("model_info")
    if not isinstance(info, dict):
        return None
    out = {key: value for key, value in info.items() if not key.startswith(ignore)}
    details = model.get("details") if isinstance(model.get("details"), dict) else {}
    out.update({f"details.{key}": details[key] for key in DETAIL_KEYS if key in details})
    return out


def info_differences(infos: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Keys whose value is not the same in every model of `infos` (a key one model lacks counts as different)."""
    out = []
    for key in sorted(set().union(*infos.values())):
        values = {role: info.get(key, MISSING) for role, info in infos.items()}
        if len({json.dumps(v, sort_keys=True, default=str) for v in values.values()}) > 1:
            out.append({"key": key, "values": {role: _short(v) for role, v in values.items()}})
    return out


def parity_info(
    base: dict[str, Any], ft: dict[str, Any], official: dict[str, Any] | None, tensor_counts: dict[str, int | None],
    ignore: tuple[str, ...],
) -> dict[str, Any]:  # fmt: skip
    """Do the two GGUFs have the same tensors and does Ollama report the same `model_info` for base and FT?

    Base vs FT is STRICT: they are the pair the thesis compares, so any difference in the conversion settings (rope
    base, context length, EOS id, ...) or in the tensor set is an error. `general.*` is ignored by default (file name,
    quantization version, ...), more prefixes via `ignore`. The official model is only a reference: its blob was
    converted long ago by another llama.cpp, so harmless metadata differences are likely. Its differences from the
    base model go to `official_differences`; they never fail the gate or lower the status.
    """
    models = (("base", base), ("ft", ft), ("official", official))
    infos = {role: comparable_info(model, ignore) for role, model in models}
    pair = {role: infos[role] for role in ("base", "ft") if infos[role] is not None}
    counts = [tensor_counts.get("base"), tensor_counts.get("ft")]
    official_compared = infos["base"] is not None and infos["official"] is not None
    return {
        "tensor_counts": dict(tensor_counts),
        "tensor_counts_equal": None if None in counts else counts[0] == counts[1],
        "ignored_prefixes": list(ignore),
        "compared": sorted(pair),
        "model_info_differences": info_differences(pair) if len(pair) == 2 else None,  # None = could not compare
        "official_compared": official_compared,
        "official_differences": (
            info_differences({"base": infos["base"], "official": infos["official"]}) if official_compared else None
        ),
    }


def serving_info(
    api: OllamaApi, cfg: CheckConfig, expected_sha256: dict[str, str], tensor_counts: dict[str, int | None],
) -> dict[str, Any]:  # fmt: skip
    """What Ollama reports about the base, FT and official models; `expected_sha256` = the GGUFs' hashes by role."""
    tags = api.tags()
    base, ft = describe_model(api, cfg.base, tags), describe_model(api, cfg.ft, tags)
    official = describe_model(api, cfg.official, tags, required=False)  # missing = NOT VERIFIED, not an error

    def with_official(field: str) -> bool | None:
        return None if official is None else _same(base[field], ft[field], official[field])

    return {
        "ollama_version": api.version(),
        "base": base,
        "ft": ft,
        "official": official,
        "expected_sha256": dict(expected_sha256),
        "template_equal": _same(base["template_sha256"], ft["template_sha256"]),
        "template_official_equal": with_official("template_sha256"),
        "parameters_equal": _same(base["parameters_sha256"], ft["parameters_sha256"]),
        "parameters_official_equal": with_official("parameters_sha256"),
        "system_equal": _same(base["system_sha256"], ft["system_sha256"]),
        "parity": parity_info(base, ft, official, tensor_counts, cfg.parity_ignore),
    }


# ---------------------------------------------------------------------------------------------------------------------
# Gates and marker summaries
# ---------------------------------------------------------------------------------------------------------------------


def _share(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize_markers(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Marker rates of one source (`base`, `ft` or `reference`) over the selected prompts."""
    rows = [(r, r[key]["metrics"], r[key].get("done_reason"), r[key].get("text")) for r in records]
    with_roll = [m for r, m, _, _ in rows if r["has_roll"]]
    without_roll = [m for r, m, _, _ in rows if not r["has_roll"]]
    cyrillic = {
        lang: [m["cyrillic_share"] for r, m, _, _ in rows if r["language"] == lang and m["cyrillic_share"] is not None]
        for lang in ("uk", "en")
    }
    cites = [m for r, m, _, _ in rows if r["reference"]["metrics"]["citations"] > 0]
    words = [m["words"] for _, m, _, _ in rows]
    return {
        "n": len(rows),
        "dice_line_rate_with_roll": _share(sum(m["dice_lines"] > 0 for m in with_roll), len(with_roll)),
        "dice_line_rate_without_roll": _share(sum(m["dice_lines"] > 0 for m in without_roll), len(without_roll)),
        "cyrillic_share_uk": _mean(cyrillic["uk"]),
        "cyrillic_share_en": _mean(cyrillic["en"]),
        "citation_rate": _share(sum(m["citations"] > 0 for _, m, _, _ in rows), len(rows)),
        "citation_rate_when_reference_cites": _share(sum(m["citations"] > 0 for m in cites), len(cites)),
        "words_mean": round(sum(words) / len(words), 1) if words else None,
        "words_median": round(statistics.median(words), 1) if words else None,
        "chars_mean": _mean([m["chars"] for _, m, _, _ in rows]),
        "truncated": sum(done == "length" for _, _, done, _ in rows),  # hit num_predict: the tail (🎲 line) may be cut
        "empty": sum(text is not None and not text.strip() for _, _, _, text in rows),
    }


def marker_deltas(markers: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    def diff(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else round(a - b, 4)

    pairs = {"ft_minus_base": ("ft", "base"), "ft_minus_reference": ("ft", "reference"),
             "base_minus_reference": ("base", "reference")}  # fmt: skip
    return {name: {k: diff(markers[a][k], markers[b][k]) for k in MARKER_KEYS} for name, (a, b) in pairs.items()}


def serving_gates(serving: dict[str, Any]) -> dict[str, dict]:
    """Gates that need only `/api/show` and the GGUF headers: they hold (or fail) before any reply is generated."""
    base, ft, expected = serving["base"], serving["ft"], serving["expected_sha256"]
    base_match, ft_match = _same(base["from_blob"], expected["base"]), _same(ft["from_blob"], expected["ft"])
    distinct = None if base["from_blob"] is None or ft["from_blob"] is None else base["from_blob"] != ft["from_blob"]
    adapter_layers = bool(base["adapter_blobs"] or ft["adapter_blobs"])
    equal, official_equal = serving["template_equal"], serving["template_official_equal"]
    parity = serving["parity"]
    counts_equal, differences = parity["tensor_counts_equal"], parity["model_info_differences"]
    return {
        "template": {"passed": equal is not False and official_equal is not False, "equal": equal,
                     "official_equal": official_equal, "verified": equal is not None and official_equal is not None},
        "weights": {"passed": base_match is not False and ft_match is not False and distinct is not False
                    and not adapter_layers,
                    "base_blob_matches_gguf": base_match, "ft_blob_matches_gguf": ft_match, "blobs_distinct": distinct,
                    "adapter_layers": adapter_layers, "verified": base_match is not None and ft_match is not None},
        "parity": {"passed": counts_equal is not False and not differences, "tensor_counts": parity["tensor_counts"],
                   "tensor_counts_equal": counts_equal, "differences": differences or [],
                   "ignored_prefixes": parity["ignored_prefixes"], "compared": parity["compared"],
                   "official_differences": parity["official_differences"],  # informational: never fails the gate
                   "verified": counts_equal is not None and differences is not None},
    }  # fmt: skip


def overall_status(gate_results: dict[str, dict]) -> str:
    """"failed" if a gate failed; "passed_unverified" if some check could not be read (NOT VERIFIED); else "passed"."""
    if not all(g["passed"] for g in gate_results.values()):
        return "failed"
    return "passed" if all(g.get("verified", True) is not False for g in gate_results.values()) else "passed_unverified"


def unverified_gates(gate_results: dict[str, dict]) -> list[str]:
    return [name for name, g in gate_results.items() if g["passed"] and g.get("verified", True) is False]


def evaluate_gates(records: list[dict[str, Any]], cfg: CheckConfig, serving: dict[str, Any]) -> dict[str, dict]:
    repeated = [r for r in records if "base_repeat" in r]
    unstable = [r["id"] for r in repeated if r["base_repeat"]["text"] != r["base"]["text"]]
    stable_share = (len(repeated) - len(unstable)) / len(repeated) if repeated else 0.0
    differing = [r["id"] for r in records if replies_differ(r)]
    differ_share = len(differing) / len(records) if records else 0.0
    return {
        "determinism": {
            "passed": bool(repeated) and stable_share >= cfg.min_stable - 1e-9,
            "checked": len(repeated), "identical": len(repeated) - len(unstable), "share": round(stable_share, 4),
            "min_share": cfg.min_stable, "unstable_ids": unstable,
        },
        "effect": {
            "passed": bool(records) and differ_share >= cfg.min_differ - 1e-9,
            "differ": len(differing), "n": len(records), "share": round(differ_share, 4),
            "min_share": cfg.min_differ, "same_ids": [r["id"] for r in records if r["id"] not in differing],
        },
        **serving_gates(serving),
    }  # fmt: skip


def failure_messages(gate_results: dict[str, dict], cfg: CheckConfig) -> list[str]:
    """One clear sentence per failed gate (`gate_results` may hold only the serving gates)."""
    out = []
    g = {name: gate for name, gate in gate_results.items() if not gate["passed"]}
    if "determinism" in g:
        d = g["determinism"]
        out.append(
            f"FAIL determinism: {d['identical']}/{d['checked']} repeated base replies are identical "
            f"(need {d['min_share']:.0%}; differing: {d['unstable_ids']}). With temperature 0 and a fixed seed the "
            "base model must answer the same prompt the same way, otherwise 'FT differs from base' proves nothing. "
            "Close other programs that use the GPU or the model, keep --num-ctx unchanged between runs, and re-run."
        )
    if "effect" in g:
        e = g["effect"]
        out.append(
            f"FAIL effect: only {e['differ']}/{e['n']} FT replies differ from the base replies ({e['share']:.0%}, need "
            f"{e['min_share']:.0%}). The fine-tuning looks IGNORED (docs/critique.md C4): check that "
            f"`ollama show --modelfile {cfg.ft}` points at the merged FT GGUF and not at the base, that the merge in "
            "training/merge_and_quantize.py really applied the LoRA, and re-create the model with `ollama create`."
        )
    if "template" in g:
        t = g["template"]
        out.append(
            f"FAIL template: {cfg.base}, {cfg.ft} and the official {cfg.official} must share one chat template "
            f"(base == FT: {t['equal']}, all three equal: {t['official_equal']}). Create both models from the "
            "Modelfiles that `python -m sft.ollama_models modelfiles` writes (the official one, only FROM changed)."
        )
    if "weights" in g:
        w = g["weights"]
        reasons = []
        if w["base_blob_matches_gguf"] is False:
            reasons.append(f"{cfg.base} was not created from {BASE_GGUF}")
        if w["ft_blob_matches_gguf"] is False:
            reasons.append(f"{cfg.ft} was not created from {FT_GGUF}")
        if w["blobs_distinct"] is False:
            reasons.append("both tags point at the same weights blob")
        if w["adapter_layers"]:
            reasons.append("a model has an ADAPTER layer (the weights are merged; Modelfiles must not use ADAPTER)")
        out.append(
            f"FAIL weights: {'; '.join(reasons)}. Re-create the models from the files in the adapter directory "
            "(`ollama create ... -f Modelfile.base|ft`) and make sure the GGUFs were not re-downloaded or replaced."
        )
    if "parity" in g:
        p = g["parity"]
        parts = []
        if p["tensor_counts_equal"] is False:
            counts = p["tensor_counts"]
            parts.append(f"the GGUFs have different tensor counts (base {counts['base']}, ft {counts['ft']})")
        if p["differences"]:
            shown = "; ".join(
                f"{d['key']} ({', '.join(f'{role}={value}' for role, value in d['values'].items())})"
                for d in p["differences"][:8]
            )
            more = f" and {len(p['differences']) - 8} more" if len(p["differences"]) > 8 else ""
            parts.append(f"model_info of base and FT differs in: {shown}{more}")
        out.append(
            f"FAIL parity: {'; '.join(parts)}. Base and FT are the pair the thesis compares, so they must come from "
            "one conversion with identical settings (rope base, context length, EOS id, tensors, ...). Redo the "
            "conversion, or, after inspecting the keys, accept benign ones with --parity-ignore <key prefix>."
        )
    return out


def composition(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "language": {lang: sum(r["language"] == lang for r in records) for lang in ("uk", "en")},
        "with_roll": sum(r["has_roll"] for r in records),
        "without_roll": sum(not r["has_roll"] for r in records),
        "reference_cites": sum(r["reference"]["metrics"]["citations"] > 0 for r in records),
    }


def settings_of(cfg: CheckConfig, n: int) -> dict[str, Any]:
    return {
        "base": cfg.base, "ft": cfg.ft, "official": cfg.official, "n": n, "repeat": min(cfg.repeat, n),
        "temperature": 0, "seed": cfg.seed, "num_ctx": cfg.num_ctx, "num_predict": cfg.num_predict,
        "min_differ": cfg.min_differ, "min_stable": cfg.min_stable, "selection_seed": SELECTION_SEED,
        "parity_ignore": list(cfg.parity_ignore),
    }


def incomplete_results(
    records: list[dict[str, Any]], cfg: CheckConfig, serving: dict[str, Any], *, error: str, val_sha256: str,
    val_file: str, elapsed_s: float, now: datetime | None = None,
) -> dict[str, Any]:  # fmt: skip
    """What a run that died midway (Ollama error) leaves behind: the replies so far, no gates, never a card."""
    return {
        "schema": RESULTS_SCHEMA,
        "incomplete": True,
        "error": error,
        "created_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed_s, 1),
        "settings": settings_of(cfg, len(records)),
        "data": {"val_file": val_file, "val_sha256": val_sha256},
        "serving": serving,
        "passed": False,
        "status": "incomplete",
        "cases": records,
    }


def assemble_results(
    records: list[dict[str, Any]], cfg: CheckConfig, serving: dict[str, Any], *, val_sha256: str, val_file: str,
    elapsed_s: float, now: datetime | None = None,
) -> dict[str, Any]:  # fmt: skip
    gate_results = evaluate_gates(records, cfg, serving)
    markers = {key: summarize_markers(records, key) for key in ("base", "ft", "reference")}
    return {
        "schema": RESULTS_SCHEMA,
        "created_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed_s, 1),
        "settings": settings_of(cfg, len(records)),
        "data": {"val_file": val_file, "val_sha256": val_sha256},
        "serving": serving,
        "selection": {"ids": [r["id"] for r in records], "composition": composition(records)},
        "gates": gate_results,
        "passed": all(g["passed"] for g in gate_results.values()),
        "status": overall_status(gate_results),
        "markers": markers,
        "deltas": marker_deltas(markers),
        "cases": records,
    }  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# Adapter files, training facts and the model card
# ---------------------------------------------------------------------------------------------------------------------


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def git_state(path: Path) -> dict[str, Any]:
    """Commit and dirtiness of the checkout containing `path` (None/None without git or outside a checkout)."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(dirty)}


def gguf_header(path: Path) -> dict[str, int]:
    """Version and counts from the GGUF header (magic, u32 version, u64 tensor count, u64 metadata key count)."""
    with path.open("rb") as fh:
        head = fh.read(24)
    if len(head) < 24 or head[:4] != b"GGUF":
        raise CheckError(f"{path} is not a GGUF file (no GGUF magic); re-download it from the merge_and_quantize run")
    version, tensors, keys = struct.unpack("<IQQ", head[4:])
    return {"version": version, "tensor_count": tensors, "metadata_kv_count": keys}


def safetensors_tensor_count(path: Path) -> int | None:
    """Number of tensors in a safetensors file (u64 header size + JSON header), None if the header is unreadable."""
    try:
        with path.open("rb") as fh:
            (size,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(size).decode("utf-8"))
    except (OSError, struct.error, ValueError):
        return None
    return sum(1 for name in header if name != "__metadata__") if isinstance(header, dict) else None


def _load_json(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise CheckError(f"{what} not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise CheckError(f"{what} {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CheckError(f"{what} {path} must hold a JSON object")
    return data


def _dig(data: Any, *keys: str) -> Any:
    for key in keys:
        data = data.get(key) if isinstance(data, dict) else None
    return data


def _find_key(data: Any, name: str) -> Any:
    """First value stored under `name` anywhere in nested dicts/lists (gguf_meta.json is written by another script)."""
    if isinstance(data, dict):
        if data.get(name) is not None:
            return data[name]
        children = list(data.values())
    elif isinstance(data, list):
        children = data
    else:
        return None
    for child in children:
        found = _find_key(child, name)
        if found is not None:
            return found
    return None


def _strings(data: Any) -> list[str]:
    if isinstance(data, str):
        return [data]
    children = list(data.values()) if isinstance(data, dict) else data if isinstance(data, list) else []
    return [s for child in children for s in _strings(child)]


def llama_cpp_commit(meta: dict[str, Any]) -> str | None:
    """merge_and_quantize.py writes `llama_cpp.commit`; a few other spellings are accepted too."""
    nested = meta.get("llama_cpp")
    if isinstance(nested, dict) and isinstance(nested.get("commit"), str):
        return nested["commit"]
    for key in ("llama_cpp_commit", "llama_cpp_sha", "llama_cpp_git_sha"):
        if isinstance(value := _find_key(meta, key), str):
            return value
    nested = _find_key(meta, "llama_cpp")
    if isinstance(nested, str):
        return nested
    for key in ("commit", "git_sha", "sha"):
        if isinstance(value := _find_key(nested, key), str):
            return value
    return None


def merge_dtype(meta: dict[str, Any]) -> str | None:
    value = _dig(meta, "merge", "dtype")
    if isinstance(value, str):
        return value
    return next((v for k in ("merge_dtype", "merged_dtype") if isinstance(v := _find_key(meta, k), str)), None)


def gguf_file_names(adapter_dir: Path) -> dict[str, str]:
    """GGUF file names by role: the ones `gguf_meta.json` records (another --quant), else the default Q4_K_M names."""
    meta_path = adapter_dir / GGUF_META
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    except ValueError:
        meta = {}
    names = {}
    for role, default in (("ft", FT_GGUF), ("base", BASE_GGUF)):
        recorded = _dig(meta, "outputs", role, "file")
        names[role] = recorded if isinstance(recorded, str) and recorded.endswith(".gguf") else default
    return names


def gguf_models_facts(adapter_dir: Path, adapter_sha256: str) -> dict[str, Any]:
    """The merged FT GGUF and the self-quantized base GGUF with the provenance `gguf_meta.json` records for them."""
    meta_path = adapter_dir / GGUF_META
    meta = _load_json(meta_path, f"{GGUF_META} (written by training/merge_and_quantize.py)")
    merged_adapter = _dig(meta, "adapter", "sha256")
    if merged_adapter is not None and merged_adapter != adapter_sha256:
        raise CheckError(
            f"{GGUF_META} says the models were merged from an adapter with sha256 {str(merged_adapter)[:12]}..., but "
            f"adapter/adapter_model.safetensors is {adapter_sha256[:12]}...; use the files of the same training run"
        )
    anywhere = [s.lower() for s in _strings(meta) if SHA256_IN_TEXT.search(s.lower())]
    names = gguf_file_names(adapter_dir)
    files: dict[str, Any] = {}  # by file name: {"role", "sha256", "size", ...}; evals/model_cards.py reads the shape
    shas: dict[str, str] = {}
    for role in ("ft", "base"):
        name = names[role]
        path = adapter_dir / name
        if not path.is_file():
            raise CheckError(
                f"{path} not found; download the output of training/merge_and_quantize.py into {adapter_dir} "
                f"({FT_GGUF}, {BASE_GGUF} and {GGUF_META}; training/README.md)"
            )
        sha = gates.sha256_file(path)
        recorded = _dig(meta, "outputs", role, "sha256")
        if isinstance(recorded, str):
            in_meta: bool | None = recorded.lower() == sha
        else:  # another meta layout: any digest in the file counts, no digest at all = unknown
            in_meta = any(sha in text for text in anywhere) if anywhere else None
        if in_meta is False:
            raise CheckError(
                f"{name} (sha256 {sha[:12]}...) is not the file {GGUF_META} describes (it records other digests); "
                "download the GGUFs and the meta file of the same merge_and_quantize run"
            )
        shas[role] = sha
        files[name] = {"role": role, "sha256": sha, "size": path.stat().st_size, "recorded_in_gguf_meta": in_meta,
                       **gguf_header(path)}  # fmt: skip
    return {
        "method": "LoRA merged into the full-precision base, then quantized; the base arm was converted and "
                  "quantized the same way without the LoRA, with one llama.cpp version (docs/critique.md C4 fallback)",
        "source": GGUF_META,
        "sha256": shas["ft"],  # the served fine-tuned weights
        "base_sha256": shas["base"],
        "roles": {"ft": names["ft"], "base": names["base"]},
        "files": files,
        "meta_file": {"path": rel(meta_path), "sha256": gates.sha256_file(meta_path)},
        "merged_from_this_adapter": None if merged_adapter is None else True,
        "llama_cpp_commit": llama_cpp_commit(meta),
        "quant": _dig(meta, "quantization", "type"),
        "merge_dtype": merge_dtype(meta),
        "versions": _dig(meta, "environment", "versions"),
        "base_revision": _dig(meta, "base", "revision"),
        "meta": meta,  # verbatim: everything merge_and_quantize.py recorded (checks, commands, timings)
        "waived_mismatches": meta["waived_mismatches"] if isinstance(meta.get("waived_mismatches"), dict) else {},
        "superseded": SUPERSEDED_LORA_GGUF,
    }  # fmt: skip


def collect_facts(
    adapter_dir: Path, val_path: Path, manifest_path: Path, split_path: Path, git: dict[str, Any],
) -> dict[str, Any]:  # fmt: skip
    """Everything about the model and its data that needs no Ollama; refuses data that is not the training data."""
    safetensors_path = adapter_dir / "adapter" / "adapter_model.safetensors"
    if not safetensors_path.is_file():
        raise CheckError(f"{safetensors_path} not found; copy the training output into {adapter_dir} (see README)")
    meta_path = adapter_dir / "train_meta.json"
    meta = _load_json(meta_path, "train_meta.json")
    manifest = _load_json(manifest_path, "data manifest.json")
    if not val_path.is_file():
        raise CheckError(f"{val_path} not found; build the dataset first (`uv run python -m sft.build build`)")

    data_files = {"val.jsonl": val_path}
    train_path = val_path.with_name("train.jsonl")
    if train_path.is_file():
        data_files["train.jsonl"] = train_path
    try:
        data_hashes = gates.verify_data_hashes(manifest, data_files)
    except gates.GateError as exc:
        raise CheckError(str(exc)) from exc
    trained_on = {"val.jsonl": _dig(meta, "data", "val_sha256"), "train.jsonl": _dig(meta, "data", "train_sha256")}
    for name, digest in data_hashes.items():
        if trained_on[name] is not None and trained_on[name] != digest:
            raise CheckError(
                f"{name} ({digest[:12]}...) is not the file the adapter was trained with "
                f"(train_meta.json: {trained_on[name][:12]}...); the check must run on the training split"
            )
    split_sha = gates.sha256_file(split_path) if split_path.is_file() else None
    manifest_split_sha = _dig(manifest, "split", "sha256")
    if split_sha is not None and manifest_split_sha is not None and split_sha != manifest_split_sha:
        raise CheckError(
            f"{rel(split_path)} ({split_sha[:12]}...) differs from the split the data was built with "
            f"({manifest_split_sha[:12]}...); the split file was changed after the dataset was built"
        )

    safetensors_sha = gates.sha256_file(safetensors_path)
    st_tensors = safetensors_tensor_count(safetensors_path)
    recorded_st = _dig(meta, "adapter", "sha256")
    peft_path = adapter_dir / "adapter" / "adapter_config.json"
    peft = _load_json(peft_path, "adapter_config.json") if peft_path.is_file() else {}
    template = meta.get("chat_template") if isinstance(meta.get("chat_template"), dict) else {}
    local_manifest_copy = adapter_dir / "manifest.json"

    gguf = gguf_models_facts(adapter_dir, safetensors_sha)  # last of the checks: it hashes two ~2 GB files
    waived = dict(gguf["waived_mismatches"])  # consistency checks the author chose to go on without
    if recorded_st is not None and recorded_st != safetensors_sha:
        waived["train_meta_adapter_sha256"] = (
            f"adapter/adapter_model.safetensors ({safetensors_sha[:12]}...) is not the adapter that train_meta.json "
            f"records ({str(recorded_st)[:12]}...)"
        )

    return {
        "waived_mismatches": waived,
        "adapter": {
            "dir": rel(adapter_dir),
            "files": {
                "adapter/adapter_model.safetensors": {
                    "sha256": safetensors_sha, "bytes": safetensors_path.stat().st_size, "tensor_count": st_tensors,
                },
            },
            "safetensors_sha256_matches_train_meta": None if recorded_st is None else recorded_st == safetensors_sha,
            "train_meta_sha256": gates.sha256_file(meta_path),
            "selection": _dig(meta, "adapter", "selection"),
            "saved_at_step": _dig(meta, "adapter", "saved_at_step"),
            "peft_config": {k: peft.get(k) for k in (
                "peft_type", "r", "lora_alpha", "lora_dropout", "target_modules", "base_model_name_or_path"
            )},
        },
        "gguf": gguf,
        "training": {
            "base": meta.get("base"),
            "seed": meta.get("seed"),
            "hyperparameters": meta.get("hyperparameters"),
            "versions": meta.get("versions"),
            "hardware": meta.get("hardware"),
            "losses": meta.get("losses"),
            "gates": meta.get("gates"),
            "chat_template": {k: template.get(k) for k in (
                "sha256", "sha256_transformers_autotokenizer", "same_as_transformers", "source", "bos_token",
                "eos_token",
            )},
            "code": meta.get("code"),
        },
        "data": {
            "dataset": manifest.get("dataset"),
            "train": {"sha256": _dig(manifest, "files", "train.jsonl", "sha256"),
                      "examples": _dig(manifest, "files", "train.jsonl", "examples")},
            "val": {"sha256": _dig(manifest, "files", "val.jsonl", "sha256"),
                    "examples": _dig(manifest, "files", "val.jsonl", "examples")},
            "verified_locally": sorted(data_hashes),
            "matches_train_meta": {
                name: None if trained_on[name] is None else trained_on[name] == digest
                for name, digest in data_hashes.items()
            },
            "manifest": {
                "file": rel(manifest_path), "sha256": gates.sha256_file(manifest_path),
                "same_as_adapter_dir_copy": (
                    gates.sha256_file(local_manifest_copy) == gates.sha256_file(manifest_path)
                    if local_manifest_copy.is_file() else None
                ),
            },
            "git_commit": manifest.get("git_sha"),
            "git_dirty": manifest.get("git_dirty"),
            "prompt": manifest.get("prompt"),
            "gold_hashes": manifest.get("gold_hashes"),
            "counts": manifest.get("counts"),
            "sources": manifest.get("sources"),
        },
        "split": {
            "path": rel(split_path), "sha256": split_sha, "manifest_sha256": manifest_split_sha,
            "seed": _dig(manifest, "split", "seed"),
            "corpus_chunk_ids_hash": _dig(manifest, "split", "corpus_chunk_ids_hash"),
        },
        "git": git,
    }  # fmt: skip


def build_card(
    facts: dict[str, Any], results: dict[str, Any], *, results_path: Path, results_sha256: str, name: str,
    now: datetime | None = None,
) -> dict[str, Any]:  # fmt: skip
    """The model card: the training/data facts, the serving facts of the run and the check results (without replies)."""
    serving = results["serving"]
    check = {k: v for k, v in results.items() if k not in ("cases", "serving", "schema")}
    check["cases"] = [
        {"id": c["id"], "language": c["language"], "has_roll": c["has_roll"], "differs": c["differs"],
         "base_stable": c["base_stable"]}
        for c in results["cases"]
    ]  # fmt: skip
    check["results_file"] = {"path": rel(results_path), "sha256": results_sha256}
    waived = facts["waived_mismatches"]
    status = results["status"]
    if status == "passed" and waived:  # a waived check is a check that did not pass: downstream must opt in
        status = "passed_unverified"
    return {
        "schema": CARD_SCHEMA,
        "name": name,
        "generated_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "status": status,  # passed | passed_unverified | failed (evals/model_cards.py wants "passed")
        "waived_mismatches": waived,
        "adapter": facts["adapter"],
        "gguf": facts["gguf"],
        "training": facts["training"],
        "data": facts["data"],
        "split": facts["split"],
        "ollama_digests": {"ft": serving["ft"]["digest"], "base": serving["base"]["digest"]},  # evals/model_cards.py
        "serving": serving,
        "check_adapter": check,
        "git": facts["git"],
    }


# ---------------------------------------------------------------------------------------------------------------------
# Report and CLI
# ---------------------------------------------------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:g}"


def format_report(results: dict[str, Any]) -> str:
    s, g = results["settings"], results["gates"]
    comp = results["selection"]["composition"]
    lines = [
        f"adapter check: {s['ft']} vs {s['base']}: {s['n']} prompts (uk {comp['language']['uk']}, en "
        f"{comp['language']['en']}, with roll {comp['with_roll']}), temperature 0, seed {s['seed']}",
        f"  determinism: {g['determinism']['identical']}/{g['determinism']['checked']} repeated base replies identical "
        f"[{'ok' if g['determinism']['passed'] else 'FAIL'}]",
        f"  effect:      {g['effect']['differ']}/{g['effect']['n']} FT replies differ from base "
        f"({g['effect']['share']:.0%}, need {g['effect']['min_share']:.0%}) "
        f"[{'ok' if g['effect']['passed'] else 'FAIL'}]",
    ]
    labels = {"template": "same chat template", "weights": "blobs = GGUF files", "parity": "tensors / model_info"}
    for key, label in labels.items():
        gate = g[key]
        state = "FAIL" if not gate["passed"] else ("ok" if gate["verified"] else "NOT VERIFIED")
        lines.append(f"  {label + ':':<21}[{state}]")
    lines.append("markers (base / ft / reference):")
    m = results["markers"]
    for key in MARKER_KEYS:
        lines.append(f"  {key:<36}{_fmt(m['base'][key]):>8} {_fmt(m['ft'][key]):>8} {_fmt(m['reference'][key]):>8}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base", default=DEFAULT_BASE, help="Ollama tag of the self-quantized base model")
    p.add_argument("--ft", default=DEFAULT_FT, help="Ollama tag of the merged fine-tuned model")
    p.add_argument("--official", default=DEFAULT_OFFICIAL, help="official library model: the template reference")
    p.add_argument("--n", type=int, default=20, help="number of val prompts (deterministic choice)")
    p.add_argument("--repeat", type=int, default=5, help="base prompts sent twice for the determinism gate")
    p.add_argument("--seed", type=int, default=0, help="Ollama sampling seed")
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--num-predict", type=int, default=512)
    p.add_argument("--min-differ", type=float, default=0.8, help="effect gate: minimum share of differing replies")
    p.add_argument("--min-stable", type=float, default=1.0, help="determinism gate: minimum share of identical repeats")
    p.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR,
                   help="training output + the two GGUFs and gguf_meta.json of merge_and_quantize.py")
    p.add_argument("--val", type=Path, default=DEFAULT_VAL, help="val.jsonl (train.jsonl and manifest.json beside it)")
    p.add_argument("--manifest", type=Path, help="data manifest (default: manifest.json next to --val)")
    p.add_argument("--split", type=Path, default=DEFAULT_SPLIT, help="frozen section split file")
    p.add_argument("--results", type=Path, help="results JSON (default: <adapter-dir>/check_adapter.json)")
    p.add_argument("--card", type=Path, help="model card JSON (default: training/model_cards/<adapter-dir name>.json)")
    p.add_argument("--ollama-url", help="default: OLLAMA_BASE_URL / http://localhost:11434")
    p.add_argument("--timeout", type=float, default=900.0, help="seconds per Ollama request")
    p.add_argument("--parity-ignore", action="append", default=[], metavar="PREFIX",
                   help="also ignore model_info keys with this prefix in the parity gate (general.* always is)")
    p.add_argument("--card-only", action="store_true", help="write only the card, from the saved results (no Ollama)")
    return p.parse_args(argv)


def incomplete_path_for(results_path: Path) -> Path:
    """`check_adapter.json` -> `check_adapter.incomplete.json`: where a run that died midway leaves its replies."""
    return results_path.with_name(f"{results_path.stem}.incomplete{results_path.suffix}")


def write_json(path: Path, payload: dict[str, Any]) -> str:
    """Write `payload` as UTF-8 / LF JSON; return the sha256 of the written bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gates.dump_json(path, payload)
    return gates.sha256_file(path)


def run(args: argparse.Namespace, client_factory: Callable[..., OllamaApi], log: Callable[[str], None]) -> int:
    cfg = CheckConfig(
        base=args.base, ft=args.ft, official=args.official, n=args.n, repeat=args.repeat, seed=args.seed,
        num_ctx=args.num_ctx, num_predict=args.num_predict, min_differ=args.min_differ, min_stable=args.min_stable,
        parity_ignore=PARITY_IGNORE + tuple(args.parity_ignore),
    )  # fmt: skip
    if cfg.n < 1 or cfg.repeat < 1:
        raise CheckError("--n and --repeat must be at least 1")
    adapter_dir = args.adapter_dir
    manifest_path = args.manifest or args.val.with_name("manifest.json")
    results_path = args.results or adapter_dir / "check_adapter.json"
    card_path = args.card or CARDS_DIR / f"{adapter_dir.name}.json"
    facts = collect_facts(adapter_dir, args.val, manifest_path, args.split, git_state(REPO_ROOT))
    val_sha256 = facts["data"]["val"]["sha256"]
    expected = {"base": facts["gguf"]["base_sha256"], "ft": facts["gguf"]["sha256"]}
    files, roles = facts["gguf"]["files"], facts["gguf"]["roles"]
    tensor_counts = {role: files[roles[role]]["tensor_count"] for role in ("base", "ft")}

    if args.card_only:
        results = _load_json(results_path, "saved results (run the check first)")
        if results.get("schema") != RESULTS_SCHEMA:
            raise CheckError(
                f"{results_path} has results schema {results.get('schema')!r}, but this version writes schema "
                f"{RESULTS_SCHEMA} (the gate status was added in 2); run the check again"
            )
        if results.get("incomplete"):
            raise CheckError(f"{results_path} is from a run that stopped midway ({results.get('error')}); run it again")
        if _dig(results, "data", "val_sha256") != val_sha256:
            raise CheckError(f"{results_path} was made with another val.jsonl; run the check again")
        if _dig(results, "serving", "expected_sha256") != expected:
            raise CheckError(f"the GGUF files in {adapter_dir} are not the ones {results_path} was produced with")
        results_sha = gates.sha256_file(results_path)
        saved = results["settings"]
        cfg = replace(cfg, **{k: saved[k] for k in ("base", "ft", "official", "min_differ", "min_stable")})
    else:
        cases = select_cases(load_cases(args.val, _load_json(manifest_path, "data manifest.json")), cfg.n)
        cfg = replace(cfg, n=len(cases), repeat=min(cfg.repeat, len(cases)))
        url = args.ollama_url or get_settings().ollama_base_url
        api = client_factory(url, args.timeout)
        serving = serving_info(api, cfg, expected, tensor_counts)
        log(f"Ollama {serving['ollama_version'] or '?'} at {url}")
        log(f"  base {cfg.base}: {serving['base']['digest']}")
        log(f"  ft   {cfg.ft}: {serving['ft']['digest']}")
        early = failure_messages(serving_gates(serving), cfg)
        for line in early:
            log(line)
        if early:
            log("the run continues so that the results are complete (Ctrl-C to stop)")
        log(f"{len(cases)} prompts: base x{len(cases)}, base again x{cfg.repeat}, ft x{len(cases)}")
        started = time.monotonic()
        records = new_records(cases)
        try:
            generate_all(api, cases, cfg, log, records)
        except OllamaError as exc:  # keep what was generated (about 10 minutes of GPU time), then fail as usual
            partial = incomplete_results(
                records, cfg, serving, error=str(exc), val_sha256=val_sha256, val_file=rel(args.val),
                elapsed_s=time.monotonic() - started,
            )
            incomplete_path = incomplete_path_for(results_path)
            write_json(incomplete_path, partial)  # not results_path: that file may be the one the committed card cites
            log(f"wrote {incomplete_path} (INCOMPLETE: the run stopped midway; no model card)")
            raise
        results = assemble_results(
            records, cfg, serving, val_sha256=val_sha256, val_file=rel(args.val), elapsed_s=time.monotonic() - started
        )
        results_sha = write_json(results_path, results)
        incomplete_path_for(results_path).unlink(missing_ok=True)  # a stale partial run is not needed any more
        log(f"wrote {results_path}")

    card = build_card(facts, results, results_path=results_path, results_sha256=results_sha, name=adapter_dir.name)
    write_json(card_path, card)
    log(f"wrote {card_path}")
    log(format_report(results))
    for line in failure_messages(results["gates"], cfg):
        log(line)
    for kind, problem in card["waived_mismatches"].items():
        log(f"WARNING: waived consistency check [{kind}]: {problem}")
    if card["status"] == "passed_unverified":
        reasons = [f"{name}: NOT VERIFIED" for name in unverified_gates(results["gates"])]
        reasons += [f"waived check: {kind}" for kind in card["waived_mismatches"]]
        log(
            f"card status is 'passed_unverified' ({'; '.join(reasons)}), not 'passed': evals/model_cards.py accepts "
            "only 'passed' unless --allow-failed-card is given. Fix what could not be verified (the gates marked "
            "NOT VERIFIED above), redo the merge without --allow-mismatch if a check was waived, and run this again."
        )
    serving = card["serving"]
    if False in (serving["parameters_equal"], serving["parameters_official_equal"], serving["system_equal"]):
        log("WARNING: base, FT and the official model differ in PARAMETER or SYSTEM settings; "
            "the Modelfiles should come from `python -m sft.ollama_models modelfiles`")
    if serving["official"] is None:
        log(f"WARNING: {cfg.official} is not in Ollama, so the template and model_info were not compared with it")
    official_diff = serving["parity"]["official_differences"]
    if official_diff:
        keys = ", ".join(d["key"] for d in official_diff[:8]) + (" ..." if len(official_diff) > 8 else "")
        log(
            f"WARNING (informational, not a failure): model_info of {cfg.base} differs from the official "
            f"{cfg.official} in {len(official_diff)} key(s): {keys}. Expected when the official blob was converted by "
            "another llama.cpp; the values are in the card (serving.parity.official_differences)."
        )
    if card["gguf"]["llama_cpp_commit"] is None:
        log(f"WARNING: no llama.cpp commit found in {GGUF_META}; the card lacks it")
    if any(f["recorded_in_gguf_meta"] is None for f in card["gguf"]["files"].values()):
        log(f"WARNING: {GGUF_META} records no sha256 digests, so the GGUF files were not matched against it")
    return 0 if results["passed"] else 1


def main(argv: list[str] | None = None, *, client_factory: Callable[..., OllamaApi] | None = None) -> int:
    args = parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    try:
        return run(args, client_factory or OllamaClient, log)
    except (CheckError, OllamaError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):  # the Windows console may be cp1251; replies carry Ukrainian and 🎲
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
