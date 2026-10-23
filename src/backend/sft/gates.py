"""Pure pre-/post-flight checks for the Phase 4 LoRA training script (`training/train_unsloth.py`).

stdlib only: the training script runs on Colab/Kaggle without the backend's dependencies and imports this file by path
(`sys.path`), and the tests exercise it without torch, unsloth or a GPU. Nothing here touches a model; the script feeds
in token counts, label arrays and strings.

The gates exist because SFT fails silently (docs/critique.md C3): an over-long example is truncated from the right
(the answer is lost), a wrong turn tag masks every label, and a chat template can inject its own default system prompt
instead of ours. Each gate raises `GateError` with a message that says what to fix.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IGNORE_INDEX = -100  # label value the loss skips (HF / TRL convention)
META_SCHEMA = 1
SHA256_RE = re.compile(r"[0-9a-f]{64}")

# Text a chat template injects when the conversation has no system message. Seeing it in a rendered example means the
# system prompt from the data did not reach the model.
DEFAULT_SYSTEM_MARKERS = ("You are Qwen, created by Alibaba Cloud",)


class GateError(RuntimeError):
    """A training pre-flight gate failed; the script must stop before spending GPU time."""


@dataclass(frozen=True)
class FamilyTags:
    """Turn markers `train_on_responses_only` uses to find the assistant tokens in the rendered text."""

    family: str
    instruction_part: str
    response_part: str


# Keyed by model family. A wrong string silently masks every label (critique C3.2): the masking gate catches it.
FAMILY_TAGS: dict[str, FamilyTags] = {
    "qwen": FamilyTags("qwen", "<|im_start|>user\n", "<|im_start|>assistant\n"),
    "gemma4": FamilyTags("gemma4", "<|turn>user\n", "<|turn>model\n"),
}

# Substring of the (lower-cased) base model id -> family. First match wins.
_FAMILY_HINTS = (("qwen", "qwen"), ("gemma-4", "gemma4"), ("gemma4", "gemma4"))


def family_for_base(base_id: str) -> FamilyTags:
    """Pick the masking tags for a base model id such as `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`."""
    lowered = base_id.lower()
    for hint, family in _FAMILY_HINTS:
        if hint in lowered:
            return FAMILY_TAGS[family]
    raise GateError(
        f"no response-masking tags for base {base_id!r}; add its family to FAMILY_TAGS (known: {sorted(FAMILY_TAGS)})"
    )


# ---------------------------------------------------------------------------------------------------------------------
# Data files and manifest
# ---------------------------------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise GateError(f"{path}: line {number} is not valid JSON: {exc}") from exc
    return rows


def manifest_sha256(manifest: dict[str, Any], filename: str) -> str:
    """The sha256 the manifest records for `filename`: `{"files": {"train.jsonl": {"sha256": "<64 hex>"}, ...}}`."""
    files = manifest.get("files")
    entry = files.get(filename) if isinstance(files, dict) else None
    value = entry.get("sha256") if isinstance(entry, dict) else None
    if value is None:
        raise GateError(
            f"manifest has no files.{filename}.sha256; expected {{'files': {{'{filename}': {{'sha256': ...}}}}}}, "
            f"top-level keys: {sorted(manifest)}"
        )
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise GateError(f"manifest files.{filename}.sha256 is not a 64-character lowercase hex digest: {value!r}")
    return value


def verify_data_hashes(manifest: dict[str, Any], files: dict[str, Path]) -> dict[str, str]:
    """Hash every data file and compare with the manifest; return `{filename: sha256}` or raise on any mismatch."""
    actual: dict[str, str] = {}
    problems = []
    for filename, path in files.items():
        actual[filename] = sha256_file(path)
        expected = manifest_sha256(manifest, filename)
        if actual[filename] != expected:
            problems.append(f"{filename}: manifest {expected[:12]}..., file {actual[filename][:12]}...")
    if problems:
        raise GateError("data does not match manifest.json (rebuild or re-upload): " + "; ".join(problems))
    return actual


# ---------------------------------------------------------------------------------------------------------------------
# Message / rendering gates
# ---------------------------------------------------------------------------------------------------------------------


def check_messages(rows: list[dict[str, Any]], split: str) -> None:
    """Every row is `{"messages": [system, (user, assistant)*, user, assistant]}` with non-empty string contents."""
    if not rows:
        raise GateError(f"{split}: no examples")
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            raise GateError(f"{split}[{index}]: expected a 'messages' list with system, user and assistant turns")
        roles = [m.get("role") for m in messages]
        expected = ["system"] + ["user", "assistant"] * ((len(messages) - 1) // 2)
        if len(messages) % 2 == 0 or roles != expected:
            raise GateError(
                f"{split}[{index}]: roles must be system, then alternating user/assistant ending with assistant; "
                f"got {roles}"
            )
        if not all(isinstance(m.get("content"), str) and m["content"].strip() for m in messages):
            raise GateError(f"{split}[{index}]: every message needs non-empty string content")
        for turn, message in enumerate(messages):
            content = message["content"]
            if unicodedata.normalize("NFC", content) != content:
                raise GateError(
                    f"{split}[{index}].messages[{turn}]: content is not NFC-normalised "
                    "(the tokenizer normalises it, so the trained text would differ from the data; fix it in the build)"
                )
            if message["role"] == "assistant" and content != content.strip():
                raise GateError(
                    f"{split}[{index}].messages[{turn}]: assistant reply has leading/trailing whitespace "
                    "(the supervised span would not match the data; strip it in the build)"
                )


def finalize_text(text: str, bos_token: str | None, eos_token: str | None) -> str:
    """Make a rendered chat the exact string the trainer should tokenize.

    - A leading BOS is dropped: the tokenizer/trainer adds it itself (otherwise it is doubled).
    - Whitespace after the closing EOS (Qwen ends a turn with `<|im_end|>\\n`) is dropped: TRL appends the EOS token
      to a text field that does not end with it, which would add a second `<|im_end|>` after the newline.
    """
    if bos_token and text.startswith(bos_token):
        text = text[len(bos_token) :]
    stripped = text.rstrip()
    if eos_token and stripped.endswith(eos_token):
        return stripped
    return text


def check_rendered(rows: list[dict[str, Any]], texts: list[str], eos_token: str | None, split: str) -> None:
    """The rendered text carries OUR system prompt (not the template's default) and ends with the EOS token."""
    if len(rows) != len(texts):
        raise GateError(f"{split}: {len(rows)} examples but {len(texts)} rendered texts")
    for index, (row, text) in enumerate(zip(rows, texts, strict=True)):
        system = row["messages"][0]["content"]
        if system not in text:
            raise GateError(
                f"{split}[{index}]: the data's system prompt is missing from the rendered chat (template replaced it?)"
            )
        for marker in DEFAULT_SYSTEM_MARKERS:
            if marker in text and marker not in system:
                raise GateError(f"{split}[{index}]: the chat template injected its default system prompt ({marker!r})")
        if eos_token and not text.endswith(eos_token):
            raise GateError(f"{split}[{index}]: rendered text does not end with {eos_token!r}")


# ---------------------------------------------------------------------------------------------------------------------
# Length and masking gates
# ---------------------------------------------------------------------------------------------------------------------


def length_stats(lengths: list[int]) -> dict[str, int]:
    ordered = sorted(lengths)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
    }


def check_lengths(lengths: list[int], max_len: int, split: str) -> dict[str, int]:
    """Stop if ANY example is longer than `max_len` tokens: the trainer would cut its tail (the answer)."""
    if not lengths:
        raise GateError(f"{split}: no examples to measure")
    stats = length_stats(lengths)
    too_long = [i for i, n in enumerate(lengths) if n > max_len]
    if too_long:
        raise GateError(
            f"{split}: {len(too_long)} of {len(lengths)} examples exceed max_len={max_len} tokens "
            f"(longest {stats['max']}, first indices {too_long[:5]}); "
            "shorten the context in the data build or raise --max-len"
        )
    return stats


def check_masking(labels: list[list[int]], split: str) -> dict[str, float | int]:
    """Every example must have at least one supervised label and at least one masked one.

    Zero supervised labels: the response tags did not match (or the answer was truncated), so the example teaches
    nothing. Zero masked labels: masking was not applied, so the prompt (system + sources) would be trained on too.
    """
    if not labels:
        raise GateError(f"{split}: no examples to check")
    unsupervised = [i for i, row in enumerate(labels) if not any(t != IGNORE_INDEX for t in row)]
    unmasked = [i for i, row in enumerate(labels) if row and all(t != IGNORE_INDEX for t in row)]
    if unsupervised:
        raise GateError(
            f"{split}: {len(unsupervised)} of {len(labels)} examples have no supervised token "
            f"(first {unsupervised[:5]}); "
            "check the response-masking tags for this model family and that nothing is truncated"
        )
    if unmasked:
        raise GateError(
            f"{split}: {len(unmasked)} of {len(labels)} examples have no masked token (first {unmasked[:5]}); "
            "response-only masking was not applied"
        )
    supervised = [sum(t != IGNORE_INDEX for t in row) for row in labels]
    total = sum(len(row) for row in labels)
    return {
        "n": len(labels),
        "supervised_tokens": sum(supervised),
        "total_tokens": total,
        "supervised_fraction": round(sum(supervised) / total, 4) if total else 0.0,
        "min_supervised_per_example": min(supervised),
    }


def find_last_subsequence(sequence: list[int], needle: list[int]) -> int:
    """Start index of the LAST occurrence of `needle` in `sequence`, or -1."""
    if not needle:
        raise ValueError("needle must not be empty")
    for start in range(len(sequence) - len(needle), -1, -1):
        if list(sequence[start : start + len(needle)]) == needle:
            return start
    return -1


def mask_all_but_last_response(input_ids: list[int], response_ids: list[int]) -> list[int]:
    """Labels that supervise only the FINAL assistant reply.

    Everything up to and including the last occurrence of the response tag (the "assistant" turn opener) is -100;
    the reply tokens after it are kept. Earlier assistant turns (history) are masked: the system prompt carries the
    knowledge base, rolls and adjudication of the CURRENT turn only, so a history reply is not a valid target under it.
    No tag found: everything is masked, which the masking gate then reports.
    """
    start = find_last_subsequence(list(input_ids), list(response_ids))
    if start < 0:
        return [IGNORE_INDEX] * len(input_ids)
    cut = start + len(response_ids)
    return [IGNORE_INDEX] * cut + list(input_ids[cut:])


def check_final_response_only(
    labels: list[list[int]],
    input_ids: list[list[int]],
    decode: Callable[[list[int]], str],
    final_replies: list[str],
    eos_token: str,
    split: str,
) -> None:
    """Each example has ONE contiguous supervised span, it ends at the last token, and it decodes to reply + EOS."""
    if not (len(labels) == len(input_ids) == len(final_replies)):
        raise GateError(f"{split}: labels, input_ids and replies differ in count")
    for index, (row, ids, reply) in enumerate(zip(labels, input_ids, final_replies, strict=True)):
        if len(row) != len(ids):
            raise GateError(f"{split}[{index}]: {len(row)} labels for {len(ids)} input ids")
        positions = [i for i, t in enumerate(row) if t != IGNORE_INDEX]
        if not positions:
            raise GateError(f"{split}[{index}]: no supervised token")
        if positions != list(range(positions[0], positions[-1] + 1)) or positions[-1] != len(row) - 1:
            raise GateError(
                f"{split}[{index}]: supervised tokens are not one span ending at the last token "
                "(history replies are being trained on?)"
            )
        if any(row[i] != ids[i] for i in positions):
            raise GateError(f"{split}[{index}]: a label differs from its input id")
        decoded = decode([ids[i] for i in positions])
        expected = unicodedata.normalize("NFC", reply) + eos_token  # the Qwen2 tokenizer normalises to NFC
        if decoded != expected:
            detail = describe_mismatch(decoded, expected)
            raise GateError(f"{split}[{index}]: supervised text is not the final reply + EOS: {detail}")


def describe_mismatch(got: str, expected: str, radius: int = 20) -> str:
    """Index of the first differing character with +-`radius` characters of both strings around it."""
    at = next((i for i, (g, e) in enumerate(zip(got, expected, strict=False)) if g != e), min(len(got), len(expected)))
    lo, hi = max(0, at - radius), at + radius
    return (
        f"first difference at char {at} (got {len(got)} chars, expected {len(expected)}): "
        f"got {got[lo:hi]!r} vs expected {expected[lo:hi]!r}; hint: NFC? leading whitespace/newline? "
        "the chat template changed the reply?"
    )


def check_token_ends(input_ids: list[list[int]], eos_id: int, split: str) -> None:
    """Each tokenized example ends with exactly one EOS (TRL must not have appended a second one)."""
    for index, ids in enumerate(input_ids):
        if len(ids) < 2 or ids[-1] != eos_id or ids[-2] == eos_id:
            raise GateError(f"{split}[{index}]: tokenized example must end with exactly one EOS id {eos_id}")


def collator_labels_match(batch_labels: list[list[int]], gated_labels: list[list[int]]) -> None:
    """The labels the collator feeds the loss carry the same supervised tokens as the gated dataset labels.

    Padding is -100 as well, so compare the supervised tokens (and the batch row may only be longer, never shorter).
    """
    if len(batch_labels) != len(gated_labels):
        raise GateError("collator returned a different number of rows than it was given")
    for index, (got, want) in enumerate(zip(batch_labels, gated_labels, strict=True)):
        if len(got) < len(want) or [t for t in got if t != IGNORE_INDEX] != [t for t in want if t != IGNORE_INDEX]:
            raise GateError(f"collator labels of example {index} differ from the gated labels (collator ignores them?)")


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """The `checkpoint-<step>` directory with the highest step, or None."""
    found = []
    for path in Path(ckpt_dir).glob("checkpoint-*") if Path(ckpt_dir).exists() else []:
        suffix = path.name.removeprefix("checkpoint-")
        if path.is_dir() and suffix.isdigit():
            found.append((int(suffix), path))
    return max(found)[1] if found else None


# ---------------------------------------------------------------------------------------------------------------------
# Resume and meta
# ---------------------------------------------------------------------------------------------------------------------


def config_diff(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Keys whose values differ between the config a checkpoint was started with and the current one."""
    keys = sorted(set(saved) | set(current))
    return [f"{k}: {saved.get(k)!r} -> {current.get(k)!r}" for k in keys if saved.get(k) != current.get(k)]


def summarize_losses(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Train/eval loss curves from `trainer.state.log_history`, plus a plain 'did eval loss go down' flag."""
    train = [(e["step"], e["loss"]) for e in log_history if "loss" in e and "step" in e]
    evals = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e and "step" in e]
    summary: dict[str, Any] = {
        "train_loss_first": train[0][1] if train else None,
        "train_loss_last": train[-1][1] if train else None,
        "eval_loss_first": evals[0][1] if evals else None,
        "eval_loss_last": evals[-1][1] if evals else None,
        "eval_loss_best": min(v for _, v in evals) if evals else None,
        "eval_loss_best_step": min(evals, key=lambda e: e[1])[0] if evals else None,
        "eval_loss_curve": [{"step": s, "eval_loss": v} for s, v in evals],
    }
    summary["eval_loss_decreased"] = bool(evals) and len(evals) > 1 and evals[-1][1] < evals[0][1]
    return summary


def build_meta(
    *,
    base: dict[str, Any],
    data: dict[str, Any],
    hyperparameters: dict[str, Any],
    seed: int,
    versions: dict[str, str | None],
    hardware: dict[str, Any],
    chat_template: dict[str, Any],
    gates: dict[str, Any],
    losses: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble `train_meta.json`. Pure: the script gathers the facts, this only fixes the shape."""
    meta: dict[str, Any] = {
        "schema": META_SCHEMA,
        "base": base,
        "data": data,
        "hyperparameters": hyperparameters,
        "seed": seed,
        "versions": versions,
        "hardware": hardware,
        "chat_template": chat_template,
        "gates": gates,
        "losses": losses,
    }
    if extra:
        meta.update(extra)
    return meta


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
