"""Model cards of the Phase 4 models, and the checks that tie an Ollama tag to its card.

A card is a repo-relative JSON file named by `ModelProfile.model_card` (training/model_cards/<id>.json), written by
`python -m sft.check_adapter`. One card describes a PAIR of models, so two profiles point at it: the fine-tuned one
(`finetuned: true`, role "ft") and its base, the same model converted and quantized the same way without the LoRA
(role "base"). The harness reads from the card (keys as sft/check_adapter.py writes them):

- `status`: "passed" only when the adapter-effect and serving gates passed. Anything else stops a run (critique C4:
  the check "must pass before any FT arm is run"), unless the author overrides it with --allow-failed-card;
- `ollama_digests.ft` / `ollama_digests.base`: the Ollama digest of each role. Evals refuse to run when Ollama reports
  another digest for a tag, so a re-created or re-pulled tag cannot be mistaken for the model the card describes;
- the sha256 of the served weights of the role (`gguf.files[*]` with that `role`, else `gguf.sha256` for the fine-tuned
  model and `gguf.base_sha256` for the base), of the train and val files (`data.train.sha256`, `data.val.sha256`) and
  of the section split (`split.sha256`).

The run's config.json gets a `model_cards` entry per profile that has a card. The run fingerprint covers what
identifies the models (status, digest, weights, data, split) but not `card_sha256`: the card file also holds
timestamps and git state, so re-running the check on the same models must not orphan a run. Runs without such a
profile have no `model_cards` key.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from keeper.deps import LLMs
from keeper.llm import ModelProfile
from server.settings import REPO_ROOT

from .datasets import file_sha256

CHECK_COMMAND = "python -m sft.check_adapter"
MIN_DIGEST_CHARS = 12   # `ollama list` shows the first 12 hex digits; the API returns the whole sha256
PASSED = "passed"

# snapshot field -> card path
DATA_PATHS: dict[str, tuple[str, ...]] = {
    "train_sha256": ("data", "train", "sha256"),
    "val_sha256": ("data", "val", "sha256"),
    "split_sha256": ("split", "sha256"),
}
WEIGHTS_PATHS = {"ft": ("gguf", "sha256"), "base": ("gguf", "base_sha256")}   # if gguf.files lacks the role


class ModelCardError(Exception):
    """The card of a Phase 4 profile is missing or unusable."""


def has_card(profile: ModelProfile) -> bool:
    return profile.finetuned or profile.model_card is not None


def role(profile: ModelProfile) -> str:
    """Which model of the card the profile is: the fine-tuned one, or its base."""
    return "ft" if profile.finetuned else "base"


def load_card(name: str, profile: ModelProfile, root: Path = REPO_ROOT) -> dict:
    if not profile.model_card:
        raise ModelCardError(f"profile {name!r} is finetuned but has no model_card in config/models.yaml")
    path = root / profile.model_card
    if not path.is_file():
        raise ModelCardError(f"model card {profile.model_card} (profile {name!r}) not found; create the Ollama models "
                             f"and write the card with `{CHECK_COMMAND}`")
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ModelCardError(f"model card {profile.model_card} is not valid JSON: {e}") from e
    if not isinstance(card, dict):
        raise ModelCardError(f"model card {profile.model_card} must be a JSON object")
    return card


def _dig(card: dict, path: tuple[str, ...]):
    node = card
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
    return node


def card_digest(card: dict, profile: ModelProfile) -> str | None:
    return _dig(card, ("ollama_digests", role(profile))) or None


def weights_sha256(card: dict, profile: ModelProfile) -> str | None:
    """The sha256 of the profile's own GGUF (the fine-tuned one or the base one), never of the other role's."""
    files = _dig(card, ("gguf", "files"))
    for entry in files.values() if isinstance(files, dict) else ():
        if isinstance(entry, dict) and entry.get("role") == role(profile) and entry.get("sha256"):
            return entry["sha256"]
    return _dig(card, WEIGHTS_PATHS[role(profile)])


def snapshot(llms: LLMs, names: Iterable[str], root: Path = REPO_ROOT) -> dict[str, dict]:
    """`model_cards` for config.json: one entry per profile with a card among `names`; {} when there is none."""
    out: dict[str, dict] = {}
    for name in sorted(set(names)):
        profile = llms.profile(name)
        if not has_card(profile):
            continue
        card = load_card(name, profile, root)
        out[name] = {"card": profile.model_card, "role": role(profile), "status": card.get("status"),
                     "ollama_digest": card_digest(card, profile), "weights_sha256": weights_sha256(card, profile),
                     **{field: _dig(card, path) for field, path in DATA_PATHS.items()},
                     "card_sha256": file_sha256(root / profile.model_card)}   # informational, not in the fingerprint
    return out


# ------------------------------------------------------------------------------------------------- Ollama


def with_tag(model: str) -> str:
    return model if ":" in model else f"{model}:latest"   # what Ollama lists for an untagged name


def same_digest(a: str, b: str) -> bool:
    """Equal digests, either as the full sha256 or as the short id of `ollama list` (a prefix of at least 12 digits)."""
    a, b = (x.strip().lower().removeprefix("sha256:") for x in (a, b))
    short, full = sorted((a, b), key=len)
    return len(short) >= MIN_DIGEST_CHARS and full.startswith(short)


def ollama_listing(url: str, timeout: float = 10.0) -> dict[str, str]:
    """Tag -> digest of every model the Ollama server has (the ollama client is a langchain-ollama dependency)."""
    import ollama

    return {with_tag(m.model): m.digest for m in ollama.Client(host=url, timeout=timeout).list().models
            if m.model and m.digest}


def check_cards(llms: LLMs, names: Iterable[str], *, root: Path = REPO_ROOT,
                listing: Callable[[], Mapping[str, str]] | None = None,
                allow_failed: bool = False) -> tuple[list[str], list[str]]:
    """(errors, warnings) for the profiles that have a model card among `names`; both empty when all is well.

    Every card must exist, parse and have status "passed" (a failed one is only a warning with `allow_failed`). With
    `listing`, every such Ollama profile must also be served under the digest its card records for the profile's role
    (fine-tuned or base). A card shared by the pair is reported once.
    """
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    wanted: list[tuple[str, ModelProfile, str, str]] = []   # profile name, profile, digest, where it is written
    for name in sorted(set(names)):
        profile = llms.profile(name)
        if not has_card(profile):
            continue
        key = profile.model_card or name
        try:
            card = load_card(name, profile, root)
        except ModelCardError as e:
            if key not in seen:
                errors.append(str(e))
            seen.add(key)
            continue
        if key not in seen and card.get("status") != PASSED:
            message = (f"model card {profile.model_card} has status {card.get('status')!r}: the adapter check did not "
                       f"pass, so the fine-tuned model may not differ from its base; rerun `{CHECK_COMMAND}`")
            if allow_failed:
                warnings.append(f"{message} (continuing: --allow-failed-card)")
            else:
                errors.append(f"{message}, or pass --allow-failed-card to run anyway")
        seen.add(key)
        if listing is None or profile.provider != "ollama":
            continue
        digest = card_digest(card, profile)
        field = f"ollama_digests.{role(profile)}"
        if not digest:
            errors.append(f"model card {profile.model_card} has no {field}; rewrite it with `{CHECK_COMMAND}`")
            continue
        wanted.append((name, profile, digest, f"{profile.model_card} {field}"))
    if not wanted:
        return errors, warnings
    try:
        have = listing()
    except Exception as e:  # noqa: BLE001 - any transport failure means the digest cannot be verified
        return [*errors, f"cannot list the Ollama models to verify the digests: {type(e).__name__}: {e}"], warnings
    for name, profile, want, source in wanted:
        tag = with_tag(profile.model)
        got = have.get(tag)
        if got is None:
            errors.append(f"Ollama has no model {tag!r} (profile {name!r}); create it before the run")
        elif not same_digest(got, want):
            errors.append(f"Ollama model {tag!r} (profile {name!r}) has digest {got[:MIN_DIGEST_CHARS]} but "
                          f"{source} is {want[:MIN_DIGEST_CHARS]}: the tag was re-created or re-pulled since the card "
                          f"was written. Restore the original model, or re-verify it with `{CHECK_COMMAND}` and start "
                          "a new run name")
    return errors, warnings
