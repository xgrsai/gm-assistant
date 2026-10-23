"""Merge frozen inputs, targets and the author's review into the chat-format SFT dataset (train / val + manifest).

    cd src/backend
    uv run python -m sft.build build --version v1                  # lint everything, write artifacts/sft/coc7e-v1/
    uv run python -m sft.build build --version v1 --lint-only      # only lint (what S5 runs while writing targets)
    uv run python -m sft.build review-export --sample 100          # CSV for the author (artifacts/sft/coc7e-v1/)
    uv run python -m sft.build review-export --review data/sft/coc7e/review-v1.csv    # CSV verdicts -> reviews.v1.jsonl

Files (version `v1`; all JSONL, UTF-8, LF):
    data/sft/coc7e/inputs.v1.jsonl    SftInput    frozen by `sft.build inputs --write` (rolls, KB chunks, rendered chat)
    data/sft/coc7e/targets.v1.jsonl   SftTarget   the Keeper replies (S5)
    data/sft/coc7e/reviews.v1.jsonl   SftReview   derived from data/sft/coc7e/review-v1.csv by `review-export --review`
    artifacts/sft/coc7e-v1/{train,val}.jsonl + manifest.json    the output the Colab/Kaggle training script consumes

An example is `inputs[i].messages + [{"role": "assistant", "content": <final text>}]` where the final text is the
target, the reviewer's edit (verdict `edit`), and the example is dropped on verdict `reject`. A target whose
`input_hash` differs from its input is refused (the input was rebuilt after the target was written), and so is a
target WITHOUT an `input_hash` (`[no-input-hash]`, unless `--allow-no-input-hash`): staleness cannot be checked. A
review carries the `target_hash` of the text the reviewer saw. reviews.jsonl is a HISTORY keyed by (id, target_hash);
only the verdict for the CURRENT text applies. A target rewritten after it was reviewed is built as unreviewed, with a
warning and a manifest entry (`reviews.rewritten`), and `review-export` offers it again first; a review without a hash
is applied and counted in the manifest. `--review` refuses a CSV row whose hash is not the current text (it names the
CSV line) and merges the rest by (id, target_hash). The lints
run on the FINAL text, so an edit has to pass them too. EVERY problem of every example is listed
(`<id>: [code] reason`) and the exit code is 1; nothing is written unless all examples pass.

Lints (code in brackets):
  gate       sft.gates.check_messages on the finished chat: NFC, no outer whitespace on the reply, role order.
  input      the frozen input is stale: `input_hash` does not match its messages, or it was rendered with another
             prompt hash / prompt version / corpus than the current ones (rebuild the inputs and re-write the target).
  number     a number that appears nowhere the narrator could read it. Allowed: the system prompt from SCENE OPENING
             on (opening, keeper notes, INVESTIGATOR sheet, MECHANICS), the ORACLE passages only (distractor
             passages do not count: the model must not learn to use them), the first line (title, era) and the
             decade of every year in it ("1920-ті"), the player's text and the history. The HARD RULES / STYLE
             preamble is excluded, so "80-200 words" is not a licence to say "80". Whitelist: NUMBER_WHITELIST
             (the percentile die "100"). On the 🎲 line only the roll value(s) of the records (and the SAN lost of a
             sanity roll) are allowed: keeper-v2 FORMAT says "no extra numbers". Numbers written as words are not
             checked. A rules question without an oracle passage is deliberately strict too: the model is trained
             NOT to state rule numbers that the prompt does not give.
  dice-line  the 🎲 line is present exactly once if and only if the example has rolls (keeper-v2 FORMAT rule).
  outcome    HARD RULE 1: the 🎲 line must state the rolled outcome in words, in agreement with every record
             (`passed`, `level`): a success word («успіх», «вдалося», "success", "you succeed") for a passed roll and
             a failure word («провал», «невдача», «не вдалося», "failure", "the roll fails") for a failed one, and not
             the opposite. Negations are read as what they mean: «без успіху», «успіху немає», «майже успіх», "no
             success", "unsuccessful" are failures; «провалу не сталося», «уникнув провалу», "not a failure" are not.
             A LEVEL named right before «успіх» / "success" («звичайний», «складний», «екстремальний», «критичний
             успіх»; "regular / hard / extreme / critical success") must lie between the required level of the check
             and the rolled level (understating down to the required level is fine, overstating is not); level
             `critical` needs «критичний успіх» / "critical success", level `extreme` on a check that was not
             extreme needs «екстремальний успіх» / "extreme success", level `fumble` needs «фатальний провал» /
             "fumble" (and it may not be said otherwise). For a FAILED roll a level at or below the rolled one is a
             true explanation, not a claim of success («звичайний успіх, але для складного кидка замало — провал»).
             Difficulty phrases («складна перевірка», "hard check") are not levels. With both a passed and a failed
             roll (SAN roll plus the INT check) both kinds of word must be present. A SAN loss stated on the line
             (the number right after «втрата» / "lost" / "loss") must equal the record's `san_loss`.
  citation   [R#] markers: all must exist in the prompt (keeper.citations.validate) and only RAG examples may carry
             them. `distractor_only` / `none`: no citations. `oracle`: no distractor citations, and by kind
             (keeper-v2 rule 7: markers only out of character, i.e. on the 🎲 line or in a rules / meta answer):
               rules_question  at least one oracle passage cited, anywhere
               meta            citations allowed, not required
               in_fiction+roll at least one oracle passage cited, and only on the 🎲 line
               in_fiction, no roll  no markers at all (there is no out-of-character place for them)
  citation-placement   a marker inside the in-character narration (or any marker in an in_fiction reply without a roll).
  ruling     "Keeper ruling" / «рішення Хранителя» (nominative, exactly, with a capital «Хранителя», so a scene's
             «рішення хранителя архіву» is not the phrase; KEEPER_RULING_PHRASE, a strict subset of the
             evaluation regex `evals.harness.KEEPER_RULING`) is REQUIRED iff use_rag and kind is rules_question and kb
             mode is distractor_only or none (keeper-v2 rule 6: «If the KNOWLEDGE BASE does not cover a rules
             question ... say "Keeper ruling" explicitly»), and FORBIDDEN otherwise (oracle: the KB answers;
             in_fiction / meta narration; use_rag off, where rule 6 is not even rendered). The forbidden check
             also catches inflected forms («за рішенням Хранителя»).
  language   uk: Cyrillic share of the letters >= MIN_CYRILLIC_SHARE; en: <= MAX_CYRILLIC_SHARE_EN.
  length     words of the narration (reply without the 🎲 line and markers) within WORD_RANGE[kind]; keeper-v2 asks
             for 80-200 words of narration, the range leaves room for the shorter or longer rules and meta replies.
             The manifest records the distribution (`narration_words`).
  tokens     the whole chat (system, history, reply) estimated with keeper.nodes.CHARS_PER_TOKEN[language] is at most
             MAX_EST_TOKENS. The estimate is rough (training --max-len is 8192 with the exact tokenizer, and the
             Colab length gate is the final check); the manifest records max / p95.
  decontam:* sft.decontam on the user turns, history, target and sections (n-gram, jaccard, subset, forbidden, ...).

The val split is by GROUP: the group key is the example id without a trailing `-uk` / `-en`, so the two language
versions of one scene (`<base>-uk`, `<base>-en`) always land on the same side. Per `kind`, groups are ordered by
sha256(VAL_SEED, group key) and taken until round(VAL_FRACTION * n) examples are in val (at least one for a kind with
3+ examples); the file order does not matter.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError

from evals.datasets import CORPUS_DIR, DATASET_DIR, EvalDatasets
from evals.harness import CYRILLIC, LATIN
from keeper import citations
from keeper.game.coc7e.dice import REQUIRED, Level
from keeper.nodes import CHARS_PER_TOKEN
from keeper.prompts import PromptSet
from keeper.rag.corpus import load_corpus
from server.settings import REPO_ROOT
from sft import gates
from sft.build import PROMPT_VERSION, SFT_DIR, BuildError, sha16, write_text
from sft.decontam import Decontaminator, Example
from sft.schemas import SftInput, SftReview, SftTarget
from sft.split import CHUNK_SIZE, OVERLAP, SPLIT_PATH, chunk_ids_hash, load_split

MANIFEST_SCHEMA = 1
VAL_FRACTION = 0.1
VAL_SEED = 1927
REVIEW_SEED = 1927_10
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "sft"

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
NUMBER_WHITELIST = frozenset({"100"})             # the percentile die; "7" (edition) is not needed
DICE = "🎲"
# "Хранителя" is case-sensitive (the title of the Keeper), so a scene's «рішення хранителя архіву» is not the phrase.
KEEPER_RULING_PHRASE = re.compile(r"(?i:keeper ruling)|(?i:рішення)\s+Хранителя")      # the required form (nominative)
KEEPER_RULING_ANY = re.compile(r"(?i:keeper ruling)|(?i:рішенн\w*)\s+Хранителя")   # any form: the forbidden check
SYSTEM_BODY_MARKER = "SCENE OPENING"              # the keeper-v2 system prompt body starts here (after the rules)
KB_MARKER = "KNOWLEDGE BASE (retrieved for this turn):"
PASSAGE_START = re.compile(r"^\[R(\d+)\] \(", re.M)
MIN_CYRILLIC_SHARE = 0.7                          # uk; Latin remains for «Пильність (Spot Hidden)», "Keeper ruling"
MAX_CYRILLIC_SHARE_EN = 0.05
WORD_RANGE = {"in_fiction": (70, 220), "rules_question": (25, 240), "meta": (12, 160)}
MAX_EST_TOKENS = 7000
MESSAGE_OVERHEAD_TOKENS = 4                       # role tags per message in the chat template
LANG_SUFFIX = re.compile(r"-(?:uk|en)$")          # translation pairs share the base id

# Outcome vocabulary of the 🎲 line (uk + en). Failure phrases include the negations of success («без успіху»,
# «успіху немає», «не вдалося», "no success", "unsuccessful"); they are removed before the line is searched for a
# success word, and the negations of failure («провалу не сталося», «уникнув провалу», "not a failure") are removed
# before it is searched for a failure word.
FAILURE_WORD = re.compile(
    r"провал\w*|невдач\w*|невдал\w*|не\s+вда[вл]\w*|без\s+успіх\w*|успіх\w*\s+(?:немає|не\s+\w+)|майже\s+успіх\w*|"
    r"no\s+success|not\s+(?:a\s+)?success\w*|unsuccessful|failure|failed|\bfails?\b|fumble|фатальн\w*", re.I)
NEGATED_FAILURE = re.compile(
    r"(?:not\s+(?:a\s+)?|no\s+|without\s+)(?:failure|fail)\b|(?:не|без)\s+провал\w*|провал\w*\s+не\s|"
    r"провал\w*\s+(?:\w+\s+){0,2}?уникну\w*|уникну\w*\s+провал\w*", re.I)
# Explicit forms of «вдатися», so «вдалині» / «вдавлений» are not read as success.
SUCCESS_WORD = re.compile(
    r"успіх|успішн|\b(?:у?вда|уда)(?:лос[яь]|лий|ла|ло|лис[яь]|вс[яь]|єтьс[яь])\b|success|succeed\w*", re.I)
FUMBLE_WORD = re.compile(r"фатальн|fumble|критичн\w*\s+провал|critical\s+fail", re.I)
# A level of success named right before «успіх» / "success"; a difficulty phrase («складна перевірка», "hard check")
# is not a level of success and does not match.
LEVEL_PHRASE = re.compile(
    r"(?P<uk>звичайн|регулярн|складн|важк|екстремальн|надзвичайн|критичн)\w*\s+успіх|"
    r"(?P<en>regular|hard|extreme|critical)\s+success", re.I)
LEVEL_OF_WORD = {"звичайн": Level.REGULAR, "регулярн": Level.REGULAR, "regular": Level.REGULAR,
                 "складн": Level.HARD, "важк": Level.HARD, "hard": Level.HARD,
                 "екстремальн": Level.EXTREME, "надзвичайн": Level.EXTREME, "extreme": Level.EXTREME,
                 "критичн": Level.CRITICAL, "critical": Level.CRITICAL}
# The SAN lost: the number must follow «втрат…» / "lost" / "loss" (optionally the word for Sanity) directly, with
# nothing but punctuation and spaces in between, or precede «пунктів» / "points".
SAN_LOSS_NUMBER = re.compile(
    r"(?:втрат\w*|lost|loss)(?:\s+(?:розсудливост\w*|sanity|san)\b)?[^\w]{0,6}(\d+)|(\d+)\s*(?:пункт\w*|points?)", re.I)
HASH_PREFIX = "h:"                                # a letter prefix: Excel turns a bare hex string into a number

KINDS = ("in_fiction", "rules_question", "meta")


# ------------------------------------------------------------------------------------------ lints


@dataclass(frozen=True)
class Problem:
    code: str
    reason: str


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in NUMBER.findall(citations.BRACKET.sub(" ", text))}


def split_system(system: str) -> tuple[str, dict[int, str]]:
    """(the system prompt before the knowledge base, {marker: passage text with its [R#] header})."""
    cut = system.find(KB_MARKER)
    if cut < 0:
        return system, {}
    kb = system[cut + len(KB_MARKER):]
    starts = list(PASSAGE_START.finditer(kb))
    passages = {int(m.group(1)): kb[m.start(): starts[i + 1].start() if i + 1 < len(starts) else len(kb)]
                for i, m in enumerate(starts)}
    return system[:cut], passages


def allowed_numbers(inp: SftInput) -> set[str]:
    """Numbers the narration may use (see the module docstring); distractor passages are not a source."""
    head, passages = split_system(inp.messages[0].content)
    cut = head.find(SYSTEM_BODY_MARKER)
    body = head if cut < 0 else head.split("\n", 1)[0] + "\n" + head[cut:]
    roles = roles_of(inp)
    oracle = "\n".join(t for m, t in passages.items() if roles.get(m) == "oracle")
    allowed = _numbers(body) | _numbers(oracle) | _numbers("\n".join(m.content for m in inp.messages[1:]))
    allowed |= {str(int(y) // 10 * 10) for y in YEAR.findall(body)}          # "1920-ті" for a 1927 scene
    return allowed | set(NUMBER_WHITELIST)


def dice_numbers(inp: SftInput) -> set[str]:
    """The only numbers the 🎲 line may carry: the d100 value(s) and the SAN lost."""
    out = {str(r["roll"]) for r in inp.rolls if "roll" in r}
    return out | {str(r["san_loss"]) for r in inp.rolls if r.get("san_loss") is not None}


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def narration(text: str) -> str:
    """The reply without the 🎲 line(s) and citation markers."""
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith(DICE)]
    return citations.BRACKET.sub("", "\n".join(lines))


def narration_words(text: str) -> int:
    return _words(narration(text))


def estimate_tokens(inp: SftInput, text: str) -> int:
    cpt = CHARS_PER_TOKEN.get(inp.language, 3.0)
    chars = sum(len(m.content) for m in inp.messages) + len(text)
    return math.ceil(chars / cpt) + MESSAGE_OVERHEAD_TOKENS * (len(inp.messages) + 1)


def chat_rows(inp: SftInput, text: str) -> dict:
    return {"messages": [m.model_dump() for m in inp.messages] + [{"role": "assistant", "content": text}]}


def lint_input(inp: SftInput, prompt_hash: str, corpus_hash: str) -> list[Problem]:
    out = []
    if sha16(json.dumps([m.model_dump() for m in inp.messages], ensure_ascii=False)) != inp.input_hash:
        out.append(Problem("input", "input_hash does not match the messages (the inputs file was edited by hand)"))
    r = inp.render
    if r.prompt_version != PROMPT_VERSION or r.prompt_hash != prompt_hash:
        out.append(Problem("input", f"rendered with prompt {r.prompt_version} {r.prompt_hash}, current is "
                                    f"{PROMPT_VERSION} {prompt_hash}: rebuild the inputs and re-write the target"))
    if r.corpus_chunk_ids_hash != corpus_hash:
        out.append(Problem("input", "rendered from another corpus than the current one: rebuild the inputs"))
    return out


def roles_of(inp: SftInput) -> dict[int, str]:
    return {h.marker: h.role for h in inp.hits}


def _markers(ids: Iterable[int]) -> str:
    return "[R" + ", R".join(map(str, ids)) + "]"


def lint_numbers(inp: SftInput, text: str) -> list[Problem]:
    dice_lines = [ln for ln in text.splitlines() if ln.lstrip().startswith(DICE)]
    rest = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith(DICE))
    out = []
    extra = sorted(_numbers(rest) - allowed_numbers(inp), key=lambda n: (len(n), n))
    if extra:
        out.append(Problem("number", "numbers not found in the sheet, MECHANICS, oracle passages or player text: "
                                     f"{', '.join(extra)}"))
    line_extra = sorted(_numbers("\n".join(dice_lines)) - dice_numbers(inp), key=lambda n: (len(n), n))
    if line_extra:
        out.append(Problem("number", f"the {DICE} line carries numbers other than the roll value(s) "
                                     f"{sorted(dice_numbers(inp))}: {', '.join(line_extra)} "
                                     "(FORMAT: no extra numbers)"))
    return out


def lint_dice_line(inp: SftInput, text: str) -> list[Problem]:
    dice_lines = [ln for ln in text.splitlines() if ln.lstrip().startswith(DICE)]
    if inp.rolls and not (len(dice_lines) == 1 and text.count(DICE) == 1):
        return [Problem("dice-line", f"the example has {len(inp.rolls)} roll(s): exactly one line starting with "
                                     f"{DICE} is required (found {text.count(DICE)} {DICE}, "
                                     f"{len(dice_lines)} such line(s))")]
    if not inp.rolls and DICE in text:
        return [Problem("dice-line", f"{DICE} used but there is no roll in MECHANICS")]
    return []


def _level(record: dict) -> Level:
    return Level[str(record.get("level", "failure")).upper()]


def _required(record: dict) -> Level:
    return REQUIRED.get(record.get("difficulty", "regular"), Level.REGULAR)


def level_phrases(text: str) -> list[tuple[Level, re.Match]]:
    return [(LEVEL_OF_WORD[(m.group("uk") or m.group("en")).lower()], m) for m in LEVEL_PHRASE.finditer(text)]


def lint_outcome(inp: SftInput, text: str) -> list[Problem]:
    """The 🎲 line says what the dice said (HARD RULE 1), including the level of success it names."""
    dice_lines = [ln for ln in text.splitlines() if ln.lstrip().startswith(DICE)]
    if not inp.rolls or len(dice_lines) != 1:
        return []                                   # a missing / doubled line is reported by dice-line
    line = dice_lines[0]
    records = inp.rolls
    passed = [r for r in records if r.get("passed")]
    failed = [r for r in records if not r.get("passed")]
    facts = ", ".join(f"{r.get('target')} roll {r.get('roll')} -> {r.get('level')} (difficulty "
                      f"{r.get('difficulty', 'regular')}, {'passed' if r.get('passed') else 'failed'})"
                      for r in records)
    head = f"MECHANICS: {facts}; the {DICE} line"
    said_failure = bool(FAILURE_WORD.search(NEGATED_FAILURE.sub(" ", line)))
    base = FAILURE_WORD.sub(" ", line)              # «не вдалося», «без успіху», "unsuccessful" are not successes
    phrases = level_phrases(base)
    if not passed:
        # a level named at or below the rolled level explains the failure («звичайний успіх, але для складного
        # кидка замало — провал»); it is not a claim of success
        top = max(_level(r) for r in failed)
        spans = [m.span() for lvl, m in phrases if lvl <= top]
        base = "".join(ch if not any(a <= k < b for a, b in spans) else " " for k, ch in enumerate(base))
    said_success = bool(SUCCESS_WORD.search(base))
    out = []
    if passed and not said_success:
        out.append(Problem("outcome", f"{head} does not state a success"))
    if failed and not said_failure:
        out.append(Problem("outcome", f"{head} does not state a failure"))
    if not passed and said_success:
        out.append(Problem("outcome", f"{head} says success"))
    if not failed and said_failure:
        out.append(Problem("outcome", f"{head} says failure"))
    if passed:
        for lvl, m in phrases:
            fits = any(_required(r) <= lvl <= _level(r) for r in passed) or any(lvl <= _level(r) for r in failed)
            if not fits:
                out.append(Problem("outcome", f"{head} names «{m.group(0)}» ({lvl.name.lower()}), which the roll does "
                                              "not support: a named level must be between the required level of the "
                                              "check and the rolled level"))
        named = {lvl for lvl, _ in phrases}
        if any(_level(r) == Level.CRITICAL for r in passed) and Level.CRITICAL not in named:
            out.append(Problem("outcome", f"{head} must name the critical success («критичний успіх»)"))
        if any(_level(r) == Level.EXTREME and _required(r) != Level.EXTREME for r in passed) \
                and Level.EXTREME not in named:
            out.append(Problem("outcome", f"{head} must name the extreme success («екстремальний успіх»): the roll "
                                          "beat the required level of the check"))
    levels = {r.get("level") for r in records}
    said_fumble = bool(FUMBLE_WORD.search(line))
    if "fumble" in levels and not said_fumble:
        out.append(Problem("outcome", f"{head} must name the fumble («фатальний провал»)"))
    if "fumble" not in levels and said_fumble:
        out.append(Problem("outcome", f"{head} claims a fumble («фатальний провал») that was not rolled"))
    losses = {int(r["san_loss"]) for r in records if r.get("san_loss") is not None}
    for match in SAN_LOSS_NUMBER.finditer(line):
        stated = int(match.group(1) or match.group(2))
        if losses and stated not in losses:
            out.append(Problem("outcome", f"the {DICE} line states a SAN loss of {stated}, MECHANICS says "
                                          f"{sorted(losses)}"))
    return out


def lint_citations(inp: SftInput, text: str) -> list[Problem]:
    out = []
    cited = citations.cited_markers(text)
    roles = roles_of(inp)
    if cited and not inp.use_rag:
        return [Problem("citation", f"cites {_markers(cited)} but the example has no knowledge base (use_rag is off)")]
    invalid = [m for m in cited if m not in roles]
    if invalid:
        out.append(Problem("citation", f"markers {['R' + str(m) for m in invalid]} are not in the prompt "
                                       f"(it has {['R' + str(h.marker) for h in inp.hits] or 'no passages'})"))
    elif cited and inp.kb_mode != "oracle":
        out.append(Problem("citation", f"cites {_markers(cited)} but kb mode {inp.kb_mode!r} has no passage that "
                                       "answers the question"))
    if not (inp.use_rag and inp.kb_mode == "oracle"):
        return out
    distractors = [m for m in cited if roles.get(m) == "distractor"]
    if distractors:
        out.append(Problem("citation", f"cites distractor passage(s) {_markers(distractors)}"))
    lines = text.splitlines()
    in_narration = citations.cited_markers("\n".join(ln for ln in lines if not ln.lstrip().startswith(DICE)))
    on_dice_line = citations.cited_markers("\n".join(ln for ln in lines if ln.lstrip().startswith(DICE)))
    oracle_cited = [m for m in cited if roles.get(m) == "oracle"]
    if inp.kind == "rules_question" and not oracle_cited:
        out.append(Problem("citation", "kb mode 'oracle' in a rules question: no oracle passage is cited as [R#]"))
    if inp.kind == "in_fiction":
        if in_narration:
            where = ("inside the in-character narration" if inp.rolls else
                     "in an in_fiction reply without a roll, which has no out-of-character place for them")
            out.append(Problem("citation-placement", f"{_markers(in_narration)} {where}: keeper-v2 rule 7 allows "
                                                     f"markers only on the {DICE} line or in a rules / meta answer"))
        if inp.rolls and not any(roles.get(m) == "oracle" for m in on_dice_line):
            out.append(Problem("citation", f"kb mode 'oracle' with a roll: no oracle passage is cited as [R#] on the "
                                           f"{DICE} line"))
    return out


def lint_ruling(inp: SftInput, text: str) -> list[Problem]:
    required = inp.use_rag and inp.kind == "rules_question" and inp.kb_mode != "oracle"
    if required and not KEEPER_RULING_PHRASE.search(text):
        return [Problem("ruling", f"kb mode {inp.kb_mode!r} in a rules question: the reply must say \"Keeper ruling\" "
                                  "(or «рішення Хранителя», nominative, exactly) explicitly")]
    if not required and KEEPER_RULING_ANY.search(text):
        why = ("use_rag is off, so keeper-v2 rule 6 is not in the prompt" if not inp.use_rag else
               "the knowledge base answers: cite it instead" if inp.kb_mode == "oracle" else
               f"it is only for rules questions the knowledge base does not cover (this is a {inp.kind} turn)")
        return [Problem("ruling", f"\"Keeper ruling\" / «рішення Хранителя» must not appear: {why}")]
    return []


def lint_text(inp: SftInput, text: str) -> list[Problem]:
    """All lints that need only the input and the final reply (no decontamination)."""
    out = lint_numbers(inp, text) + lint_dice_line(inp, text) + lint_outcome(inp, text)
    out += lint_citations(inp, text) + lint_ruling(inp, text)
    # language
    cyr, lat = len(CYRILLIC.findall(narration(text))), len(LATIN.findall(narration(text)))
    share = cyr / (cyr + lat) if cyr + lat else 0.0
    if inp.language == "uk" and share < MIN_CYRILLIC_SHARE:
        out.append(Problem("language", f"Ukrainian reply has a Cyrillic share of {share:.2f} < {MIN_CYRILLIC_SHARE}"))
    if inp.language == "en" and (share > MAX_CYRILLIC_SHARE_EN or not lat):
        out.append(Problem("language", f"English reply has a Cyrillic share of {share:.2f} (max "
                                       f"{MAX_CYRILLIC_SHARE_EN}) and {lat} Latin letters"))
    # narration length
    low, high = WORD_RANGE[inp.kind]
    words = narration_words(text)
    if not low <= words <= high:
        out.append(Problem("length", f"{words} words outside {low}-{high} for kind {inp.kind!r}"))
    # chat length
    est = estimate_tokens(inp, text)
    if est > MAX_EST_TOKENS:
        out.append(Problem("tokens", f"estimated {est} tokens > {MAX_EST_TOKENS} "
                                     f"(chars/{CHARS_PER_TOKEN[inp.language]} for {inp.language}); "
                                     "shorten the reply or the context"))
    # the training gate on the finished chat
    try:
        gates.check_messages([chat_rows(inp, text)], inp.id)
    except gates.GateError as e:
        out.append(Problem("gate", str(e)))
    return out


def decontam_example(inp: SftInput, text: str) -> Example:
    return Example(
        id=inp.id, user_turns=tuple([t.text for t in inp.history if t.role == "player"] + [inp.player_text]),
        assistant_turns=tuple(t.text for t in inp.history if t.role == "keeper"), target=text,
        oracle_sections=tuple(r.key() for r in inp.oracle_sections),
        context_sections=tuple(r.key() for r in inp.context_sections), scenario_id=inp.scenario_id)


# ------------------------------------------------------------------------------------------ context


@dataclass
class DatasetContext:
    decontam: Decontaminator
    prompt_hash: str
    corpus_hash: str
    split_seed: int
    split_file: str
    split_sha256: str
    gold_hashes: dict[str, str]


def file_sha256(path: Path) -> str:
    """sha256 of a text file with LF endings, so a Windows checkout hashes like a Linux one."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_dataset_context() -> DatasetContext:
    ds = EvalDatasets.load(DATASET_DIR)
    chunks = load_corpus([CORPUS_DIR], chunk_size=CHUNK_SIZE, overlap=OVERLAP, root=REPO_ROOT)
    split = load_split()
    return DatasetContext(
        decontam=Decontaminator.from_datasets(ds, split, chunks=chunks), prompt_hash=PromptSet().hash(PROMPT_VERSION),
        corpus_hash=chunk_ids_hash(chunks), split_seed=split.seed,
        split_file=SPLIT_PATH.relative_to(REPO_ROOT).as_posix(), split_sha256=file_sha256(SPLIT_PATH),
        gold_hashes=dict(ds.hashes))


def git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
                             check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes (the build is then not reproducible from the sha alone)."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=10, check=True)
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


# ------------------------------------------------------------------------------------------ reading


def load_models[T: BaseModel](path: Path, model: type[T], required: bool = True) -> list[T]:
    if not path.is_file():
        if required:
            raise BuildError([f"file not found: {path}"])
        return []
    rows, problems = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate(json.loads(line)))
        except json.JSONDecodeError as e:
            problems.append(f"{path.name}:{n}: invalid JSON ({e})")
        except ValidationError as e:
            problems.append(f"{path.name}:{n}: " + "; ".join(
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()))
    if problems:
        raise BuildError(problems)
    return rows


@dataclass
class Row:
    inp: SftInput
    text: str
    origin: str          # "target" | "edited"
    lint_codes: list[str] = field(default_factory=list)
    target_text: str = ""  # the target as written (the text a review's target_hash refers to)
    verdict: str = ""      # the verdict that applies to THIS target text ("" = none), see assemble
    rewritten: bool = False  # verdicts exist, but only for earlier versions of the target text


@dataclass
class Assembled:
    rows: list[Row] = field(default_factory=list)         # sorted by id
    problems: list[str] = field(default_factory=list)     # "<id>: [code] reason"
    missing: list[str] = field(default_factory=list)      # inputs without a target
    rejected: list[str] = field(default_factory=list)
    edited: list[str] = field(default_factory=list)
    unreviewed: int = 0
    targets_without_input_hash: int = 0
    reviews_without_target_hash: int = 0
    rewritten: list[str] = field(default_factory=list)        # reviewed once, target rewritten since: unreviewed again
    rejected_rows: list[Row] = field(default_factory=list)    # dropped by a verdict that applies to the current text


def _dupes(ids: Iterable[str]) -> list[str]:
    return sorted(i for i, n in Counter(ids).items() if n > 1)


def assemble(inputs: list[SftInput], targets: list[SftTarget], reviews: list[SftReview],
             ctx: DatasetContext, allow_missing: bool = False, allow_no_input_hash: bool = False) -> Assembled:
    out = Assembled()
    problems: list[str] = []
    for name, ids in (("inputs", [i.id for i in inputs]), ("targets", [t.id for t in targets])):
        problems += [f"{i}: duplicate id in {name}" for i in _dupes(ids)]
    # reviews are a history: one verdict per (id, target text hash); only the one for the CURRENT text applies
    problems += [f"{i}: duplicate review for the same target text" for i in
                 sorted({k[0] for k, n in Counter((r.id, r.target_hash) for r in reviews).items() if n > 1})]
    by_input = {i.id: i for i in inputs}
    by_target = {t.id: t for t in targets}
    history: dict[str, list[SftReview]] = {}
    for r in reviews:
        history.setdefault(r.id, []).append(r)
    problems += [f"{t}: target for an unknown input id" for t in sorted(by_target.keys() - by_input.keys())]
    problems += [f"{r}: review for an id without a target" for r in sorted(history.keys() - by_target.keys())]
    out.missing = sorted(by_input.keys() - by_target.keys())
    if out.missing and not allow_missing:
        problems += [f"{i}: no target" for i in out.missing]

    kept: list[Row] = []
    for inp in inputs:
        target = by_target.get(inp.id)
        if target is None:
            continue
        past = history.get(inp.id, [])
        review = next((r for r in past if r.target_hash == sha16(target.text)), None)
        if review is None:                                  # hand-made reviews carry no hash and always apply
            review = next((r for r in past if r.target_hash is None), None)
            out.reviews_without_target_hash += review is not None
        rewritten = review is None and bool(past)           # only earlier versions of the text were reviewed
        if rewritten:
            out.rewritten.append(inp.id)
        if review is not None and review.verdict == "reject":
            out.rejected.append(inp.id)
            out.rejected_rows.append(Row(inp, target.text, "target", [], target.text, "reject"))
            continue
        found = [(p.code, p.reason) for p in lint_input(inp, ctx.prompt_hash, ctx.corpus_hash)]
        if target.input_hash is None:
            out.targets_without_input_hash += 1
            if not allow_no_input_hash:
                found.append(("no-input-hash", "the target has no input_hash, so it cannot be checked against the "
                                               "frozen input (add it, or pass --allow-no-input-hash)"))
        elif target.input_hash != inp.input_hash:
            found.append(("stale", f"target was written against input {target.input_hash}, the input is now "
                                   f"{inp.input_hash}: re-write the target"))
        if review is not None and review.verdict == "edit":
            text, origin = unicodedata.normalize("NFC", review.edited_text or ""), "edited"
            out.edited.append(inp.id)
        else:
            text, origin = target.text, "target"
            out.unreviewed += review is None
        found += [(p.code, p.reason) for p in lint_text(inp, text)]
        found += [(f"decontam:{v.check}", f"{v.field}: {v.reason}")
                  for v in ctx.decontam.check(decontam_example(inp, text))]
        problems += [f"{inp.id}: [{code}] {reason}" for code, reason in found]
        kept.append(Row(inp, text, origin, [code for code, _ in found], target.text,
                        review.verdict if review is not None else "", rewritten))
    out.rows = sorted(kept, key=lambda r: r.inp.id)
    out.problems = problems
    return out


# ------------------------------------------------------------------------------------------ split and write


def group_key(example_id: str) -> str:
    """Translation pairs (`<base>-uk`, `<base>-en`) share a group and so land on the same side of the split."""
    return LANG_SUFFIX.sub("", example_id)


def _order_key(seed: int, tag: str, example_id: str) -> str:
    return hashlib.sha256(f"{seed}:{tag}:{example_id}".encode()).hexdigest()


def split_rows(rows: list[Row], fraction: float = VAL_FRACTION, seed: int = VAL_SEED) -> tuple[list[Row], list[Row]]:
    """(train, val): per kind, groups (ids without `-uk` / `-en`) by hash until round(fraction * n) are in val."""
    groups: dict[str, list[Row]] = {}
    for r in sorted(rows, key=lambda r: r.inp.id):
        groups.setdefault(group_key(r.inp.id), []).append(r)
    by_kind: dict[str, list[str]] = {}
    for key, members in groups.items():
        by_kind.setdefault(members[0].inp.kind, []).append(key)
    val_ids: set[str] = set()
    for kind in sorted(by_kind):
        n_kind = sum(len(groups[k]) for k in by_kind[kind])
        n_val = round(fraction * n_kind)
        if n_kind >= 3:
            n_val = max(1, n_val)
        taken = 0
        for key in sorted(by_kind[kind], key=lambda k: _order_key(seed, "val", k)):
            if taken >= n_val:
                break
            val_ids |= {r.inp.id for r in groups[key]}
            taken += len(groups[key])
    ordered = sorted(rows, key=lambda r: r.inp.id)
    return [r for r in ordered if r.inp.id not in val_ids], [r for r in ordered if r.inp.id in val_ids]


def dump_rows(rows: list[Row]) -> str:
    return "".join(json.dumps(chat_rows(r.inp, r.text), ensure_ascii=False) + "\n" for r in rows)


def _counts(rows: list[Row]) -> dict:
    def tally(key) -> dict:
        return dict(sorted(Counter(key(r.inp) for r in rows).items()))

    return {"examples": len(rows), "kind": tally(lambda i: i.kind),
            "use_rag": tally(lambda i: "rag" if i.use_rag else "no_rag"),
            "language": tally(lambda i: i.language), "kb_mode": tally(lambda i: i.kb_mode),
            "with_rolls": sum(1 for r in rows if r.inp.rolls)}


def _source(path: Path, root: Path = REPO_ROOT) -> dict | None:
    if not path.is_file():
        return None
    try:
        shown = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        shown = path.name
    return {"file": shown, "sha256": file_sha256(path)}


def make_manifest(version: str, train: list[Row], val: list[Row], texts: dict[str, str], ctx: DatasetContext,
                  assembled: Assembled, sources: dict[str, Path], sha: str | None,
                  dirty: bool | None = None) -> dict:
    rows = train + val
    estimates = [estimate_tokens(r.inp, r.text) for r in rows]
    words = {"all": gates.length_stats([narration_words(r.text) for r in rows]),
             **{k: gates.length_stats(w) for k in KINDS if (w := [narration_words(r.text) for r in rows
                                                                   if r.inp.kind == k])}}
    return {
        "schema": MANIFEST_SCHEMA, "dataset": f"coc7e-{version}", "version": version, "git_sha": sha,
        "git_dirty": dirty,
        "partial": bool(assembled.missing),
        "missing_targets": {"count": len(assembled.missing), "ids": assembled.missing},
        "files": {name: {"sha256": gates.sha256_text(text), "examples": len(part)}
                  for (name, text), part in zip(texts.items(), (train, val), strict=True)},
        "prompt": {"version": PROMPT_VERSION, "hash": ctx.prompt_hash},
        "split": {"file": ctx.split_file, "sha256": ctx.split_sha256, "seed": ctx.split_seed,
                  "corpus_chunk_ids_hash": ctx.corpus_hash},
        "gold_hashes": dict(sorted(ctx.gold_hashes.items())),
        "sources": {name: _source(path) for name, path in sources.items()},
        "val_split": {"fraction": VAL_FRACTION, "seed": VAL_SEED,
                      "method": "per-kind sha256(seed, group key) order, group key = id without -uk/-en"},
        "counts": {"all": _counts(rows), "train": _counts(train), "val": _counts(val)},
        "ids": {"train": [r.inp.id for r in train], "val": [r.inp.id for r in val]},
        "reviews": {"edited": len(assembled.edited), "rejected": len(assembled.rejected),
                    "rejected_ids": assembled.rejected, "unreviewed": assembled.unreviewed,
                    "without_target_hash": assembled.reviews_without_target_hash,
                    "rewritten": {"count": len(assembled.rewritten), "ids": assembled.rewritten}},
        "targets_without_input_hash": assembled.targets_without_input_hash,
        "narration_words": words,
        "token_estimate": {**gates.length_stats(estimates), "limit": MAX_EST_TOKENS,
                           "chars_per_token": CHARS_PER_TOKEN},
        "lints": {"number_whitelist": sorted(NUMBER_WHITELIST), "min_cyrillic_share": MIN_CYRILLIC_SHARE,
                  "word_range": {k: list(v) for k, v in WORD_RANGE.items()}},
    }


def write_dataset(version: str, assembled: Assembled, ctx: DatasetContext, out_dir: Path,
                  sources: dict[str, Path], sha: str | None, dirty: bool | None = None) -> dict:
    train, val = split_rows(assembled.rows)
    if not train or not val:
        raise BuildError([f"the split left {len(train)} train and {len(val)} val examples: both must be non-empty "
                          f"(have {len(assembled.rows)} examples)"])
    texts = {"train.jsonl": dump_rows(train), "val.jsonl": dump_rows(val)}
    manifest = make_manifest(version, train, val, texts, ctx, assembled, sources, sha, dirty)
    for name, text in texts.items():
        write_text(out_dir / name, text)
    gates.dump_json(out_dir / "manifest.json", manifest)
    gates.verify_data_hashes(manifest, {name: out_dir / name for name in texts})   # what training will do
    for name in texts:
        gates.check_messages(gates.load_jsonl(out_dir / name), name)
    return manifest


# ------------------------------------------------------------------------------------------ review CSV

CSV_COLUMNS = ["id", "kind", "kb_mode", "language", "use_rag", "scene", "player_text", "rolls", "passages",
               "passages_text", "lint", "target", "target_hash", "status", "current_verdict", "current_edit",
               "verdict", "edited_text", "note"]
CSV_DELIMITER = ";"      # Excel in a Ukrainian / European locale opens `;`-separated files into columns


def rolls_summary(rolls: list[dict]) -> str:
    parts = []
    for r in rolls:
        bits = [str(r.get("kind", "")), str(r.get("target") or "")]
        for key in ("value", "difficulty", "roll", "level", "san_loss", "san_after"):
            if r.get(key) is not None:
                bits.append(f"{key}={r[key]}")
        parts.append(" ".join(b for b in bits if b))
    return " | ".join(parts)


def passages_summary(inp: SftInput) -> str:
    return " | ".join(f"[R{h.marker}] {h.role}: {h.source} > {h.section}" for h in inp.hits)


def passages_text(inp: SftInput) -> str:
    """Full text of every KB passage as rendered into the prompt, with its role, so the reviewer can check citations."""
    if not inp.hits:
        return ""
    _, rendered = split_system(inp.messages[0].content)
    return "\n\n".join(f"{h.role.upper()} {rendered.get(h.marker, f'[R{h.marker}] (not found in the prompt)').strip()}"
                       for h in inp.hits)


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """How many of `n` review slots each stratum gets: proportional, at least one each, never above its size."""
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)
    if len(sizes) > n:   # fewer slots than strata: the largest strata, one each
        keep = sorted(sizes, key=lambda k: (-sizes[k], k))[:n]
        return {k: (1 if k in keep else 0) for k in sizes}
    quota = {k: min(s, max(1, n * s // total)) for k, s in sizes.items()}
    while sum(quota.values()) > n:
        k = max((k for k in quota if quota[k] > 1), key=lambda k: (quota[k], k))
        quota[k] -= 1
    while sum(quota.values()) < n:
        k = max((k for k in quota if quota[k] < sizes[k]), key=lambda k: (n * sizes[k] / total - quota[k], k))
        quota[k] += 1
    return quota


def sample_for_review(rows: list[Row], n: int, seed: int = REVIEW_SEED) -> list[Row]:
    strata: dict[str, list[Row]] = {}
    for r in rows:
        strata.setdefault(f"{r.inp.kind}/{r.inp.kb_mode}", []).append(r)
    quota = allocate({k: len(v) for k, v in strata.items()}, n)
    picked = []
    for key, group in strata.items():
        picked += sorted(group, key=lambda r: _order_key(seed, "review", r.inp.id))[:quota[key]]
    return sorted(picked, key=lambda r: r.inp.id)


def review_status(r: Row) -> str:
    if r.rewritten:
        return "rewritten after review"
    return "reviewed" if r.verdict else "new"


def select_for_review(assembled: Assembled, n: int, *, only_stale: bool = False, include_reviewed: bool = False,
                      seed: int = REVIEW_SEED) -> list[Row]:
    """Rows to show the author: targets rewritten since they were reviewed come first (they have no verdict for their
    current text and would otherwise stay unreviewed), then a stratified sample of the never-reviewed ones. A row that
    already has a verdict for its current text is left out unless `include_reviewed`."""
    stale = [r for r in assembled.rows if r.rewritten]
    if only_stale:
        return stale[:n]
    pool = assembled.rows + (assembled.rejected_rows if include_reviewed else [])
    fresh = [r for r in pool if not r.rewritten and (include_reviewed or not r.verdict)]
    picked = stale[:n]
    picked += sample_for_review(fresh, n - len(picked), seed) if n > len(picked) and fresh else []
    return sorted(picked, key=lambda r: r.inp.id)


def write_review_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: Excel only recognises UTF-8 CSV with a BOM; read_review_csv accepts both
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n", delimiter=CSV_DELIMITER)
        w.writeheader()
        for r in rows:
            i = r.inp
            lint = "(rejected)" if r.verdict == "reject" else "ok" if not r.lint_codes else " ".join(r.lint_codes)
            w.writerow({"id": i.id, "kind": i.kind, "kb_mode": i.kb_mode, "language": i.language,
                        "use_rag": "yes" if i.use_rag else "no",
                        "scene": i.history[-1].text if i.history else "(scenario opening)",
                        "player_text": i.player_text, "rolls": rolls_summary(i.rolls),
                        "passages": passages_summary(i), "passages_text": passages_text(i), "lint": lint,
                        "target": r.target_text,
                        "target_hash": HASH_PREFIX + sha16(r.target_text), "status": review_status(r),
                        "current_verdict": r.verdict, "current_edit": r.text if r.origin == "edited" else "",
                        "verdict": "", "edited_text": "", "note": ""})


def read_review_rows(path: Path) -> list[tuple[int, SftReview]]:
    """(CSV line, review) for every row with a verdict; rows with an empty verdict are not reviewed yet and are skipped.

    Excel adds a BOM, may use `;` as the separator (Ukrainian locale) and CRLF; edited text is NFC-normalised and
    stripped of outer whitespace and CR, since a spreadsheet cell cannot be trusted to keep them. Every verdict row
    keeps its `target_hash` (the hash of the target text the reviewer saw, written with the prefix `h:` so Excel cannot
    read it as a number; the prefix is optional on input), which binds the verdict to that text.
    """
    try:
        raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    except UnicodeDecodeError as e:
        raise BuildError([f"{path.name}: not UTF-8 text ({e.reason} at byte {e.start}); in Excel use Save As -> "
                          "\"CSV UTF-8 (Comma delimited)\""]) from e
    header = raw.split("\n", 1)[0]
    delimiter = ";" if header.count(";") > header.count(",") else ","
    reader = csv.DictReader(io.StringIO(raw, newline=""), delimiter=delimiter)
    missing = {"id", "verdict", "edited_text", "note", "target_hash"} - set(reader.fieldnames or [])
    if missing:
        raise BuildError([f"{path.name}: missing column(s) {sorted(missing)} (found {reader.fieldnames})"])
    out, problems, seen = [], [], set()
    for n, row in enumerate(reader, start=2):
        verdict = (row.get("verdict") or "").strip().lower()
        if not verdict:
            continue
        rid = (row.get("id") or "").strip()
        edited = unicodedata.normalize("NFC", (row.get("edited_text") or "").replace("\r", "")).strip()
        try:
            if rid in seen:
                raise ValueError("duplicate id")
            seen.add(rid)
            if verdict != "edit" and edited:
                raise ValueError(f"verdict {verdict!r} takes no edited_text (clear the cell or use 'edit')")
            target_hash = (row.get("target_hash") or "").strip().removeprefix(HASH_PREFIX)
            if not target_hash:
                raise ValueError("target_hash is empty (it binds the verdict to the reviewed text; do not edit it)")
            review = SftReview(id=rid, verdict=verdict, edited_text=(edited or None) if verdict == "edit" else None,
                               note=(row.get("note") or "").strip(), target_hash=target_hash)
            out.append((n, review))
        except ValueError as e:
            problems.append(f"{path.name}:{n} ({rid or '?'}): {e}")
    if problems:
        raise BuildError(problems)
    return out


def read_review_csv(path: Path) -> list[SftReview]:
    return [r for _, r in read_review_rows(path)]


# ------------------------------------------------------------------------------------------ CLI


def dataset_paths(args) -> dict[str, Path]:
    v = args.version
    return {"inputs": args.inputs or SFT_DIR / f"inputs.{v}.jsonl",
            "targets": args.targets or SFT_DIR / f"targets.{v}.jsonl",
            "reviews": args.reviews or SFT_DIR / f"reviews.{v}.jsonl"}


def _report(lines: list[str]) -> None:
    print(f"{len(lines)} problem(s):")
    for line in lines:
        print(f"  {line}")


def _load_assembled(args, allow_missing: bool) -> tuple[DatasetContext, dict[str, Path], Assembled]:
    paths = dataset_paths(args)
    ctx = load_dataset_context()
    inputs = load_models(paths["inputs"], SftInput)
    targets = load_models(paths["targets"], SftTarget)
    reviews = load_models(paths["reviews"], SftReview, required=False)
    return ctx, paths, assemble(inputs, targets, reviews, ctx, allow_missing=allow_missing,
                                allow_no_input_hash=getattr(args, "allow_no_input_hash", False))


def cmd_build(args) -> int:
    try:
        ctx, paths, assembled = _load_assembled(args, allow_missing=args.allow_missing or args.lint_only)
    except BuildError as e:
        _report(e.problems)
        return 1
    print(f"{len(assembled.rows)} examples with a target, {len(assembled.rejected)} rejected by review, "
          f"{len(assembled.edited)} edited, {len(assembled.missing)} inputs without a target")
    if assembled.missing and args.lint_only:
        print(f"  (no target yet: {', '.join(assembled.missing[:10])}{' ...' if len(assembled.missing) > 10 else ''})")
    if assembled.targets_without_input_hash and getattr(args, "allow_no_input_hash", False):
        print(f"  warning: {assembled.targets_without_input_hash} target(s) have no input_hash, so staleness "
              "cannot be checked")
    if assembled.rewritten:
        print(f"  warning: {len(assembled.rewritten)} target(s) were rewritten after they were reviewed; the old "
              f"verdicts do not apply and they are built as unreviewed (re-review them with `review-export "
              f"--sample N --sample-stale`): {', '.join(assembled.rewritten[:10])}"
              f"{' ...' if len(assembled.rewritten) > 10 else ''}")
    if assembled.reviews_without_target_hash:
        print(f"  warning: {assembled.reviews_without_target_hash} review(s) have no target_hash, so they are "
              "applied whatever the target says now")
    if assembled.problems:
        _report(assembled.problems)
        return 1
    if args.lint_only:
        print("lint OK")
        return 0
    out_dir = args.out_dir or ARTIFACTS_DIR / f"coc7e-{args.version}"
    try:
        manifest = write_dataset(args.version, assembled, ctx, out_dir, paths, git_sha(), git_dirty())
    except BuildError as e:
        _report(e.problems)
        return 1
    est = manifest["token_estimate"]
    print(f"wrote {out_dir}: train {manifest['files']['train.jsonl']['examples']}, val "
          f"{manifest['files']['val.jsonl']['examples']}; estimated tokens max {est['max']}, p95 {est['p95']}")
    return 0


def merge_reviews(path: Path, rows: list[tuple[int, SftReview]], csv_name: str,
                  targets: list[SftTarget]) -> tuple[list[SftReview], int]:
    """The history in `path` plus the CSV verdicts, keyed by (id, target_hash); the verdicts of the CSV win on a key
    they share. A CSV verdict for a text that is no longer the current target is refused with its CSV line."""
    current = {t.id: sha16(t.text) for t in targets}
    problems = []
    for line, r in rows:
        if r.id not in current:
            problems.append(f"{csv_name}:{line} ({r.id}): review for an id without a target")
        elif r.target_hash != current[r.id]:
            problems.append(f"{csv_name}:{line} ({r.id}): target_hash {r.target_hash} is not the current target "
                            f"({current[r.id]}): the target was rewritten after this CSV was exported; "
                            "re-export it and review the new text (or delete the row)")
    if problems:
        raise BuildError(problems)
    merged = {(r.id, r.target_hash): r for r in load_models(path, SftReview, required=False)}
    replaced = sum(1 for _, r in rows if (r.id, r.target_hash) in merged)
    merged.update({(r.id, r.target_hash): r for _, r in rows})
    return [merged[k] for k in sorted(merged, key=lambda k: (k[0], k[1] or ""))], replaced


def cmd_review_export(args) -> int:
    try:
        if args.review:
            rows = read_review_rows(args.review)
            paths = dataset_paths(args)
            out = args.reviews_out or paths["reviews"]
            result, replaced = merge_reviews(out, rows, args.review.name, load_models(paths["targets"], SftTarget))
            write_text(out, "".join(json.dumps(r.model_dump(), ensure_ascii=False) + "\n" for r in result))
            kinds = Counter(r.verdict for r in result)
            print(f"wrote {out}: {len(result)} verdicts {dict(sorted(kinds.items()))} ({len(rows)} from the CSV, "
                  f"{replaced} replaced a verdict on the same text)")
            return 0
        ctx, _, assembled = _load_assembled(args, allow_missing=True)
    except BuildError as e:
        _report(e.problems)
        return 1
    if not assembled.rows:
        print("no examples with a target to sample")
        return 1
    sample = select_for_review(assembled, args.sample, only_stale=getattr(args, "sample_stale", False),
                               include_reviewed=getattr(args, "include_reviewed", False))
    if not sample:
        print("nothing to review: every target already has a verdict for its current text "
              "(--include-reviewed shows them anyway)")
        return 0
    out = args.out or ARTIFACTS_DIR / f"coc7e-{args.version}" / "review-sample.csv"
    write_review_csv(out, sample)
    strata = Counter(f"{r.inp.kind}/{r.inp.kb_mode}" for r in sample)
    stale = sum(1 for r in sample if r.rewritten)
    print(f"wrote {out}: {len(sample)} of {len(assembled.rows)} examples ({stale} rewritten after review) "
          f"{dict(sorted(strata.items()))}")
    if assembled.problems:
        print(f"note: {len(assembled.problems)} lint problem(s) in the full set (run `build --lint-only`)")
    return 0
