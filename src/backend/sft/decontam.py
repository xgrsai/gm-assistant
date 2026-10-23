"""Decontamination of SFT examples against the gold evaluation sets. No LLM, no embeddings: pure string checks.

An example (see `Example`) is checked on simple inputs: the texts of its user turns, prior assistant (Keeper) turns
and target, the corpus sections its answer relies on (`oracle_sections`) or merely shows (`context_sections`,
distractors included) and the scenario it was played in. Checks, each one reported as a `Violation` with its reason:

  ngram          a user turn, assistant turn or the target shares a word n-gram (N_GRAM words, case-folded,
                 apostrophes dropped) with a gold text: `question_uk` / `question_en` of rules_qa and
                 conflict_probes, `scene` and `player_text` of adjudication. Gold texts of MIN_VERBATIM_TOKENS to
                 N_GRAM-1 words must appear verbatim (word-bounded); shorter ones (e.g. «Відскакую вбік!») are too
                 generic to check - they would flag every ordinary dodge move.
  jaccard        Jaccard similarity of the CONTENT-token sets (stop words dropped) is at least JACCARD_MAX; catches
                 reworded copies. Short questions share templates («Що таке кидок X?») so it only applies when both
                 sides have at least MIN_JACCARD_CONTENT content tokens.
  subset         for gold texts with MIN_SUBSET_CONTENT..MIN_JACCARD_CONTENT-1 content tokens (too few for Jaccard):
                 the text contains ALL of the gold's content tokens plus at most one more - a word-order permutation
                 or lightly padded copy («Як бонусний кубик працює?»). Not reported when an n-gram hit already is.
  forbidden      a conflict probe's `replace` or `perturbed_fact` occurs word-bounded in a text (the model must
                 never learn the counterfactual wording). Strings under MIN_FORBIDDEN_TOKENS words (e.g. "A POW
                 roll.") are too generic and are skipped. These strings are ENGLISH: a Ukrainian paraphrase of a
                 perturbed fact is not caught here - such examples need the oracle-section rule and review.
  oracle_section an oracle section is not in the TRAIN half of the section split (held-out or unknown)
  context_section the same rule for every section shown in the example's context
  scenario       the example is built from a scenario in FORBIDDEN_SCENARIOS (`demo-haunt` is the eval scenario)

Quoting an allowed passage is fine, but only for the n-gram rule: a shared n-gram is exempt when it also occurs in the
text of one of the example's own `context_sections` (train passages legitimately overlap gold questions, e.g. "an
investigator with CON 60 and SIZ 50"). Jaccard / subset are always evaluated, so a paraphrase of a gold question that
quotes a passage is still caught. The exemption needs `section_texts` (or `chunks`); without them nothing is exempt.

`Decontaminator.check_all` returns a `DecontamReport`; S4's builder drops or rewrites every example with a
violation and keeps the report with the dataset.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field

from evals.datasets import EvalDatasets, corpus_sections
from keeper.rag.corpus import Chunk
from sft.split import SectionKey, SectionSplit, chunk_ids_hash

N_GRAM = 8
MIN_VERBATIM_TOKENS = 4        # gold texts shorter than this are not checked at all
JACCARD_MAX = 0.6              # violation when content-token similarity >= this
MIN_JACCARD_CONTENT = 5        # ... and both sides have at least this many content tokens
MIN_SUBSET_CONTENT = 2         # gold texts with fewer content tokens are too generic for the subset rule
MIN_FORBIDDEN_TOKENS = 4
FORBIDDEN_SCENARIOS = frozenset({"demo-haunt"})

# straight, left/right single quotes, acute accent, modifier apostrophe/prime, backtick
_APOSTROPHE_CHARS = "'" + "".join(chr(c) for c in (0x2018, 0x2019, 0x00B4, 0x02B9, 0x02BC)) + "`"
_TOKEN = re.compile(r"\w+(?:[" + re.escape(_APOSTROPHE_CHARS) + r"]\w+)*")
_APOSTROPHES = str.maketrans("", "", _APOSTROPHE_CHARS)

# Negations (не/ні, not/no) are stop words on purpose: a question of the opposite polarity about the same fact
# reveals the same answer, so it counts as a leak. No stemming: inflected paraphrases (counts/counted) can pass.
STOP_WORDS = frozenset(
    # Ukrainian
    "як що таке чи це в у на і й та а з із до за по про для від при є не ні же ж би б я ти ми ви він вона вони "
    "мені можна може які який яка яке чим де коли цей ця ці той то так є був була були бути буде "
    # English
    "a an the is are was were be been being how what which who whom when where why can could do does did of to in "
    "on at for with by and or it its this that these those as if i you my we they he she there not no".split())


def tokenize(text: str) -> list[str]:
    return [t.translate(_APOSTROPHES) for t in _TOKEN.findall(unicodedata.normalize("NFC", text).casefold())]


def content_tokens(tokens: Iterable[str]) -> frozenset[str]:
    return frozenset(t for t in tokens if t not in STOP_WORDS)


def ngrams(tokens: list[str], n: int = N_GRAM) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: Collection[str], b: Collection[str]) -> float:
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def _contains(haystack: list[str], needle: list[str]) -> bool:
    """Contiguous token-sequence containment (word boundaries respected)."""
    m = len(needle)
    return m > 0 and any(haystack[i:i + m] == needle for i in range(len(haystack) - m + 1))


@dataclass(frozen=True)
class Example:
    """One SFT example as plain texts. `oracle_sections` / `context_sections` are TRUSTED: the builder (S4) must derive
    `context_sections` from exactly the retrieval hits rendered into the prompt (distractors included), because the
    train-only rule and the quoting exemption rely on it."""

    id: str
    user_turns: tuple[str, ...]
    target: str
    assistant_turns: tuple[str, ...] = ()           # prior Keeper turns (history), not the target
    oracle_sections: tuple[SectionKey, ...] = ()    # sections the answer relies on
    context_sections: tuple[SectionKey, ...] = ()   # every section shown in the context, distractors included
    scenario_id: str | None = None


@dataclass(frozen=True)
class Violation:
    example_id: str
    check: str          # ngram | jaccard | subset | forbidden | oracle_section | context_section | scenario
    field: str          # user_turn[i] | assistant_turn[i] | target | oracle_sections | context_sections | scenario_id
    reason: str


@dataclass
class DecontamReport:
    examples: int = 0
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def failed_ids(self) -> set[str]:
        return {v.example_id for v in self.violations}

    def by_check(self) -> dict[str, int]:
        return dict(Counter(v.check for v in self.violations))

    def format(self) -> str:
        head = (f"{self.examples} examples, {len(self.failed_ids())} with violations "
                f"({len(self.violations)} total) {self.by_check()}")
        return "\n".join([head, *(f"  {v.example_id} [{v.check}] {v.field}: {v.reason}" for v in self.violations)])


@dataclass(frozen=True)
class _Gold:
    label: str
    tokens: list[str]
    grams: set[tuple[str, ...]]
    content: frozenset[str]


def _gold(label: str, text: str) -> _Gold:
    toks = tokenize(text)
    return _Gold(label, toks, ngrams(toks), content_tokens(toks))


def gold_texts(ds: EvalDatasets) -> list[tuple[str, str]]:
    out = []
    for q in ds.rules_qa:
        out += [(f"{q.id}.question_uk", q.question_uk), (f"{q.id}.question_en", q.question_en)]
    for p in ds.conflict_probes:
        out += [(f"{p.id}.question_uk", p.question_uk), (f"{p.id}.question_en", p.question_en)]
    for a in ds.adjudication:
        out += [(f"{a.id}.scene", a.scene), (f"{a.id}.player_text", a.player_text)]
    return out


def forbidden_strings(ds: EvalDatasets) -> list[tuple[str, str]]:
    out = []
    for p in ds.conflict_probes:
        out += [(f"{p.id}.replace", p.replace), (f"{p.id}.perturbed_fact", p.perturbed_fact)]
    return out


def section_texts_from_chunks(chunks: list[Chunk]) -> dict[SectionKey, str]:
    return {key: "\n".join(c.text for c in cs) for key, cs in corpus_sections(chunks).items()}


class Decontaminator:
    def __init__(self, gold: Iterable[tuple[str, str]], forbidden: Iterable[tuple[str, str]],
                 train_sections: Collection[SectionKey], heldout_sections: Collection[SectionKey] = (),
                 forbidden_scenarios: Collection[str] = FORBIDDEN_SCENARIOS,
                 section_texts: Mapping[SectionKey, str] | None = None):
        self.gold = [g for label, text in gold if len((g := _gold(label, text)).tokens) >= MIN_VERBATIM_TOKENS]
        self.forbidden = [(label, toks) for label, text in forbidden
                          if len(toks := tokenize(text)) >= MIN_FORBIDDEN_TOKENS]
        self.train = set(train_sections)
        self.heldout = set(heldout_sections)
        self.forbidden_scenarios = set(forbidden_scenarios)
        self.section_tokens = {k: tokenize(t) for k, t in (section_texts or {}).items()}

    @classmethod
    def from_datasets(cls, ds: EvalDatasets, split: SectionSplit, chunks: list[Chunk] | None = None,
                      section_texts: Mapping[SectionKey, str] | None = None) -> Decontaminator:
        """`chunks` (the corpus the split was frozen on) are verified against the split's chunk-ids hash and give the
        section texts for the quoting exemption; explicit `section_texts` override them (unverified)."""
        if split.gold_hashes != ds.hashes:
            raise ValueError(f"section split was frozen against other gold files: {split.gold_hashes} != {ds.hashes}")
        if chunks is not None:
            if chunk_ids_hash(chunks) != split.corpus_chunk_ids_hash:
                raise ValueError("section split was frozen against another corpus (chunk-ids hash differs)")
            section_texts = section_texts if section_texts is not None else section_texts_from_chunks(chunks)
        return cls(gold_texts(ds), forbidden_strings(ds), split.train_keys(), split.heldout_keys(),
                   section_texts=section_texts)

    def _quoted_from(self, context: Collection[SectionKey]) -> list[list[str]]:
        return [self.section_tokens[k] for k in context if k in self.section_tokens]

    def check_text(self, example_id: str, where: str, text: str,
                   context_sections: Collection[SectionKey] = ()) -> list[Violation]:
        toks = tokenize(text)
        if not toks:
            return []
        quoted = self._quoted_from(context_sections)
        quoted_grams = set().union(*(ngrams(q) for q in quoted)) if quoted else set()
        grams, content = ngrams(toks), content_tokens(toks)
        out = []
        for g in self.gold:
            if len(g.tokens) < N_GRAM:
                shared = [tuple(g.tokens)] if _contains(toks, g.tokens) else []
                blocking = [s for s in shared if not any(_contains(q, list(s)) for q in quoted)]
            else:
                shared = sorted(grams & g.grams)
                blocking = [s for s in shared if s not in quoted_grams]
            if blocking:
                out.append(Violation(example_id, "ngram", where,
                                     f"{len(blocking)} shared {N_GRAM}-grams with {g.label}, "
                                     f"e.g. {' '.join(blocking[0])!r}"))
            if min(len(content), len(g.content)) >= MIN_JACCARD_CONTENT:
                sim = jaccard(content, g.content)
                if sim >= JACCARD_MAX:
                    out.append(Violation(example_id, "jaccard", where,
                                         f"content-token Jaccard {sim:.2f} >= {JACCARD_MAX} with {g.label}"))
            elif (not blocking and MIN_SUBSET_CONTENT <= len(g.content) < MIN_JACCARD_CONTENT
                  and g.content <= content and len(content - g.content) <= 1):
                out.append(Violation(example_id, "subset", where,
                                     f"contains all content tokens of {g.label} ({', '.join(sorted(g.content))}) "
                                     f"plus at most one more"))
        for label, needle in self.forbidden:
            if _contains(toks, needle):
                out.append(Violation(example_id, "forbidden", where,
                                     f"contains the forbidden conflict-probe string {label}"))
        return out

    def _section_violations(self, ex: Example, keys: Iterable[SectionKey], check: str, where: str,
                            skip: Collection[SectionKey] = ()) -> list[Violation]:
        out = []
        for key in keys:
            if key in self.train or key in skip:
                continue
            kind = "held-out" if key in self.heldout else "unknown"
            out.append(Violation(ex.id, check, where,
                                 f"{kind} section {key[0]!r} / {key[1]!r} is not in the train split"))
        return out

    def check(self, ex: Example) -> list[Violation]:
        out = []
        ctx = set(ex.context_sections) | set(ex.oracle_sections)
        for i, turn in enumerate(ex.user_turns):
            out += self.check_text(ex.id, f"user_turn[{i}]", turn, ctx)
        for i, turn in enumerate(ex.assistant_turns):
            out += self.check_text(ex.id, f"assistant_turn[{i}]", turn, ctx)
        out += self.check_text(ex.id, "target", ex.target, ctx)
        out += self._section_violations(ex, ex.oracle_sections, "oracle_section", "oracle_sections")
        out += self._section_violations(ex, ex.context_sections, "context_section", "context_sections",
                                        skip=ex.oracle_sections)
        if ex.scenario_id in self.forbidden_scenarios:
            out.append(Violation(ex.id, "scenario", "scenario_id",
                                 f"scenario {ex.scenario_id!r} is reserved for evaluation"))
        return out

    def check_all(self, examples: Iterable[Example]) -> DecontamReport:
        report = DecontamReport()
        for ex in examples:
            report.examples += 1
            report.violations += self.check(ex)
        return report
