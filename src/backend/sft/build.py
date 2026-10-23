"""Build the frozen SFT narrator inputs from hand-written seeds.

    cd src/backend
    uv run python -m sft.build inputs            # validate the seeds, print a summary (nothing is written)
    uv run python -m sft.build inputs --write    # write data/sft/coc7e/inputs.v1.jsonl and the briefs
    uv run python -m sft.build inputs --check    # exit 1 if the committed inputs differ from a rebuild

For every seed (`sft.schemas.SftSeed`) the builder
  1. validates it against the training scenario pack (investigator, skill and characteristic names exactly as on the
     sheet, sanity events) and against the frozen section split (an oracle section must be in the TRAIN half);
  2. rolls the dice with `rules.resolve` (code decides outcomes; the seed only says what to roll) with a seed derived
     from the example id, and `turn_no` = the number of player turns in the history including this one;
  3. picks the knowledge-base chunks like the retriever returns them: all chunks of the oracle sections plus
     deterministic distractors from other TRAIN sections, shuffled by the example id, grouped by source type in quota
     order, markers [R1]..., within the retrieval quotas of the rendering spec. A distractor never comes from an oracle
     or `kb.exclude_sections` section, nor from a SIBLING of one (same source, same top-level heading): a related
     section could answer the question, and showing an answering passage as "not an answer" would teach the model to
     ignore the knowledge base;
  4. renders the chat with `keeper.nodes.narrate_messages` (the production narrator prompt, keeper-v2, for the
     generator profile of the Phase 4 base model);
  5. runs the training gate `sft.gates.check_messages` over the chat (NFC, no outer whitespace on replies; a
     placeholder reply stands in for the target) and the decontaminator over the user and assistant turns (the
     target is empty until S5).
All problems of all seeds are reported together ("seed <id>: <reason>"); nothing is written unless every seed is fine.

The output is deterministic: the same seeds, scenarios, corpus and code give byte-identical files. It is meant to be
COMMITTED and then treated as frozen: the recorded rolls and chunk ids keep targets consistent with the dice even if
the rules code changes later; `--check` shows whether a rebuild would differ.

Training scenarios live in `data/sft/coc7e/scenarios/<id>/scenario.yaml` (same `ScenarioPack` schema as the app's
`data/scenarios`, loaded by a separate `ScenarioRepo` root, so they never show up in the UI). `demo-haunt` is the
evaluation scenario and is forbidden here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from pydantic import ValidationError

from evals.datasets import CORPUS_DIR, DATASET_DIR, EvalDatasets, corpus_sections
from keeper.game.coc7e import rules
from keeper.llm import ModelProfile
from keeper.nodes import narrate_messages, turn_number
from keeper.prompts import PromptSet
from keeper.rag.corpus import Chunk, load_corpus
from keeper.rag.retriever import EXCERPT_CHARS, HEADING_MARKS
from keeper.scenarios import ScenarioPack, ScenarioRepo
from keeper.spec import ExperimentSpec, RetrievalSpec
from server.settings import BACKEND_DIR, REPO_ROOT
from sft import gates
from sft.decontam import FORBIDDEN_SCENARIOS, Decontaminator, Example
from sft.schemas import KbHit, Message, RenderInfo, SectionRef, SftInput, SftSeed
from sft.split import CHUNK_SIZE, OVERLAP, SectionKey, SectionSplit, chunk_ids_hash, load_split

SFT_DIR = REPO_ROOT / "data" / "sft" / "coc7e"
SEEDS_PATH = SFT_DIR / "seeds.v1.jsonl"
SCENARIOS_DIR = SFT_DIR / "scenarios"
INPUTS_PATH = SFT_DIR / "inputs.v1.jsonl"
BRIEFS_PATH = REPO_ROOT / "artifacts" / "sft" / "coc7e-v1" / "briefs.jsonl"
MODELS_YAML = BACKEND_DIR / "config" / "models.yaml"

# What the training inputs are rendered with: the Phase 3 local-qwen setup (experiments/phase3-local-qwen.yaml,
# play.yaml). The narrator is the only trained part, so aux / adjudicate / analyze are recorded but never run.
GENERATOR = "qwen25-3b-ollama"
PROMPT_VERSION = "keeper-v2"
ARM = "sft-narrator-v1"
HISTORY_MAX_TOKENS = 6000
MIN_SCORE = 0.55
DICE_BASE_SEED = 1927_04
MIN_DISTRACTOR_CHARS = 200   # a heading-only or one-line chunk is a poor distractor

ROLL_BRIEF_KEYS = ("kind", "target", "value", "difficulty", "bonus", "penalty", "roll", "level", "passed", "pushed",
                   "notation", "san_loss", "san_after", "involuntary_action", "indefinite_insanity",
                   "permanent_insanity", "temporary_insanity", "purpose")


class BuildError(ValueError):
    """All problems found, one per line, each starting with 'seed <id>:' (or a file position)."""

    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


class SeedError(ValueError):
    pass


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def dice_seed(example_id: str, base: int = DICE_BASE_SEED) -> int:
    """Per-example dice seed: stable across processes and independent of the seed file order."""
    return int(hashlib.sha256(f"{base}:dice:{example_id}".encode()).hexdigest()[:12], 16)


def render_spec(use_rag: bool) -> ExperimentSpec:
    return ExperimentSpec(arm=ARM, generator=GENERATOR, use_rag=use_rag,
                          retrieval=RetrievalSpec(min_score=MIN_SCORE), prompt_version=PROMPT_VERSION,
                          adjudicate_prompt_version="adjudicate-v3", analyze_prompt_version="analyze-v1",
                          history_max_tokens=HISTORY_MAX_TOKENS)


def load_profile(name: str = GENERATOR, path: Path = MODELS_YAML) -> ModelProfile:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if name not in raw:
        raise ValueError(f"unknown model profile {name!r} in {path}")
    return ModelProfile(**raw[name])


@dataclass
class BuildContext:
    scenarios: ScenarioRepo
    chunks: list[Chunk]
    split: SectionSplit
    decontam: Decontaminator
    prompts: PromptSet = field(default_factory=PromptSet)
    profile: ModelProfile = field(default_factory=load_profile)

    def __post_init__(self):
        self.sections = corpus_sections(self.chunks)
        self.train = self.split.train_keys()
        self.heldout = self.split.heldout_keys()
        self.corpus_hash = chunk_ids_hash(self.chunks)


@dataclass
class Built:
    input: SftInput
    brief: dict
    example: Example
    spec: ExperimentSpec
    state: dict          # the narrate state the chat was rendered from (not written out; used by parity tests)


# ------------------------------------------------------------------------------------------ seeds


def load_seeds(path: Path) -> list[SftSeed]:
    if not path.is_file():
        raise BuildError([f"seeds file not found: {path}"])
    seeds, problems, seen = [], [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            seed = SftSeed.model_validate(json.loads(line))
        except json.JSONDecodeError as e:
            problems.append(f"{path.name}:{n}: invalid JSON ({e})")
            continue
        except ValidationError as e:
            problems.append(f"seed {_row_id(line)} ({path.name}:{n}): {_validation_text(e)}")
            continue
        if seed.id in seen:
            problems.append(f"seed {seed.id}: duplicate id ({path.name}:{n})")
        seen.add(seed.id)
        seeds.append(seed)
    if problems:
        raise BuildError(problems)
    return seeds


def _row_id(line: str) -> str:
    try:
        row = json.loads(line)
        return str(row["id"]) if isinstance(row, dict) and "id" in row else "?"
    except (json.JSONDecodeError, TypeError):
        return "?"


def _validation_text(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, err['loc'])) or 'seed'}: {err['msg']}" for err in e.errors())


# ------------------------------------------------------------------------------------------ knowledge base


def _hit(marker: int, chunk: Chunk) -> dict:
    """A retriever hit (keeper.rag.retriever.Retriever._hits). There is no similarity here, so `score` is None."""
    meta = chunk.metadata
    return {"marker": marker, "chunk_id": chunk.id, "score": None, "source": meta["source_title"],
            "section": meta["section_path"], "source_type": meta["source_type"], "page": meta.get("page"),
            "text": chunk.text, "excerpt": HEADING_MARKS.sub("", chunk.text)[:EXCERPT_CHARS]}


def _section_key(chunk: Chunk) -> SectionKey:
    return (chunk.metadata["source_title"], chunk.metadata["section_path"])


def _in_scope(chunk: Chunk, game_system: str, quota_types: set[str]) -> bool:
    m = chunk.metadata
    return (m["game_system"] == game_system and m.get("visibility", "public") == "public"
            and m["source_type"] in quota_types)


def select_chunks(seed: SftSeed, ctx: BuildContext, game_system: str,
                  quotas: dict[str, int]) -> list[tuple[Chunk, str]]:
    """The (chunk, role) pairs the prompt shows, in marker order. Raises SeedError."""
    quota_types = {t for t, n in quotas.items() if n > 0}
    chosen: list[tuple[Chunk, str]] = []
    for ref in seed.kb.oracle_sections:
        key = ref.key()
        if key in ctx.heldout:
            raise SeedError(f"oracle section {ref.source!r} / {ref.section!r} is held-out (not allowed in training)")
        if key not in ctx.train:
            known = "in the corpus but not in the split" if key in ctx.sections else "not in the corpus"
            raise SeedError(f"oracle section {ref.source!r} / {ref.section!r} is not a train section ({known})")
        in_scope = [c for c in ctx.sections[key] if _in_scope(c, game_system, quota_types)]
        if not in_scope:
            raise SeedError(f"oracle section {ref.source!r} / {ref.section!r} is not retrievable for {game_system} "
                            f"(source types {sorted(quota_types)})")
        chosen += [(c, "oracle") for c in sorted(in_scope, key=lambda c: c.metadata["chunk_index"])]
    oracle_keys = {r.key() for r in seed.kb.oracle_sections}

    exclude_keys = {r.key() for r in seed.kb.exclude_sections}
    for ref in seed.kb.exclude_sections:
        if ref.key() not in ctx.sections:
            raise SeedError(f"exclude section {ref.source!r} / {ref.section!r} is not in the corpus")
    blocked = oracle_keys | exclude_keys
    blocked_groups = {_group(k) for k in blocked}
    rng = random.Random(f"{DICE_BASE_SEED}:distractors:{seed.id}")
    pool: list[SectionKey] = sorted(
        key for key in ctx.train
        if key not in blocked and _group(key) not in blocked_groups
        and any(_eligible_distractor(c, game_system, quota_types) for c in ctx.sections.get(key, [])))
    used: dict[str, int] = {t: sum(1 for c, _ in chosen if c.metadata["source_type"] == t) for t in quota_types}
    for _ in range(seed.kb.n_distractors):
        candidates = [k for k in pool if used[_section_type(ctx, k)] < quotas[_section_type(ctx, k)]]
        if not candidates:
            raise SeedError(f"cannot add distractor #{len(chosen) + 1}: no train section left within the quotas "
                            f"{quotas}")
        key = rng.choice(candidates)
        pool.remove(key)
        options = sorted((c for c in ctx.sections[key] if _eligible_distractor(c, game_system, quota_types)),
                         key=lambda c: c.metadata["chunk_index"])
        chunk = rng.choice(options)
        used[chunk.metadata["source_type"]] += 1
        chosen.append((chunk, "distractor"))

    for t in sorted(quota_types):
        n = sum(1 for c, _ in chosen if c.metadata["source_type"] == t)
        if n > quotas[t]:
            raise SeedError(f"needs {n} {t!r} chunks in the prompt but the retrieval quota allows {quotas[t]}")
    order = random.Random(f"{DICE_BASE_SEED}:order:{seed.id}")
    order.shuffle(chosen)
    rank = {t: i for i, t in enumerate(quotas)}   # the retriever lists hits per source type, in quota order
    chosen.sort(key=lambda cr: rank.get(cr[0].metadata["source_type"], len(rank)))
    return chosen


def _group(key: SectionKey) -> tuple[str, str]:
    """Source plus parent heading path: the sections that count as siblings of each other.

    The parent path (not the first component) keeps lore files, whose paths start with the document h1
    ("Lviv in 1927: Setting Notes > Places > ..."), from collapsing into one group; a top-level section is
    its own group.
    """
    parts = key[1].split(" > ")
    return (key[0], " > ".join(parts[:-1] or parts))


def _eligible_distractor(chunk: Chunk, game_system: str, quota_types: set[str]) -> bool:
    return len(chunk.text) >= MIN_DISTRACTOR_CHARS and _in_scope(chunk, game_system, quota_types)


def _section_type(ctx: BuildContext, key: SectionKey) -> str:
    return ctx.sections[key][0].metadata["source_type"]


# ------------------------------------------------------------------------------------------ one seed


def _to_message(m: AnyMessage) -> Message:
    role = {"system": "system", "human": "user", "ai": "assistant"}[m.type]
    return Message(role=role, content=m.text)


def _check_against_pack(seed: SftSeed, pack: ScenarioPack, sheet: dict) -> None:
    a = seed.adjudication
    names = rules.sheet_names(sheet)
    if a.roll_type == "skill" and a.target not in names["skills"]:
        raise SeedError(f"skill {a.target!r} is not on the sheet of {sheet['name']!r} (skills: {names['skills']})")
    if a.roll_type == "characteristic" and a.target not in names["characteristics"]:
        raise SeedError(f"characteristic {a.target!r} is not one of {names['characteristics']}")
    if a.sanity_event and a.sanity_event not in pack.sanity_events:
        raise SeedError(f"sanity_event {a.sanity_event!r} is not defined by scenario {pack.id!r} "
                        f"(has {sorted(pack.sanity_events)})")


def build_one(seed: SftSeed, ctx: BuildContext) -> Built:
    """Raises SeedError with the reason; the caller prefixes the seed id."""
    if seed.scenario_id in FORBIDDEN_SCENARIOS:
        raise SeedError(f"scenario {seed.scenario_id!r} is reserved for evaluation")
    try:
        pack = ctx.scenarios.load(seed.scenario_id)
    except KeyError:
        raise SeedError(f"unknown training scenario {seed.scenario_id!r} "
                        f"(available: {ctx.scenarios.ids()} in {ctx.scenarios.root})") from None
    if seed.investigator_id and seed.investigator_id not in {p.id for p in pack.investigators}:
        raise SeedError(f"scenario {pack.id!r} has no investigator {seed.investigator_id!r}")
    investigator = pack.investigator(seed.language, seed.investigator_id)
    _check_against_pack(seed, pack, investigator)

    scenario = pack.scenario_state(seed.language)
    messages: list[AnyMessage] = [AIMessage(content=scenario["opening"])]
    messages += [HumanMessage(content=t.text) if t.role == "player" else AIMessage(content=t.text)
                 for t in seed.history]
    messages.append(HumanMessage(content=seed.player_text))
    turn_no = turn_number(messages)

    adjudication = seed.adjudication.model_dump()
    dseed = dice_seed(seed.id)
    rolls: list[dict] = []
    errors: list[str] = []
    if adjudication["needs_roll"] or adjudication["new_day"]:   # the route_after_adjudicate condition
        res = rules.resolve(investigator, adjudication, [], seed=dseed, turn_no=turn_no,
                            sanity_events=scenario.get("sanity_events"))
        investigator, rolls, errors = res.investigator, res.records, res.errors
    if errors:
        raise SeedError(f"the rules engine rejected the adjudication: {errors}")

    spec = render_spec(seed.use_rag)
    chosen = select_chunks(seed, ctx, pack.game_system, spec.retrieval.quotas) if seed.use_rag else []
    hits = [_hit(i, c) for i, (c, _) in enumerate(chosen, start=1)]
    state = {"scenario": scenario, "investigator": investigator, "messages": messages, "adjudication": adjudication,
             "turn_rolls": rolls, "errors": errors, "rules_hits": hits}
    try:
        rendered = narrate_messages(ctx.prompts, ctx.profile, spec, state)
    except ValueError as e:
        raise SeedError(f"cannot render the prompt: {e}") from e
    chat = [_to_message(m) for m in rendered]
    try:   # the final assistant message is the target, which does not exist yet: a placeholder satisfies the shape
        gates.check_messages([{"messages": [m.model_dump() for m in chat] + [{"role": "assistant", "content": "x"}]}],
                             seed.id)
    except gates.GateError as e:
        raise SeedError(f"training gate: {e}") from e

    kb_hits = [KbHit(marker=h["marker"], chunk_id=h["chunk_id"], source=h["source"], section=h["section"],
                     source_type=h["source_type"], role=role) for h, (_, role) in zip(hits, chosen, strict=True)]
    context_keys = list(dict.fromkeys(_section_key(c) for c, _ in chosen))
    oracle_keys = [r.key() for r in seed.kb.oracle_sections]
    built = SftInput(
        id=seed.id, seed_hash=sha16(seed.model_dump_json()), scenario_id=seed.scenario_id, language=seed.language,
        kind=seed.kind, use_rag=seed.use_rag, investigator_id=seed.investigator_id, turn_no=turn_no,
        dice_seed=dseed, history=seed.history, player_text=seed.player_text, adjudication=adjudication,
        rolls=rolls, errors=errors, kb_mode=seed.kb.mode, oracle_sections=seed.kb.oracle_sections,
        context_sections=[SectionRef(source=s, section=p) for s, p in context_keys], hits=kb_hits,
        render=RenderInfo(spec=spec.model_dump(mode="json"), spec_hash=spec.spec_hash(),
                          generator_profile=ctx.profile.model_dump(mode="json"), prompt_version=spec.prompt_version,
                          prompt_hash=ctx.prompts.hash(spec.prompt_version), corpus_chunk_ids_hash=ctx.corpus_hash,
                          split_seed=ctx.split.seed, dice_base_seed=DICE_BASE_SEED),
        messages=chat, input_hash=sha16(json.dumps([m.model_dump() for m in chat], ensure_ascii=False)))
    example = Example(
        id=seed.id, user_turns=tuple([t.text for t in seed.history if t.role == "player"] + [seed.player_text]),
        assistant_turns=tuple(t.text for t in seed.history if t.role == "keeper"), target="",
        oracle_sections=tuple(oracle_keys), context_sections=tuple(context_keys), scenario_id=seed.scenario_id)
    return Built(built, make_brief(built, pack, investigator, scenario, chosen), example, spec, state)


def make_brief(inp: SftInput, pack: ScenarioPack, investigator: dict, scenario: dict,
               chosen: list[tuple[Chunk, str]]) -> dict:
    """What the target author needs, short: the scene, the turn, the frozen rolls and the passages with their roles."""
    a = inp.adjudication
    return {
        "id": inp.id, "language": inp.language, "kind": inp.kind, "use_rag": inp.use_rag, "kb_mode": inp.kb_mode,
        "input_hash": inp.input_hash,
        "scenario": {"id": pack.id, "title": scenario["title"], "era": scenario["era"]},
        "investigator": {"name": investigator["name"], "occupation": investigator["occupation"]},
        "turn_no": inp.turn_no,
        "scene": inp.history[-1].text if inp.history else scenario["opening"],
        "player_text": inp.player_text,
        "adjudication": brief_adjudication(a),
        "rolls": [{k: r[k] for k in ROLL_BRIEF_KEYS if k in r} for r in inp.rolls],
        "passages": [{"marker": f"R{h.marker}", "role": h.role, "where": f"{h.source_type}: {h.source} > {h.section}",
                      "text": c.text} for h, (c, _) in zip(inp.hits, chosen, strict=True)],
        "guidance": guidance(inp),
    }


def brief_adjudication(a: dict) -> dict:
    out = {"kind": a["kind"], "roll_type": a["roll_type"]}
    if a["roll_type"] != "none":
        out |= {k: a[k] for k in ("target", "difficulty", "bonus_dice", "penalty_dice", "sanity_event", "sanity_loss")
                if a.get(k)}
    out |= {k: a[k] for k in ("new_day", "reason") if a.get(k)}
    return out


def guidance(inp: SftInput) -> list[str]:
    out = ["Write only the Keeper reply, in the reply language, as the keeper-v2 prompt demands."]
    if inp.rolls:
        out.append("Narrate exactly the rolled result, then one short out-of-character line starting with 🎲.")
    if inp.kind == "rules_question":
        out.append("Out-of-character rules answer, then return to the scene in one sentence.")
    elif inp.kind == "meta":
        out.append("Out-of-character: answer briefly, then invite the player back into the scene.")
    if inp.use_rag and inp.kb_mode == "oracle":
        out.append("Cite the passage(s) with role 'oracle' as [R#] in out-of-character text only; do not cite "
                   "distractors and do not use their content.")
    elif inp.use_rag and inp.kb_mode == "distractor_only":
        out.append("No passage answers this: say \"Keeper ruling\" explicitly and answer from general Call of "
                   "Cthulhu 7e knowledge; do not cite [R#].")
    elif inp.use_rag:
        out.append("No passages were retrieved: do not cite [R#]; for a rules question say \"Keeper ruling\".")
    return out


# ------------------------------------------------------------------------------------------ all seeds


def build_all(seeds: Iterable[SftSeed], ctx: BuildContext) -> list[Built]:
    built, problems = [], []
    for seed in seeds:
        try:
            built.append(build_one(seed, ctx))
        except SeedError as e:
            problems.append(f"seed {seed.id}: {e}")
    report = ctx.decontam.check_all(b.example for b in built)
    problems += [f"seed {v.example_id}: decontamination [{v.check}] {v.field}: {v.reason}" for v in report.violations]
    if problems:
        raise BuildError(problems)
    return built


def dump_jsonl(rows: Iterable[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def render_inputs(built: list[Built]) -> str:
    return dump_jsonl(b.input.model_dump(mode="json") for b in built)


def render_briefs(built: list[Built]) -> str:
    return dump_jsonl(b.brief for b in built)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def summary(built: list[Built]) -> str:
    def count(key) -> dict:
        out: dict = {}
        for b in built:
            out[key(b.input)] = out.get(key(b.input), 0) + 1
        return dict(sorted(out.items(), key=str))

    return (f"{len(built)} inputs: kind {count(lambda i: i.kind)}, kb {count(lambda i: i.kb_mode)}, "
            f"rag {count(lambda i: i.use_rag)}, language {count(lambda i: i.language)}, "
            f"with rolls {sum(1 for b in built if b.input.rolls)}")


# ------------------------------------------------------------------------------------------ CLI


def load_context(scenarios_dir: Path) -> BuildContext:
    ds = EvalDatasets.load(DATASET_DIR)
    chunks = load_corpus([CORPUS_DIR], chunk_size=CHUNK_SIZE, overlap=OVERLAP, root=REPO_ROOT)
    split = load_split()
    return BuildContext(scenarios=ScenarioRepo(scenarios_dir), chunks=chunks, split=split,
                        decontam=Decontaminator.from_datasets(ds, split, chunks=chunks))


def cmd_inputs(args: argparse.Namespace) -> int:
    try:
        built = build_all(load_seeds(args.seeds), load_context(args.scenarios))
    except BuildError as e:
        print(f"{len(e.problems)} problem(s):")
        for line in e.problems:
            print(f"  {line}")
        return 1
    text = render_inputs(built)
    print(summary(built))
    if args.write:
        write_text(args.out, text)
        write_text(args.briefs, render_briefs(built))
        print(f"wrote {args.out}\nwrote {args.briefs}")
    elif args.check:
        committed = args.out.read_bytes().decode("utf-8").replace("\r\n", "\n") if args.out.exists() else None
        if committed != text:
            print(f"DIFFERS: {args.out} is missing or does not match a rebuild "
                  "(seeds, scenarios, corpus, prompts or rules code changed?)")
            return 1
        print("OK: committed inputs reproduced")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("inputs", help="seeds -> frozen inputs + briefs")
    q.add_argument("--seeds", type=Path, default=SEEDS_PATH)
    q.add_argument("--scenarios", type=Path, default=SCENARIOS_DIR, help="training scenario packs root")
    q.add_argument("--out", type=Path, default=INPUTS_PATH)
    q.add_argument("--briefs", type=Path, default=BRIEFS_PATH)
    mode = q.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write the inputs and the briefs")
    mode.add_argument("--check", action="store_true", help="exit 1 if the committed inputs differ from a rebuild")
    q.set_defaults(func=cmd_inputs)
    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
