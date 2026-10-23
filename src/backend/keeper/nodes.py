"""Graph nodes.

RAG arms: analyze (English search query) -> retrieve (Qdrant) -> adjudicate -> [mechanics] -> narrate.
Prompt-only arms skip analyze/retrieve. The LLM decides what to roll and how to tell it; code rolls the dice.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError

from . import citations
from .game.coc7e import rules
from .llm import ModelProfile
from .prompts import PromptSet
from .spec import ExperimentSpec, KeeperContext
from .state import KeeperState

if TYPE_CHECKING:
    from .deps import Deps

LANGUAGE_NAMES = {"uk": "Ukrainian", "en": "English"}
# Approximate characters per token, used only to budget history; recalibrate from run-log usage.
CHARS_PER_TOKEN = {"uk": 2.8, "en": 4.0}


# Keep structured-output schemas to the portable JSON-schema subset; clamp and validate numbers in code.
class Adjudication(BaseModel):
    kind: Literal["in_fiction", "rules_question", "meta"]
    needs_roll: bool
    roll_type: Literal["skill", "characteristic", "luck", "sanity", "none"] = "none"
    target: str | None = Field(None, description="Exact skill or characteristic name from the sheet, or null")
    difficulty: Literal["regular", "hard", "extreme"] = "regular"
    bonus_dice: int = Field(0, description="0, 1 or 2")
    penalty_dice: int = Field(0, description="0, 1 or 2")
    sanity_event: str | None = Field(None, description="Key from SANITY EVENTS, or null")
    sanity_loss: str | None = Field(None, description="Sanity loss in X/Y notation such as '0/1D6', or null")
    is_push: bool = False
    reason: str = ""


class AdjudicationV3(Adjudication):
    new_day: bool = Field(False, description="True only if the player's action moves the story on to a later game day")


# Each adjudicate prompt version keeps the schema it was written for, so older campaigns and eval runs
# send exactly the request they always did.
ADJUDICATION_SCHEMAS: dict[str, type[Adjudication]] = {"adjudicate-v3": AdjudicationV3}


def adjudication_schema(version: str) -> type[Adjudication]:
    return ADJUDICATION_SCHEMAS.get(version, Adjudication)


class SearchQuery(BaseModel):
    query_en: str = Field(description="Standalone English search query for the rules and setting knowledge base")


class EmptyGeneration(RuntimeError):
    """Safety block or output budget spent on thinking: never persist an empty Keeper turn."""


def transcript(msgs: list[AnyMessage], n: int = 6) -> str:
    lines = []
    for m in msgs[-n:]:
        who = "PLAYER" if m.type == "human" else "KEEPER"
        lines.append(f"{who}: {m.text}")
    return "\n".join(lines)


def turn_number(msgs: list[AnyMessage]) -> int:
    return sum(1 for m in msgs if m.type == "human")


def assert_ends_with_human(msgs: list[AnyMessage]) -> None:
    if not msgs or msgs[-1].type != "human":
        raise ValueError("last message must be a player turn (Gemini >= 3.6 rejects model prefill)")


# Shared by the graph nodes and the evaluation harness, so evals measure exactly the production logic.

def player_turn(text: str) -> str:
    """Player text is passed as data, never as instructions (defence against rhetorical injection)."""
    return "<player_turn>\n" + text + "\n</player_turn>"

async def search_query(deps: Deps, spec, scenario: dict, messages: list[AnyMessage]) -> tuple[str, list[str]]:
    """English standalone search query for the last player turn; falls back to the raw text on parse errors."""
    llm = deps.llms.structured(spec.aux, SearchQuery, seed=spec.seed)
    system = deps.prompts.render(spec.analyze_prompt_version, scenario=scenario, recent=transcript(messages[:-1]))
    player = messages[-1].text
    try:
        q: SearchQuery = await llm.ainvoke([SystemMessage(system), HumanMessage(player_turn(player))])
        query = q.query_en.strip()
    except (OutputParserException, ValidationError):
        query = ""
    return (query, []) if query else (player, ["analyze_parse_error"])


async def adjudicate_turn(deps: Deps, spec, scenario: dict, investigator: dict, roll_log: list[dict],
                          messages: list[AnyMessage], rules_hits: list[dict]) -> tuple[dict, list[str]]:
    profile = spec.aux if spec.adjudicator == "aux" else spec.generator
    llm = deps.llms.structured(profile, adjudication_schema(spec.adjudicate_prompt_version), seed=spec.seed)
    system = deps.prompts.render(
        spec.adjudicate_prompt_version,
        scenario=scenario,
        sheet=rules.sheet_names(investigator),
        last_roll=rules.last_player_roll(roll_log),
        recent=transcript(messages[:-1]),
        use_rag=spec.use_rag,
        rules_hits=rules_hits,
    )
    player = messages[-1].text
    try:
        adj: Adjudication = await llm.ainvoke([SystemMessage(system), HumanMessage(player_turn(player))])
    except (OutputParserException, ValidationError):  # transport errors propagate to the RetryPolicy
        return {"kind": "in_fiction", "needs_roll": False, "roll_type": "none"}, ["adjudication_parse_error"]
    data = adj.model_dump()
    if data["kind"] != "in_fiction" or data["roll_type"] == "none":
        data["needs_roll"] = False
    if not data["needs_roll"]:
        data["roll_type"] = "none"
    if data["kind"] != "in_fiction" and data.get("new_day"):
        data["new_day"] = False  # out-of-character talk never moves the game clock
    return data, []


def narrate_messages(prompts: PromptSet, profile: ModelProfile, spec: ExperimentSpec,
                     state: Mapping[str, Any]) -> list[AnyMessage]:
    """The exact message list the narrator model receives: system prompt followed by the trimmed history.

    Used by the `narrate` node and by the SFT builder, so training examples are rendered as they are served.
    `profile` is the generator's ModelProfile (its window and output budget size the history); `state` needs
    `scenario`, `investigator` and `messages` (ending with a player turn), and optionally `adjudication`,
    `turn_rolls`, `errors` and `rules_hits`. Raises ValueError if trimming leaves no player turn last.
    """
    scenario = state["scenario"]
    language = scenario.get("language", "uk")
    system = prompts.render(
        spec.prompt_version,
        scenario=scenario,
        language_name=LANGUAGE_NAMES.get(language, "the player's language"),
        sheet=rules.public_sheet(state["investigator"]),
        adjudication=state.get("adjudication"),
        rolls=state.get("turn_rolls") or [],
        errors=state.get("errors") or [],
        use_rag=spec.use_rag,
        rules_hits=state.get("rules_hits") or [],
    )
    counter = partial(count_tokens_approximately, chars_per_token=CHARS_PER_TOKEN.get(language, 3.0))
    system_tokens = counter([SystemMessage(system)])
    # Floor: a maximal 4000-character player message must always fit, or trimming would drop it.
    budget = max(2_000, min(spec.history_max_tokens,
                            profile.context_window - system_tokens - profile.max_tokens - 256))
    history = trim_messages(state["messages"], strategy="last", token_counter=counter, max_tokens=budget,
                            start_on="human", include_system=False)
    assert_ends_with_human(history)
    return [SystemMessage(system), *history]


class Nodes:
    def __init__(self, deps: Deps):
        self.d = deps

    async def seed_campaign(self, state: KeeperState, runtime: Runtime[KeeperContext]) -> dict:
        b = state["bootstrap"]
        pack = self.d.scenarios.load(b["scenario_id"])
        language = b.get("language", "uk")
        scenario = pack.scenario_state(language)
        return {
            "scenario": scenario,
            "investigator": pack.investigator(language, b.get("investigator_id")),
            "messages": [AIMessage(content=scenario["opening"], id=f"opening-{runtime.context.campaign_id}")],
            "roll_log": [],
            "citation_log": [],
            "bootstrap": None,
        }

    async def prepare_turn(self, state: KeeperState, runtime: Runtime[KeeperContext]) -> Command:
        assert_ends_with_human(state.get("messages", []))
        reset = {"search_query": "", "rules_hits": [], "adjudication": None, "turn_rolls": [], "errors": []}
        # Routing depends on the run context (the arm), which conditional-edge routers do not receive.
        return Command(update=reset, goto="analyze" if runtime.context.spec.use_rag else "adjudicate")

    async def analyze(self, state: KeeperState, runtime: Runtime[KeeperContext]) -> dict:
        runtime.stream_writer({"type": "progress", "payload": {"node": "analyze"}})
        if state.get("forced_hits") is not None:  # eval: frozen context, no query needed
            return {"search_query": ""}
        query, errors = await search_query(self.d, runtime.context.spec, state["scenario"], state["messages"])
        return {"search_query": query, "errors": [*state.get("errors", []), *errors]}

    async def retrieve(self, state: KeeperState, runtime: Runtime[KeeperContext]) -> dict:
        runtime.stream_writer({"type": "progress", "payload": {"node": "retrieve"}})
        if (forced := state.get("forced_hits")) is not None:
            return {"rules_hits": forced, "forced_hits": None}
        if self.d.retriever is None:
            raise RuntimeError("this arm uses RAG but no retriever is configured")
        hits = await self.d.retriever.search(state["search_query"], runtime.context.spec.retrieval,
                                             game_system=state["scenario"].get("game_system", "coc7e"))
        return {"rules_hits": hits}

    async def adjudicate(self, state: KeeperState, runtime: Runtime[KeeperContext]) -> dict:
        runtime.stream_writer({"type": "progress", "payload": {"node": "adjudicate"}})
        if forced := state.get("forced_kind"):  # eval: e.g. rules questions skip the adjudication LLM
            return {"adjudication": {"kind": forced, "needs_roll": False, "roll_type": "none"}, "forced_kind": None}
        data, errors = await adjudicate_turn(self.d, runtime.context.spec, state["scenario"], state["investigator"],
                                             state.get("roll_log") or [], state["messages"],
                                             state.get("rules_hits") or [])
        return {"adjudication": data, "errors": [*state.get("errors", []), *errors]}

    async def mechanics(self, state: KeeperState, runtime: Runtime[KeeperContext]) -> dict:
        runtime.stream_writer({"type": "progress", "payload": {"node": "mechanics"}})
        res = rules.resolve(
            state["investigator"], state["adjudication"], state.get("roll_log") or [],
            seed=runtime.context.spec.seed, turn_no=turn_number(state["messages"]),
            sanity_events=state["scenario"].get("sanity_events"), cfg=self.d.rules,
        )
        for rec in res.records:
            runtime.stream_writer({"type": "roll", "payload": rec})
        return {"investigator": res.investigator, "turn_rolls": res.records, "roll_log": res.records,
                "errors": [*state.get("errors", []), *res.errors]}

    async def narrate(self, state: KeeperState, runtime: Runtime[KeeperContext]) -> dict:
        runtime.stream_writer({"type": "progress", "payload": {"node": "narrate"}})
        spec = runtime.context.spec
        profile = self.d.llms.profile(spec.generator)
        messages = narrate_messages(self.d.prompts, profile, spec, state)
        llm = self.d.llms.chat(spec.generator, seed=spec.seed)
        msg = await llm.ainvoke(messages)
        if not msg.text.strip():
            raise EmptyGeneration(str(msg.response_metadata.get("finish_reason")))
        update: dict = {"messages": [msg]}  # the original message object keeps Gemini thought signatures
        if spec.use_rag:
            valid, invalid = citations.validate(msg.text, state.get("rules_hits") or [])
            update["citation_log"] = [{"turn_no": turn_number(state["messages"]), "citations": valid,
                                       "invalid_markers": invalid}]
        return update
