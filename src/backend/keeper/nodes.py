"""Graph nodes. The LLM decides what to roll (adjudicate) and how to tell it (narrate); code rolls (mechanics)."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field, ValidationError

from .game.coc7e import rules
from .spec import KeeperContext
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
            "bootstrap": None,
        }

    async def prepare_turn(self, state: KeeperState, runtime: Runtime[KeeperContext]) -> dict:
        assert_ends_with_human(state.get("messages", []))
        return {"adjudication": None, "turn_rolls": [], "errors": []}

    async def adjudicate(self, state: KeeperState, runtime: Runtime[KeeperContext]) -> dict:
        runtime.stream_writer({"type": "progress", "payload": {"node": "adjudicate"}})
        spec = runtime.context.spec
        profile = spec.aux if spec.adjudicator == "aux" else spec.generator
        llm = self.d.llms.structured(profile, Adjudication, seed=spec.seed)
        roll_log = state.get("roll_log") or []
        system = self.d.prompts.render(
            spec.adjudicate_prompt_version,
            scenario=state["scenario"],
            sheet=rules.sheet_names(state["investigator"]),
            last_roll=rules.last_player_roll(roll_log),
            recent=transcript(state["messages"][:-1]),
        )
        player = state["messages"][-1].text
        try:
            adj: Adjudication = await llm.ainvoke(
                [SystemMessage(system), HumanMessage(f"<player_turn>\n{player}\n</player_turn>")])
        except (OutputParserException, ValidationError):  # transport errors propagate to the RetryPolicy
            return {"adjudication": {"kind": "in_fiction", "needs_roll": False, "roll_type": "none"},
                    "errors": [*state.get("errors", []), "adjudication_parse_error"]}
        data = adj.model_dump()
        if data["kind"] != "in_fiction" or data["roll_type"] == "none":
            data["needs_roll"] = False
        if not data["needs_roll"]:
            data["roll_type"] = "none"
        return {"adjudication": data}

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
        scenario = state["scenario"]
        language = scenario.get("language", "uk")
        system = self.d.prompts.render(
            spec.prompt_version,
            scenario=scenario,
            language_name=LANGUAGE_NAMES.get(language, "the player's language"),
            sheet=rules.public_sheet(state["investigator"]),
            adjudication=state.get("adjudication"),
            rolls=state.get("turn_rolls") or [],
            errors=state.get("errors") or [],
            use_rag=spec.use_rag,
            rules_hits=[],
        )
        profile = self.d.llms.profile(spec.generator)
        counter = partial(count_tokens_approximately, chars_per_token=CHARS_PER_TOKEN.get(language, 3.0))
        system_tokens = counter([SystemMessage(system)])
        # Floor: a maximal 4000-character player message must always fit, or trimming would drop it.
        budget = max(2_000, min(spec.history_max_tokens,
                                profile.context_window - system_tokens - profile.max_tokens - 256))
        history = trim_messages(state["messages"], strategy="last", token_counter=counter, max_tokens=budget,
                                start_on="human", include_system=False)
        assert_ends_with_human(history)
        llm = self.d.llms.chat(spec.generator, seed=spec.seed)
        msg = await llm.ainvoke([SystemMessage(system), *history])
        if not msg.text.strip():
            raise EmptyGeneration(str(msg.response_metadata.get("finish_reason")))
        return {"messages": [msg]}  # the original message object keeps Gemini thought signatures
