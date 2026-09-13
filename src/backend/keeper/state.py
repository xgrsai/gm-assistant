import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class KeeperState(TypedDict, total=False):
    # durable
    messages: Annotated[list[AnyMessage], add_messages]  # player HumanMessages + Keeper AIMessages only
    scenario: dict[str, Any]                             # {id, title, era, language, opening, keeper_notes, ...}
    investigator: dict[str, Any]                         # server-owned sheet; writers: seed_campaign, mechanics
    roll_log: Annotated[list[dict], operator.add]        # append-only audit trail of every roll
    citation_log: Annotated[list[dict], operator.add]    # one record per RAG turn: valid citations + invalid markers
    # input of campaign creation only; cleared by seed_campaign
    bootstrap: dict | None
    # eval-only inputs (same graph topology as play), each cleared by the node that consumes it:
    forced_hits: list[dict] | None   # frozen or perturbed retrieval context -> skips analyze LLM + Qdrant
    forced_kind: str | None          # e.g. "rules_question" -> skips the adjudication LLM call
    # per-turn scratch, reset by prepare_turn
    search_query: str
    rules_hits: list[dict]
    adjudication: dict | None
    turn_rolls: list[dict]
    errors: list[str]
