import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class KeeperState(TypedDict, total=False):
    # durable
    messages: Annotated[list[AnyMessage], add_messages]  # player HumanMessages + Keeper AIMessages only
    scenario: dict[str, Any]                             # {id, title, era, language, opening}
    investigator: dict[str, Any]                         # server-owned sheet; writers: seed_campaign, mechanics
    roll_log: Annotated[list[dict], operator.add]        # append-only audit trail of every roll
    # input of campaign creation only; cleared by seed_campaign
    bootstrap: dict | None
    # per-turn scratch, reset by prepare_turn
    adjudication: dict | None
    turn_rolls: list[dict]
    errors: list[str]
