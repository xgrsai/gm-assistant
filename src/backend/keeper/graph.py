"""One graph for every experiment arm; the arm is data in the run context."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import httpx
from langchain_core.exceptions import ModelConnectionError, ModelTimeoutError
from langgraph.errors import NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, TimeoutPolicy
from qdrant_client.http.exceptions import ResponseHandlingException

from .nodes import Nodes
from .spec import KeeperContext
from .state import KeeperState

if TYPE_CHECKING:
    from .deps import Deps

# Provider SDKs already retry 429/5xx; the graph retries only stalls and transport failures.
LLM_RETRY = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0,
                        retry_on=(NodeTimeoutError, ModelTimeoutError, ModelConnectionError,
                                  httpx.TransportError, ConnectionError, ResponseHandlingException))


def route_entry(state: KeeperState) -> Literal["seed_campaign", "prepare_turn"]:
    return "seed_campaign" if state.get("bootstrap") else "prepare_turn"


def route_after_adjudicate(state: KeeperState) -> Literal["mechanics", "narrate"]:
    adj = state.get("adjudication") or {}
    # A new game day changes the sheet even without a roll, so it also goes through the rules code.
    return "mechanics" if adj.get("needs_roll") or adj.get("new_day") else "narrate"


def build_graph(deps: Deps) -> StateGraph:
    n = Nodes(deps)
    g = StateGraph(KeeperState, context_schema=KeeperContext)
    g.add_node("seed_campaign", n.seed_campaign)
    g.add_node("prepare_turn", n.prepare_turn, destinations=("analyze", "adjudicate"))
    g.add_node("analyze", n.analyze, retry_policy=LLM_RETRY, timeout=60)
    g.add_node("retrieve", n.retrieve, retry_policy=LLM_RETRY, timeout=30)
    g.add_node("adjudicate", n.adjudicate, retry_policy=LLM_RETRY, timeout=60)
    g.add_node("mechanics", n.mechanics)
    # No RetryPolicy on narrate: its tokens are already on the wire; the player's Retry resumes instead.
    g.add_node("narrate", n.narrate, timeout=TimeoutPolicy(run_timeout=240, idle_timeout=90))

    g.add_conditional_edges(START, route_entry, ["seed_campaign", "prepare_turn"])
    g.add_edge("seed_campaign", END)
    g.add_edge("analyze", "retrieve")
    g.add_edge("retrieve", "adjudicate")
    g.add_conditional_edges("adjudicate", route_after_adjudicate, ["mechanics", "narrate"])
    g.add_edge("mechanics", "narrate")
    g.add_edge("narrate", END)
    return g
