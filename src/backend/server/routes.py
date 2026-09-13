from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterable
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, RemoveMessage
from pydantic import BaseModel, StringConstraints

from keeper.game.coc7e.rules import public_sheet
from keeper.spec import KeeperContext

from .errors import error_event, map_error
from .store import Campaign

router = APIRouter()
log = logging.getLogger("gm.turns")

TURN_NODES = {"prepare_turn", "adjudicate", "mechanics", "narrate"}


# ---------------------------------------------------------------------------------------------- helpers


def thread(campaign_id: str) -> dict:
    return {"configurable": {"thread_id": campaign_id}}


def serialize_messages(values: dict) -> list[dict]:
    """Player/Keeper messages; each Keeper reply carries the rolls made in its turn."""
    by_turn: dict[int, list[dict]] = {}
    for rec in values.get("roll_log") or []:
        by_turn.setdefault(rec.get("turn_no", 0), []).append(rec)
    items, turn = [], 0
    for m in values.get("messages") or []:
        if m.type == "human":
            turn += 1
            items.append({"id": m.id, "role": "player", "text": m.text})
        else:
            items.append({"id": m.id, "role": "keeper", "text": m.text, "rolls": by_turn.get(turn, []) if turn else []})
    return items


def http_error(status: int, code: str, retryable: bool = False) -> HTTPException:
    return HTTPException(status, detail={"code": code, "retryable": retryable})


async def get_campaign(campaign_id: str, request: Request) -> Campaign:
    campaign = await request.app.state.store.get(campaign_id)
    if campaign is None:
        raise http_error(404, "not_found")
    return campaign


async def idle_campaign(request: Request, campaign: Campaign = Depends(get_campaign)) -> Campaign:
    if request.app.state.locks[campaign.id].locked():
        raise http_error(409, "turn_in_progress", True)
    return campaign


# ---------------------------------------------------------------------------------------------- meta


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/meta")
async def meta(request: Request) -> dict:
    deps = request.app.state.deps
    spec = deps.experiments.resolve()
    return {"version": request.app.version, "default_arm": spec.arm,
            "generator_model": deps.llms.profile(spec.generator).model}


@router.get("/scenarios")
async def scenarios(request: Request) -> dict:
    return {"items": [p.public() for p in request.app.state.deps.scenarios.all()]}


# ---------------------------------------------------------------------------------------------- campaigns


class CampaignIn(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
    scenario_id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9-]{1,64}$")] = "demo-haunt"
    language: Literal["uk", "en"] = "uk"


@router.get("/campaigns")
async def list_campaigns(request: Request) -> dict:
    return {"items": [c.public() for c in await request.app.state.store.list()]}


@router.post("/campaigns", status_code=201)
async def create_campaign(body: CampaignIn, request: Request) -> dict:
    st = request.app.state
    try:
        pack = st.deps.scenarios.load(body.scenario_id)
    except KeyError:
        raise HTTPException(422, detail={"code": "unknown_scenario", "retryable": False}) from None
    spec = st.deps.experiments.resolve()
    campaign = await st.store.create(title=body.title, scenario_id=pack.id,
                                     scenario_title=pack.localized(pack.title, body.language),
                                     language=body.language, spec=spec)
    await st.graph.ainvoke({"bootstrap": {"scenario_id": pack.id, "language": body.language}},
                           thread(campaign.id), context=KeeperContext(spec=spec, campaign_id=campaign.id),
                           durability="sync")
    return campaign.public()


@router.get("/campaigns/{campaign_id}")
async def get_campaign_detail(request: Request, campaign: Campaign = Depends(get_campaign)) -> dict:
    snap = await request.app.state.graph.aget_state(thread(campaign.id))
    return {**campaign.public(), "investigator": public_sheet(snap.values.get("investigator") or {})}


@router.get("/campaigns/{campaign_id}/messages")
async def get_messages(request: Request, campaign: Campaign = Depends(get_campaign)) -> dict:
    snap = await request.app.state.graph.aget_state(thread(campaign.id))
    return {"items": serialize_messages(snap.values), "incomplete": bool(snap.next),
            "investigator": public_sheet(snap.values.get("investigator") or {})}


@router.delete("/campaigns/{campaign_id}", status_code=204)
async def delete_campaign(request: Request, campaign: Campaign = Depends(idle_campaign)) -> Response:
    st = request.app.state
    await st.checkpointer.adelete_thread(campaign.id)
    await st.store.delete(campaign.id)
    st.locks.pop(campaign.id, None)
    return Response(status_code=204)


# ---------------------------------------------------------------------------------------------- turns


class TurnIn(BaseModel):
    client_turn_id: UUID
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


async def new_turn(body: TurnIn, request: Request, campaign: Campaign = Depends(idle_campaign)) -> Campaign:
    """Checks that must happen BEFORE the stream starts, so they can still be real HTTP status codes."""
    snap = await request.app.state.graph.aget_state(thread(campaign.id))
    if snap.next:
        raise http_error(409, "turn_incomplete", True)
    if any(m.id == f"h-{body.client_turn_id}" for m in snap.values.get("messages") or []):
        raise http_error(409, "duplicate_turn")  # add_messages would silently replace it in place
    return campaign


async def pending_turn(request: Request, campaign: Campaign = Depends(idle_campaign)) -> Campaign:
    snap = await request.app.state.graph.aget_state(thread(campaign.id))
    if not snap.next:
        raise http_error(409, "nothing_to_retry")
    return campaign


async def run_turn(request: Request, campaign: Campaign, inputs: dict | None) -> AsyncIterable[ServerSentEvent]:
    st = request.app.state
    graph, spec, deps = st.graph, campaign.spec, st.deps
    lock: asyncio.Lock = st.locks[campaign.id]
    if lock.locked():  # re-check: another request may have started between the dependency and here
        yield ServerSentEvent(event="error", data=error_event("turn_in_progress", True))
        return
    usage = UsageMetadataCallbackHandler()
    config = {**thread(campaign.id), "callbacks": [usage], "tags": [spec.arm],
              "metadata": {"arm": spec.arm, "spec_hash": spec.spec_hash()}}
    async with lock:
        t0, seen, timings, status = time.monotonic(), set(), {}, "cancelled"
        values: dict = {}
        try:
            async for part in graph.astream(inputs, config, context=KeeperContext(spec=spec, campaign_id=campaign.id),
                                            stream_mode=["updates", "messages", "custom"], version="v2",
                                            durability="sync"):  # dice checkpoint is on disk before narrate
                if part["type"] == "messages":
                    chunk, meta = part["data"]
                    if meta.get("langgraph_node") != "narrate":
                        continue
                    if isinstance(chunk, AIMessageChunk):
                        seen.add(chunk.id)
                    elif isinstance(chunk, AIMessage) and chunk.id in seen:
                        continue  # final message already streamed as chunks
                    if text := chunk.text:
                        yield ServerSentEvent(event="token", data={"text": text})
                elif part["type"] == "custom":
                    yield ServerSentEvent(event=part["data"]["type"], data=part["data"]["payload"])
                elif part["type"] == "updates":
                    for node in part["data"] or {}:
                        if node in TURN_NODES:
                            timings[node] = round((time.monotonic() - t0) * 1000)
            values = (await graph.aget_state(config)).values
            status = "done"
            await st.store.touch(campaign.id)
            yield ServerSentEvent(event="done", data={"message": serialize_messages(values)[-1],
                                                      "investigator": public_sheet(values["investigator"])})
        except Exception as exc:  # CancelledError (client left) is a BaseException and propagates
            log.exception("turn failed (campaign %s)", campaign.id)
            err = map_error(exc)
            status = err["code"]
            try:
                values = (await graph.aget_state(config)).values
            except Exception:
                log.exception("could not read state after failure")
            yield ServerSentEvent(event="error", data=err)
        finally:
            write_runlog(deps, campaign, values, usage, timings, status)


def write_runlog(deps, campaign: Campaign, values: dict, usage: UsageMetadataCallbackHandler, timings: dict,
                 status: str) -> None:
    spec = campaign.spec
    msgs = values.get("messages") or []
    last = msgs[-1] if msgs else None
    player = next((m for m in reversed(msgs) if m.type == "human"), None)
    try:
        deps.runlog.write({
            "campaign_id": campaign.id, "status": status, "arm": spec.arm, "spec_hash": spec.spec_hash(),
            "spec": spec.model_dump(mode="json"),
            "prompts": {v: deps.prompts.hash(v) for v in (spec.prompt_version, spec.adjudicate_prompt_version)},
            "models": {"generator": deps.llms.profile(spec.generator).model_dump(),
                       "aux": deps.llms.profile(spec.aux).model_dump()},
            "turn_no": sum(1 for m in msgs if m.type == "human"),
            "player_text": player.text if player else None,
            "keeper_text": last.text if last is not None and last.type == "ai" else None,
            "finish_reason": (last.response_metadata or {}).get("finish_reason") if last is not None else None,
            "adjudication": values.get("adjudication"), "rolls": values.get("turn_rolls") or [],
            "errors": values.get("errors") or [], "usage": usage.usage_metadata, "timings_ms": timings,
        })
    except Exception:
        log.exception("run log write failed")


@router.post("/campaigns/{campaign_id}/turns", response_class=EventSourceResponse)
async def post_turn(body: TurnIn, request: Request,
                    campaign: Campaign = Depends(new_turn)) -> AsyncIterable[ServerSentEvent]:
    inputs = {"messages": [HumanMessage(content=body.text, id=f"h-{body.client_turn_id}")]}
    async for event in run_turn(request, campaign, inputs):
        yield event


@router.post("/campaigns/{campaign_id}/turns/retry", response_class=EventSourceResponse)
async def retry_turn(request: Request, campaign: Campaign = Depends(pending_turn)) -> AsyncIterable[ServerSentEvent]:
    async for event in run_turn(request, campaign, None):  # None = resume from the last checkpoint
        yield event


@router.post("/campaigns/{campaign_id}/turns/discard")
async def discard_turn(request: Request, campaign: Campaign = Depends(pending_turn)) -> dict:
    """Drop an incomplete turn: remove the player's message and close the pending run."""
    st = request.app.state
    async with st.locks[campaign.id]:
        snap = await st.graph.aget_state(thread(campaign.id))
        msgs = snap.values.get("messages") or []
        player = next((m for m in reversed(msgs) if m.type == "human"), None)
        update = {"messages": [RemoveMessage(id=player.id)]} if player else {}
        await st.graph.aupdate_state(thread(campaign.id), update, as_node="narrate")
    return {"ok": True}
