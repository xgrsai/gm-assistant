"""narrate_messages is the single source of the narrator's input: the node and SFT tooling share it."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from keeper.deps import Deps
from keeper.experiments import ExperimentRegistry
from keeper.nodes import Nodes, assert_ends_with_human, narrate_messages
from keeper.prompts import PromptSet
from keeper.runlog import RunLog
from keeper.scenarios import ScenarioRepo
from keeper.spec import ExperimentSpec

from .fakes import FakeFactory

REPO_ROOT = Path(__file__).resolve().parents[3]

HITS = [{"marker": 1, "source_type": "rules", "source": "rules.md", "section": "Sanity", "text": "Lose SAN."}]
ROLLS = [{"kind": "skill", "target": "Spot Hidden", "outcome": "success"}]


def make_spec(use_rag: bool, **kw) -> ExperimentSpec:
    return ExperimentSpec(arm="rag" if use_rag else "prompt", generator="gen", aux="aux", use_rag=use_rag,
                          prompt_version="keeper-v2", seed=5, **kw)


def make_state(history_turns: int = 1) -> dict:
    pack = ScenarioRepo(REPO_ROOT / "data" / "scenarios").load("demo-haunt")
    messages = [AIMessage(content="Ви стоїте перед будинком.", id="opening")]
    for i in range(history_turns):
        messages += [HumanMessage(content=f"Я оглядаю кімнату {i}."), AIMessage(content=f"Ви бачите стіл {i}.")]
    messages.append(HumanMessage(content="Що я бачу під столом?"))
    return {
        "scenario": pack.scenario_state("uk"),
        "investigator": pack.investigator("uk", None),
        "messages": messages,
        "adjudication": {"kind": "in_fiction", "needs_roll": True, "roll_type": "skill", "target": "Spot Hidden"},
        "turn_rolls": ROLLS,
        "errors": [],
        "rules_hits": HITS,
    }


def make_deps(fake: FakeFactory, tmp_path, spec: ExperimentSpec) -> Deps:
    return Deps(llms=fake, prompts=PromptSet(), scenarios=ScenarioRepo(REPO_ROOT / "data" / "scenarios"),
                experiments=ExperimentRegistry({spec.arm: spec}, spec.arm),
                runlog=RunLog(tmp_path / "runlog", enabled=False))


@pytest.mark.parametrize("use_rag", [False, True])
async def test_narrate_passes_exactly_narrate_messages(use_rag, tmp_path):
    spec = make_spec(use_rag)
    fake = FakeFactory(narrations=["Під столом лежить ключ [R1]."])
    deps = make_deps(fake, tmp_path, spec)
    state = make_state()
    runtime = SimpleNamespace(context=SimpleNamespace(spec=spec, campaign_id="c"), stream_writer=lambda _: None)

    await Nodes(deps).narrate(state, runtime)

    expected = narrate_messages(deps.prompts, deps.llms.profile(spec.generator), spec, state)
    assert fake.chat_inputs == [expected]
    assert isinstance(expected[0], SystemMessage)
    assert expected[1:] == state["messages"][1:]  # start_on="human" drops the opening narration
    assert ("KNOWLEDGE BASE" in expected[0].text) is use_rag
    assert ("Lose SAN." in expected[0].text) is use_rag


def test_narrate_messages_trims_history_and_ends_with_player_turn():
    spec = make_spec(False, history_max_tokens=1)  # budget floors at 2000 tokens
    state = make_state(history_turns=400)
    profile = FakeFactory().profile("gen")

    msgs = narrate_messages(PromptSet(), profile, spec, state)

    assert isinstance(msgs[0], SystemMessage)
    assert 2 < len(msgs) < len(state["messages"])
    assert msgs[1].type == "human"  # trimmed history starts on a player turn
    assert msgs[-1] is state["messages"][-1]
    assert_ends_with_human(msgs)


def test_narrate_messages_rejects_history_not_ending_with_player_turn():
    state = make_state()
    state["messages"] = state["messages"][:-1]
    with pytest.raises(ValueError):
        narrate_messages(PromptSet(), FakeFactory().profile("gen"), make_spec(False), state)
