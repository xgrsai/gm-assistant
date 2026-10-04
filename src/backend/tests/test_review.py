"""Gold-data review page: the embedded data resolves against the corpus, and exported verdicts summarise."""

from __future__ import annotations

import json

import pytest

from evals.datasets import DATASET_DIR, EvalDatasets
from evals.review import FLAGS, PLACEHOLDER, page_data, render, review_units, summarize


@pytest.fixture(scope="module")
def ds():
    return EvalDatasets.load(DATASET_DIR)


def embedded(html: str) -> dict:
    payload = html.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    return json.loads(payload)


def test_page_data_resolves_against_corpus_and_scenario(ds):
    data = page_data(ds)
    units = data["units"]
    assert [len(units[k]) for k in ("rules_qa", "adjudication", "conflict_probes")] == [100, 40, 24]
    ids = {u["id"] for us in units.values() for u in us}
    assert set(FLAGS) <= ids and data["flags"] == FLAGS          # every drafting flag points at a real unit
    keys = {(s["source"], s["section"]) for s in data["sections"]}
    for u in units["rules_qa"]:
        assert all((g["source"], g["section"]) in keys for g in u["item"]["gold_sections"])
    for u in units["conflict_probes"]:
        g = u["item"]["gold_section"]
        assert (g["source"], g["section"]) in keys
    assert (data["when_to_roll"]["source"], data["when_to_roll"]["section"]) in keys
    targets = {t for u in units["adjudication"] for i in u["items"] for t in i["expected"]["allowed_targets"]}
    assert targets <= set(data["sheet"]["skills"]) | set(data["sheet"]["characteristics"])
    assert data["ambiguities"] and data["meta"]["hashes"] == ds.hashes


def test_render_keeps_markup_inside_the_json_block(ds):
    html = render({"meta": {}, "x": "</script><!-- <b>"})
    assert PLACEHOLDER not in html
    assert embedded(html)["x"] == "</script><!-- <b>"


def test_summary_counts_verdicts_and_flags_stale_ones(ds):
    units = review_units(ds)
    qa = units["rules_qa"]
    export = {
        "format": "gold-review/1", "dataset": "data/eval/coc7e", "reviewer": "author",
        "units": {
            qa[0]["id"]: {"verdict": "ok", "item_hash": qa[0]["hash"]},
            qa[1]["id"]: {"verdict": "wrong", "note": "fact 2 is not in the section", "fix": "drop fact 2",
                          "item_hash": qa[1]["hash"]},
            qa[2]["id"]: {"verdict": "ok", "item_hash": "edited-later"},
            "qa-999": {"verdict": "ok", "item_hash": "x"},
        },
    }
    text = summarize(export, units)
    assert "rules_qa: 100 units, ok 2, wording 0, ambiguous 0, wrong 1, unreviewed 97 (stale 1)" in text
    assert f"{qa[1]['id']} [wrong] note: fact 2 is not in the section | fix: drop fact 2" in text
    assert f"{qa[2]['id']} [ok, stale]" in text
    assert "adjudication: 40 units, ok 0, wording 0, ambiguous 0, wrong 0, unreviewed 40" in text
    assert "not in the current dataset: qa-999" in text


def test_unit_hash_changes_with_content(ds):
    before = review_units(ds)["rules_qa"][0]
    item = ds.rules_qa[0].model_copy(update={"question_en": "changed"})
    changed = review_units(ds.model_copy(update={"rules_qa": [item, *ds.rules_qa[1:]]}))["rules_qa"][0]
    assert before["id"] == changed["id"] and before["hash"] != changed["hash"]
