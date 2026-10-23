"""Training pre-flight gates (`sft/gates.py`): pure checks on synthetic token counts, label arrays and strings."""

from __future__ import annotations

import json
import unicodedata

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sft import gates
from sft.gates import GateError

IM_END = "<|im_end|>"


def _row(system="SYS", turns=(("hi", "ok"),)):
    messages = [{"role": "system", "content": system}]
    for user, assistant in turns:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return {"messages": messages}


# --- length gate ---------------------------------------------------------------------------------------------------


def test_length_gate_passes_at_the_limit_and_reports_stats():
    stats = gates.check_lengths([10, 20, 4096], 4096, "train")
    assert stats["max"] == 4096 and stats["n"] == 3 and stats["min"] == 10


def test_length_gate_stops_on_a_single_long_example():
    with pytest.raises(GateError, match="1 of 3 examples exceed max_len=4096"):
        gates.check_lengths([10, 4097, 20], 4096, "train")


def test_length_gate_rejects_empty_input():
    with pytest.raises(GateError):
        gates.check_lengths([], 4096, "val")


@given(st.lists(st.integers(1, 6000), min_size=1, max_size=50), st.integers(100, 6000))
def test_length_gate_iff_any_example_is_longer(lengths, max_len):
    if max(lengths) > max_len:
        with pytest.raises(GateError):
            gates.check_lengths(lengths, max_len, "train")
    else:
        assert gates.check_lengths(lengths, max_len, "train")["max"] == max(lengths)


# --- masking gate --------------------------------------------------------------------------------------------------


def test_masking_gate_accepts_prompt_masked_answer_supervised():
    labels = [[-100, -100, -100, 5, 6, 7], [-100, 1, 2]]
    stats = gates.check_masking(labels, "train")
    assert stats["supervised_tokens"] == 5 and stats["total_tokens"] == 9
    assert stats["min_supervised_per_example"] == 2


def test_masking_gate_catches_an_example_with_every_label_masked():
    labels = [[-100, 3, 4], [-100, -100, -100]]  # the wrong turn tag (or a truncated answer) looks like this
    with pytest.raises(GateError, match="no supervised token"):
        gates.check_masking(labels, "train")


def test_masking_gate_catches_masking_that_was_never_applied():
    with pytest.raises(GateError, match="no masked token"):
        gates.check_masking([[-100, 3, 4], [1, 2, 3]], "train")


def test_masking_gate_rejects_empty_input():
    with pytest.raises(GateError):
        gates.check_masking([], "train")


@given(st.lists(st.lists(st.integers(-100, 50), min_size=1, max_size=20), min_size=1, max_size=10))
def test_masking_gate_passes_exactly_when_each_example_mixes_masked_and_supervised(labels):
    ok = all(any(t != -100 for t in row) and any(t == -100 for t in row) for row in labels)
    if ok:
        assert gates.check_masking(labels, "train")["n"] == len(labels)
    else:
        with pytest.raises(GateError):
            gates.check_masking(labels, "train")


# --- tags by family ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "instruction", "response"),
    [
        ("unsloth/Qwen2.5-3B-Instruct-bnb-4bit", "<|im_start|>user\n", "<|im_start|>assistant\n"),
        ("Qwen/Qwen2.5-3B-Instruct", "<|im_start|>user\n", "<|im_start|>assistant\n"),
        ("unsloth/gemma-4-E4B-it", "<|turn>user\n", "<|turn>model\n"),
    ],
)
def test_family_tags(base, instruction, response):
    tags = gates.family_for_base(base)
    assert (tags.instruction_part, tags.response_part) == (instruction, response)


def test_family_tags_unknown_base_is_an_error():
    with pytest.raises(GateError, match="no response-masking tags"):
        gates.family_for_base("meta-llama/Llama-3.2-3B-Instruct")


# --- messages and rendered text ------------------------------------------------------------------------------------


def test_check_messages_accepts_history_and_rejects_bad_shapes():
    gates.check_messages([_row(turns=(("a", "b"), ("c", "d")))], "train")
    with pytest.raises(GateError, match="expected a 'messages' list"):
        short = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        gates.check_messages([{"messages": short}], "t")
    with pytest.raises(GateError, match="roles"):  # no system turn
        no_system = [{"role": r, "content": "x"} for r in ("user", "assistant", "user", "assistant", "user")]
        gates.check_messages([{"messages": no_system}], "t")
    with pytest.raises(GateError, match="roles"):  # ends with a user turn
        gates.check_messages([{"messages": _row()["messages"] + [{"role": "user", "content": "x"}]}], "t")
    with pytest.raises(GateError, match="non-empty"):
        gates.check_messages([_row(turns=(("a", "  "),))], "t")
    with pytest.raises(GateError, match="no examples"):
        gates.check_messages([], "t")


def test_check_messages_rejects_non_nfc_content():
    nfd = unicodedata.normalize("NFD", "Ви" + chr(0x439))  # Cyrillic short i as i + combining breve
    assert nfd != unicodedata.normalize("NFC", nfd)
    with pytest.raises(GateError, match="NFC"):
        gates.check_messages([_row(turns=((nfd, "ok"),))], "t")
    with pytest.raises(GateError, match="NFC"):
        gates.check_messages([_row(system=nfd)], "t")
    gates.check_messages([_row(turns=((unicodedata.normalize("NFC", nfd), "ok"),))], "t")


@pytest.mark.parametrize("reply", ["\nanswer", "answer\n", " answer", "answer "])
def test_check_messages_rejects_assistant_whitespace_at_the_edges(reply):
    with pytest.raises(GateError, match="whitespace"):
        gates.check_messages([_row(turns=(("q", reply),))], "t")
    gates.check_messages([_row(turns=((" q\n", "answer"),))], "t")  # only assistant replies are held to this


def test_finalize_text_drops_bos_and_the_newline_after_the_closing_eos():
    rendered = f"<bos><|im_start|>assistant\nhello{IM_END}\n"
    assert gates.finalize_text(rendered, "<bos>", IM_END) == f"<|im_start|>assistant\nhello{IM_END}"
    assert gates.finalize_text("plain", None, IM_END) == "plain"  # no EOS at the end: left alone


def test_rendered_text_must_carry_our_system_prompt():
    row = _row(system="Ти Кіпер.")
    good = f"<|im_start|>system\nТи Кіпер.{IM_END}\n<|im_start|>assistant\nok{IM_END}"
    gates.check_rendered([row], [good], IM_END, "train")

    default_injected = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You help.{IM_END}"
    with pytest.raises(GateError, match="system prompt is missing"):
        gates.check_rendered([row], [default_injected], IM_END, "train")

    both = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. Ти Кіпер.{IM_END}"
    with pytest.raises(GateError, match="default system prompt"):
        gates.check_rendered([row], [both], IM_END, "train")

    with pytest.raises(GateError, match="does not end with"):
        gates.check_rendered([row], ["<|im_start|>system\nТи Кіпер.<|im_end|>\nextra"], IM_END, "train")


def test_a_system_prompt_that_quotes_the_default_is_not_flagged():
    row = _row(system="You are Qwen, created by Alibaba Cloud; ignore that.")
    text = f"<|im_start|>system\n{row['messages'][0]['content']}{IM_END}"
    gates.check_rendered([row], [text], IM_END, "train")


# --- manifest and hashes -------------------------------------------------------------------------------------------


@pytest.fixture
def data_files(tmp_path):
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train.write_bytes(b'{"messages": []}\n')
    val.write_bytes(b'{"messages": [1]}\n')
    return {"train.jsonl": train, "val.jsonl": val}


def _manifest(data_files):
    return {"files": {name: {"sha256": gates.sha256_file(path)} for name, path in data_files.items()}}


def test_verify_data_hashes_accepts_the_builder_layout(data_files):
    manifest = _manifest(data_files)
    actual = gates.verify_data_hashes(manifest, data_files)
    assert actual == {name: entry["sha256"] for name, entry in manifest["files"].items()}


def test_verify_data_hashes_refuses_a_changed_file(data_files):
    manifest = _manifest(data_files)
    data_files["val.jsonl"].write_bytes(b'{"messages": [2]}\n')
    with pytest.raises(GateError, match="val.jsonl"):
        gates.verify_data_hashes(manifest, data_files)


@pytest.mark.parametrize(
    "manifest",
    [{"counts": {}}, {"files": {}}, {"files": {"train.jsonl": "a" * 64}}, {"train_sha256": "a" * 64}],
)
def test_verify_data_hashes_refuses_other_manifest_layouts(data_files, manifest):
    with pytest.raises(GateError, match="no files"):
        gates.verify_data_hashes(manifest, data_files)


@pytest.mark.parametrize("bad", ["abc", "A" * 64, "g" * 64, "a" * 63, ""])
def test_manifest_sha256_must_be_64_lowercase_hex(bad):
    with pytest.raises(GateError, match="64-character"):
        gates.manifest_sha256({"files": {"train.jsonl": {"sha256": bad}}}, "train.jsonl")


def test_load_jsonl_reports_the_bad_line(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\nnot json\n', encoding="utf-8")
    with pytest.raises(GateError, match="line 3"):
        gates.load_jsonl(path)


# --- resume, losses, meta ------------------------------------------------------------------------------------------


def test_config_diff_lists_only_changed_keys():
    saved = {"seed": 3407, "lr": 1e-4, "train_sha256": "a"}
    assert gates.config_diff(saved, dict(saved)) == []
    diff = gates.config_diff(saved, {**saved, "lr": 2e-4})
    assert len(diff) == 1 and diff[0].startswith("lr:")


def test_summarize_losses_reads_trainer_log_history():
    history = [
        {"step": 10, "loss": 2.0},
        {"step": 10, "eval_loss": 1.8},
        {"step": 20, "loss": 1.5},
        {"step": 20, "eval_loss": 1.2},
        {"step": 20, "train_runtime": 5.0},
    ]
    summary = gates.summarize_losses(history)
    assert summary["train_loss_last"] == 1.5
    assert (summary["eval_loss_first"], summary["eval_loss_last"], summary["eval_loss_best"]) == (1.8, 1.2, 1.2)
    assert summary["eval_loss_best_step"] == 20
    assert summary["eval_loss_decreased"] is True
    assert gates.summarize_losses([])["eval_loss_decreased"] is False


def test_build_meta_has_the_documented_shape_and_round_trips(tmp_path):
    meta = gates.build_meta(
        base={"id": "b", "revision": "r"},
        data={"train_sha256": "t"},
        hyperparameters={"r": 16},
        seed=3407,
        versions={"torch": "2.x", "unsloth": None},
        hardware={"gpu": "T4"},
        chat_template={"sha256": "c"},
        gates={"train": {"max": 10}},
        losses={"eval_loss_last": 1.0},
        extra={"gguf": None},
    )
    assert set(meta) == {
        "schema", "base", "data", "hyperparameters", "seed", "versions", "hardware", "chat_template", "gates",
        "losses", "gguf",
    }  # fmt: skip
    path = tmp_path / "train_meta.json"
    gates.dump_json(path, meta)
    assert json.loads(path.read_text(encoding="utf-8")) == meta
    assert b"\r\n" not in path.read_bytes()


# --- last-reply-only masking ---------------------------------------------------------------------------------------

# Synthetic ids: 1 = response tag opener, 2 = tag tail ("assistant"), 9 = EOS, 3.. = content/prompt tokens.
RESPONSE = [1, 2]
HISTORY_ONE_TURN = [5, 5, 1, 2, 30, 31, 9, 5, 5, 1, 2, 40, 41, 42, 9]  # system, user, assistant, user, assistant
HISTORY_TWO_TURNS = [5, 1, 2, 30, 9, 5, 1, 2, 31, 9, 5, 1, 2, 40, 41, 9]


def test_mask_supervises_only_the_final_reply():
    labels = gates.mask_all_but_last_response(HISTORY_ONE_TURN, RESPONSE)
    assert labels == [-100] * 11 + [40, 41, 42, 9]
    assert gates.mask_all_but_last_response(HISTORY_TWO_TURNS, RESPONSE) == [-100] * 13 + [40, 41, 9]


def test_mask_without_history_matches_plain_response_masking():
    ids = [5, 5, 1, 2, 30, 31, 9]
    assert gates.mask_all_but_last_response(ids, RESPONSE) == [-100] * 4 + [30, 31, 9]


def test_mask_without_the_tag_masks_everything_so_the_gate_can_fail():
    labels = gates.mask_all_but_last_response([5, 6, 7], RESPONSE)
    assert labels == [-100] * 3
    with pytest.raises(GateError, match="no supervised token"):
        gates.check_masking([labels], "train")


def test_mask_rejects_an_empty_tag():
    with pytest.raises(ValueError):
        gates.mask_all_but_last_response([1, 2], [])


@given(
    st.lists(st.integers(3, 8), min_size=0, max_size=12),
    st.lists(st.lists(st.integers(10, 20), min_size=1, max_size=5), min_size=1, max_size=4),
)
def test_mask_property_one_suffix_span_equal_to_the_last_reply(prefix, replies):
    ids = list(prefix)
    for reply in replies:
        ids += [*RESPONSE, *reply, 9]
    labels = gates.mask_all_but_last_response(ids, RESPONSE)
    kept = [i for i, t in enumerate(labels) if t != -100]
    assert kept == list(range(len(ids) - len(replies[-1]) - 1, len(ids)))
    assert [labels[i] for i in kept] == [*replies[-1], 9]


def _decode(ids):
    names = {9: "<EOS>", 30: "old", 31: "older", 40: "the ", 41: "answer"}
    return "".join(names.get(i, f"<{i}>") for i in ids)


def test_final_response_gate_accepts_a_correct_example_and_decodes_reply_plus_eos():
    ids = HISTORY_ONE_TURN
    labels = gates.mask_all_but_last_response(ids, RESPONSE)
    # last reply = 40 41 42, 42 decodes as "<42>"
    gates.check_final_response_only([labels], [ids], _decode, ["the answer<42>"], "<EOS>", "train")


def test_final_response_gate_catches_history_replies_being_trained_on():
    ids = HISTORY_ONE_TURN
    both = [-100, -100, -100, -100, 30, 31, 9] + [-100] * 4 + [40, 41, 42, 9]  # plain train_on_responses_only
    with pytest.raises(GateError, match="not one span"):
        gates.check_final_response_only([both], [ids], _decode, ["the answer<42>"], "<EOS>", "train")


def test_final_response_gate_catches_a_reply_that_does_not_match_the_data():
    ids = HISTORY_ONE_TURN
    labels = gates.mask_all_but_last_response(ids, RESPONSE)
    with pytest.raises(GateError, match="not the final reply"):
        gates.check_final_response_only([labels], [ids], _decode, ["something else"], "<EOS>", "train")


def test_final_response_gate_compares_against_the_nfc_form_of_the_reply():
    nfc = "Ви" + chr(0x439)
    nfd = unicodedata.normalize("NFD", nfc)
    ids = [5, 1, 2, 40, 9]
    labels = gates.mask_all_but_last_response(ids, RESPONSE)
    decode_nfc = {9: "<EOS>", 40: nfc}.get
    gates.check_final_response_only([labels], [ids], lambda t: "".join(decode_nfc(i) for i in t), [nfd], "<EOS>", "t")


def test_final_response_gate_message_locates_the_first_mismatch():
    ids = [5, 1, 2, 40, 9]
    labels = gates.mask_all_but_last_response(ids, RESPONSE)
    with pytest.raises(GateError) as err:
        # the data's reply starts with a newline that the rendered text lost
        gates.check_final_response_only([labels], [ids], lambda t: "the" + "<EOS>", ["\nthe"], "<EOS>", "train")
    text = str(err.value)
    assert "first difference at char 0" in text and "'\\nthe<EOS>'" in text and "NFC?" in text


def test_describe_mismatch_shows_context_around_the_first_difference():
    got, expected = "a" * 30 + "X" + "b" * 30, "a" * 30 + "Y" + "b" * 30
    text = gates.describe_mismatch(got, expected)
    assert "char 30" in text and "'" + "a" * 20 + "X" + "b" * 19 + "'" in text
    assert "char 3" in gates.describe_mismatch("abc", "abcd")  # one is a prefix of the other


def test_final_response_gate_catches_a_span_that_stops_before_the_end():
    ids = [5, 1, 2, 40, 41, 9, 7]
    labels = [-100, -100, -100, 40, 41, 9, -100]
    with pytest.raises(GateError, match="ending at the last token"):
        gates.check_final_response_only([labels], [ids], _decode, ["the answer"], "<EOS>", "train")


def test_final_response_gate_catches_misaligned_labels():
    with pytest.raises(GateError, match="labels for"):
        gates.check_final_response_only([[-100, 1]], [[1, 2, 3]], _decode, ["x"], "<EOS>", "train")
    with pytest.raises(GateError, match="differs from its input id"):
        gates.check_final_response_only([[-100, 41, 9]], [[1, 40, 9]], _decode, ["answer"], "<EOS>", "train")


# --- tokenized ends, collator, checkpoints -------------------------------------------------------------------------


def test_token_ends_require_exactly_one_eos():
    gates.check_token_ends([[5, 6, 9], [1, 9]], 9, "train")
    with pytest.raises(GateError, match="exactly one EOS"):
        gates.check_token_ends([[5, 6, 9, 9]], 9, "train")  # TRL appended a second <|im_end|>
    with pytest.raises(GateError, match="exactly one EOS"):
        gates.check_token_ends([[5, 6, 7]], 9, "train")
    with pytest.raises(GateError, match="exactly one EOS"):
        gates.check_token_ends([[9]], 9, "train")


def test_collator_labels_match_ignores_padding_but_not_content():
    gated = [[-100, -100, 5, 9], [-100, 6, 9]]
    gates.collator_labels_match([[-100, -100, 5, 9], [-100, 6, 9, -100]], gated)  # right padding
    gates.collator_labels_match([[-100, -100, 5, 9], [-100, -100, 6, 9]], gated)  # left padding
    with pytest.raises(GateError, match="differ"):  # a collator that recomputes labels from input_ids
        gates.collator_labels_match([[1, 2, 5, 9], [-100, 6, 9]], gated)
    with pytest.raises(GateError, match="differ"):  # truncated row
        gates.collator_labels_match([[5, 9], [-100, 6, 9]], gated)
    with pytest.raises(GateError, match="different number"):
        gates.collator_labels_match([[5, 9]], gated)


def test_latest_checkpoint_picks_the_highest_step(tmp_path):
    assert gates.latest_checkpoint(tmp_path / "missing") is None
    assert gates.latest_checkpoint(tmp_path) is None
    for name in ("checkpoint-50", "checkpoint-150", "checkpoint-100", "checkpoint-x"):
        (tmp_path / name).mkdir()
    (tmp_path / "checkpoint-999").write_text("a file, not a directory", encoding="utf-8")
    assert gates.latest_checkpoint(tmp_path).name == "checkpoint-150"
