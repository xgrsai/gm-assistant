"""Pure parts and control flow of `training/merge_and_quantize.py` (Colab CPU: merge the LoRA, quantize base and FT).

The script is loaded by path (it lives outside the backend package). No torch, transformers, safetensors, llama.cpp or
network: tensors are plain lists of floats, subprocesses and the weight reader are replaced by fakes. What really runs
only on Colab (the model download, the PEFT merge, the llama.cpp build and conversion) is not covered here.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sft import gates
from sft.gates import GateError

SCRIPT = Path(__file__).resolve().parents[3] / "training" / "merge_and_quantize.py"
if not SCRIPT.exists():
    pytest.skip("training/merge_and_quantize.py is not part of this checkout", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("merge_and_quantize", SCRIPT)
mq = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mq
_spec.loader.exec_module(mq)

ADAPTER_CONFIG = {
    "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 16, "lora_alpha": 16, "lora_dropout": 0, "bias": "none",
    "target_modules": ["v_proj", "gate_proj", "up_proj", "q_proj", "k_proj", "down_proj", "o_proj"],
    "base_model_name_or_path": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit", "peft_version": "0.20.0", "use_dora": False,
    "use_rslora": False, "rank_pattern": {}, "alpha_pattern": {}, "modules_to_save": None,
}  # fmt: skip
SHA = "a" * 64
FLOATS = st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)


def diff(base, merged):
    return mq.tensor_diff(base, merged)


def gguf_bytes(template: str | None, *, junk: bytes = b"") -> bytes:
    """A GGUF-like header: magic, version, counts, then string KV pairs (the layout find_gguf_string reads)."""

    def kv(key: str, value: str) -> bytes:
        k, v = key.encode(), value.encode()
        return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v

    body = kv("general.architecture", "qwen2") + junk
    if template is not None:
        body += kv("tokenizer.chat_template", template)
    return b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + body + b"\x00" * 64


# --- names, refs, small checks -------------------------------------------------------------------------------------


def test_quant_names_are_normalised_and_file_names_follow():
    assert mq.normalize_quant(" q4_k_m ") == "Q4_K_M"
    assert mq.gguf_file_name(mq.BASE_STEM, "Q4_K_M") == "qwen2.5-3b-instruct.Q4_K_M.gguf"
    assert mq.gguf_file_name(mq.FT_STEM, "Q4_K_M") == "keeper-qwen25-3b-sft-v1.Q4_K_M.gguf"
    assert mq.f16_file_name(mq.FT_STEM) == "keeper-qwen25-3b-sft-v1.f16.gguf"


@pytest.mark.parametrize("bad", ["", "Q4 K M", "../Q4", "Q4_K_M;rm", "f16", "BF16"])
def test_bad_quant_names_are_refused(bad):
    with pytest.raises(GateError, match="--quant"):
        mq.normalize_quant(bad)


def test_pinned_llama_cpp_ref_is_a_full_commit():
    assert mq.is_commit_sha(mq.LLAMA_CPP_REF)
    assert not mq.is_commit_sha("master") and not mq.is_commit_sha(mq.LLAMA_CPP_REF[:12])


def test_ref_matches_decides_only_for_hex_refs():
    ref = mq.LLAMA_CPP_REF
    assert mq.ref_matches(ref, ref) is True and mq.ref_matches(ref, ref[:10]) is True
    assert mq.ref_matches("b" * 40, ref) is False
    assert mq.ref_matches(ref, "master") is None and mq.ref_matches(None, ref) is None


def test_revision_comes_from_the_hf_cache_folder_name():
    sha = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    assert mq.revision_from_snapshot_path(f"/root/.cache/huggingface/hub/models--Qwen--X/snapshots/{sha}") == sha
    assert mq.revision_from_snapshot_path("/somewhere/else") is None


def test_missing_modules_and_gpu_warning_use_injected_lookups():
    assert mq.missing_modules(["a", "b", "c"], finder=lambda name: None if name == "b" else object()) == ["b"]
    assert mq.gpu_runtime_warning(which=lambda name: "/usr/bin/nvidia-smi") is not None
    assert mq.gpu_runtime_warning(which=lambda name: None) is None


def test_free_space_gate():
    mq.check_free_space(40 * 2**30)
    with pytest.raises(GateError, match="free"):
        mq.check_free_space(10 * 2**30)


# --- adapter ---------------------------------------------------------------------------------------------------


def test_adapter_config_summary_sorts_targets_and_computes_scaling():
    summary = mq.adapter_config_summary(ADAPTER_CONFIG)
    assert summary["target_modules"] == sorted(ADAPTER_CONFIG["target_modules"])
    assert summary["r"] == 16 and summary["scaling"] == 1.0 and summary["peft_version"] == "0.20.0"
    assert summary["base_model_name_or_path"].endswith("bnb-4bit")


@pytest.mark.parametrize(
    "patch, message",
    [({"peft_type": "IA3"}, "only LORA"), ({"target_modules": []}, "non-empty"),
     ({"target_modules": "all-linear"}, "non-empty"), ({"target_modules": ["q_proj", "embed_tokens"]}, "unsupported")],
)  # fmt: skip
def test_unsupported_adapter_configs_are_refused(patch, message):
    mq.validate_adapter_config(ADAPTER_CONFIG)
    with pytest.raises(GateError, match=message):
        mq.validate_adapter_config({**ADAPTER_CONFIG, **patch})


def test_adapter_matches_train_meta():
    assert mq.adapter_matches_train_meta({"adapter": {"sha256": SHA}}, SHA) is True
    assert mq.adapter_matches_train_meta({"adapter": {"sha256": "b" * 64}}, SHA) is False
    assert mq.adapter_matches_train_meta({"adapter": {}}, SHA) is None
    assert mq.adapter_matches_train_meta(None, SHA) is None


def test_inspect_adapter_hashes_the_files_and_reads_the_neighbouring_train_meta(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps(ADAPTER_CONFIG), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    sha = gates.sha256_file(adapter / "adapter_model.safetensors")
    (tmp_path / "train_meta.json").write_text(json.dumps({"adapter": {"sha256": sha}}), encoding="utf-8")

    info = mq.inspect_adapter(adapter)

    assert info["sha256"] == sha and info["size_bytes"] == 7
    assert info["train_meta"]["names_this_adapter"] is True and info["target_modules"][0] == "down_proj"
    assert info["config"]["lora_alpha"] == 16


def test_inspect_adapter_names_what_is_missing(tmp_path):
    with pytest.raises(GateError, match="adapter_config.json"):
        mq.inspect_adapter(tmp_path)


# --- LoRA load and weight checks (critique C4: a no-op merge must fail) --------------------------------------------


def test_expected_lora_modules_for_qwen25_3b():
    assert mq.expected_lora_modules(36, ADAPTER_CONFIG["target_modules"]) == 252


def test_probe_tensors_cover_an_early_and_a_late_target_and_a_control():
    probes = mq.probe_tensor_names(36, ADAPTER_CONFIG["target_modules"])
    assert probes["targets"] == ["model.layers.0.self_attn.q_proj.weight", "model.layers.35.mlp.down_proj.weight"]
    assert probes["control"] == "model.norm.weight"
    assert mq.probe_tensor_names(36, ["v_proj"])["targets"] == [
        "model.layers.0.self_attn.v_proj.weight", "model.layers.35.self_attn.v_proj.weight",
    ]  # fmt: skip
    with pytest.raises(GateError, match="unsupported"):
        mq.weight_name(0, "lm_head")


@given(st.integers(min_value=1, max_value=50_000_000))
def test_sample_step_bounds_the_sample_and_avoids_power_of_two_strides(count):
    step = mq.sample_step(count)
    assert step >= 1 and (step == 1 or step % 2 == 1)
    assert len(range(0, count, step)) <= 2 * mq.SAMPLE_ELEMENTS
    if count <= mq.SAMPLE_ELEMENTS:
        assert step == 1


def test_shard_for_reads_the_index_or_the_single_file():
    assert mq.shard_for(None, "anything") == "model.safetensors"
    assert mq.shard_for({"a": "model-00002-of-00002.safetensors"}, "a") == "model-00002-of-00002.safetensors"
    with pytest.raises(GateError, match="not in the checkpoint index"):
        mq.shard_for({"a": "x"}, "b")


def test_tensor_diff_counts_changed_and_nonfinite_values():
    same = diff([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert same["changed"] == 0 and same["max_abs_diff"] == 0 and same["nonfinite"] == 0
    one = diff([1.0, 2.0, 3.0], [1.0, 2.5, 3.0])
    assert one["changed"] == 1 and one["max_abs_diff"] == 0.5
    assert one["changed_fraction"] == pytest.approx(1 / 3, abs=1e-5)
    broken = diff([1.0, 2.0], [float("nan"), 2.0])
    assert broken["nonfinite"] == 1 and broken["changed"] == 0
    with pytest.raises(GateError, match="different or zero length"):
        diff([1.0], [1.0, 2.0])
    with pytest.raises(GateError):
        diff([], [])


@given(st.lists(FLOATS, min_size=1, max_size=40), st.data(), st.floats(min_value=1e-3, max_value=1))
def test_one_changed_weight_is_found_and_unchanged_weights_are_not(base, data, delta):
    at = data.draw(st.integers(min_value=0, max_value=len(base) - 1))
    merged = list(base)
    merged[at] = base[at] + delta
    result = diff(base, merged)
    assert result["changed"] == 1 and result["max_abs_diff"] == pytest.approx(delta)
    assert diff(base, list(base))["changed"] == 0


def checks(target_changes, control_changed=0, nonfinite=0, sampled=10):
    targets = {
        f"t{i}": {"sampled": sampled, "changed": changed, "nonfinite": nonfinite, "max_abs_diff": 0.01 * changed}
        for i, changed in enumerate(target_changes)
    }
    control = {"sampled": sampled, "changed": control_changed, "nonfinite": 0, "max_abs_diff": 0.5 * control_changed}
    return mq.merge_check(targets, "model.norm.weight", control)


def test_a_merge_that_changed_nothing_is_a_problem():
    problems, warnings = checks([0, 0])
    assert len(problems) == 1 and "NO-OP MERGE" in problems[0] and warnings == []


def test_a_merge_that_changed_a_target_passes_and_a_partial_one_warns():
    assert checks([5, 7]) == ([], [])
    problems, warnings = checks([5, 0])
    assert problems == [] and "t1" in warnings[0]


def test_a_handful_of_changed_values_is_not_a_merge():
    """Stray differences (rounding noise) in a few values are not a LoRA delta: at least 1% of the sample must move."""
    problems, _ = checks([5, 3], sampled=1000)  # 0.5% and 0.3%
    assert len(problems) == 1 and "NO-OP MERGE" in problems[0] and "1%" in problems[0]
    assert checks([10, 0], sampled=1000)[0] == []  # exactly 1% in one tensor is enough (the other one warns)


def test_control_tensor_and_nonfinite_values_are_problems():
    problems, _ = checks([5], control_changed=1)
    assert any("control tensor model.norm.weight differs" in p for p in problems)
    problems, _ = checks([5], nonfinite=2)
    assert any("non-finite" in p for p in problems)
    assert checks([])[0] == ["no LoRA target tensor was compared"]


def bf16(value: float) -> float:
    """The bfloat16 value below `value` (the top 16 bits of its float32 pattern), as a Python float."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0] & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def test_as_fp16_rounds_like_a_half_cast_and_saturates_to_inf():
    assert mq.as_fp16([1.296401023864746e-06]) == [1.3113021850585938e-06]  # bf16 value -> nearest fp16 subnormal
    assert mq.as_fp16([1e-9, 1.0, -2.5]) == [0.0, 1.0, -2.5]
    assert mq.as_fp16([70000.0, -70000.0]) == [float("inf"), float("-inf")]


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), max_size=30))
def test_as_fp16_is_idempotent(values):
    once = mq.as_fp16(values)
    assert mq.as_fp16(once) == once


def test_a_no_op_merge_of_a_bf16_base_is_not_mistaken_for_a_change(monkeypatch, tmp_path):
    """The base is bf16 on disk, the merge fp16: tiny weights differ after the cast alone (6-21 of 65k, measured)."""
    tiny = [bf16(1.3e-6 * (1 + i / 7)) for i in range(40)] + [bf16(0.0123 * (i + 1)) for i in range(40)]
    base, merged = tmp_path / "base", tmp_path / "merged"
    name_q, name_down = "model.layers.0.self_attn.q_proj.weight", "model.layers.35.mlp.down_proj.weight"
    weights = {
        (base, name_q): tiny, (merged, name_q): mq.as_fp16(tiny),  # what a no-op merge saves
        (base, name_down): tiny, (merged, name_down): mq.as_fp16(tiny),
        (base, "model.norm.weight"): [3.0], (merged, "model.norm.weight"): [3.0],
    }  # fmt: skip
    assert diff(tiny, mq.as_fp16(tiny))["changed"] > 0  # compared as stored, the cast alone looks like a change
    monkeypatch.setattr(mq, "read_tensor_sample", lambda folder, name: weights[(folder, name)])

    result = mq.compare_with_base(base, merged, 36, ADAPTER_CONFIG["target_modules"])

    assert all(d["changed"] == 0 for d in result["targets"].values())
    assert len(result["problems"]) == 1 and "NO-OP MERGE" in result["problems"][0]
    assert result["compared_as"] == "float16"


def test_tensor_name_sets_must_be_equal():
    assert mq.tensor_set_problems({"a", "b"}, ["b", "a"]) == []
    extra = mq.tensor_set_problems({"a"}, {"a", "lm_head.weight"})
    assert len(extra) == 1 and "only in the merged" in extra[0] and "lm_head.weight" in extra[0]
    missing = mq.tensor_set_problems({"a", "b"}, {"a"})
    assert len(missing) == 1 and "missing in the merged" in missing[0]


def test_tensor_names_come_from_the_shard_index(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}}), encoding="utf-8"
    )
    assert mq.checkpoint_weight_map(tmp_path) == {"a": "s1.safetensors", "b": "s2.safetensors"}
    assert mq.checkpoint_tensor_names(tmp_path) == {"a", "b"}
    assert mq.checkpoint_weight_map(tmp_path / "elsewhere") is None


@given(st.lists(FLOATS, min_size=1, max_size=30), st.floats(min_value=1e-3, max_value=1))
def test_merge_check_fails_exactly_when_the_weights_did_not_move(base, delta):
    moved = [value + delta for value in base]
    ok, _ = mq.merge_check({"w": diff(base, moved)}, "n", diff([1.0], [1.0]))
    noop, _ = mq.merge_check({"w": diff(base, list(base))}, "n", diff([1.0], [1.0]))
    assert ok == [] and noop and "NO-OP" in noop[0]


def test_compare_with_base_reads_both_folders_and_applies_the_checks(monkeypatch, tmp_path):
    base, merged = tmp_path / "base", tmp_path / "merged"
    weights = {
        (base, "model.layers.0.self_attn.q_proj.weight"): [0.1, 0.2, 0.3],
        (merged, "model.layers.0.self_attn.q_proj.weight"): [0.1, 0.25, 0.3],
        (base, "model.layers.35.mlp.down_proj.weight"): [1.0, 1.0],
        (merged, "model.layers.35.mlp.down_proj.weight"): [1.0, 1.0],
        (base, "model.norm.weight"): [3.0],
        (merged, "model.norm.weight"): [3.0],
    }
    monkeypatch.setattr(mq, "read_tensor_sample", lambda folder, name: weights[(folder, name)])

    result = mq.compare_with_base(base, merged, 36, ADAPTER_CONFIG["target_modules"])

    assert result["problems"] == [] and len(result["warnings"]) == 1  # the late tensor did not move
    assert result["targets"]["model.layers.0.self_attn.q_proj.weight"]["changed"] == 1
    assert result["control"]["model.norm.weight"]["changed"] == 0

    weights[(merged, "model.layers.0.self_attn.q_proj.weight")] = [0.1, 0.2, 0.3]  # now nothing moved at all
    assert "NO-OP MERGE" in mq.compare_with_base(base, merged, 36, ["q_proj", "down_proj"])["problems"][0]


def test_lora_load_problems():
    ok = {"expected_modules": 252, "found_modules": 252, "file_nonzero_b": 252, "model_nonzero_b": 252}
    assert mq.lora_load_problems(**ok) == []
    assert "attached 100" in mq.lora_load_problems(**{**ok, "found_modules": 100})[0]
    assert "untrained" in mq.lora_load_problems(**{**ok, "file_nonzero_b": 0, "model_nonzero_b": 0})[0]
    assert "not (all) loaded" in mq.lora_load_problems(**{**ok, "model_nonzero_b": 0})[0]


def test_from_pretrained_keyword_sets_start_with_the_documented_one():
    sets = mq.load_kwarg_candidates("fp16")
    assert sets[0] == {"dtype": "fp16", "low_cpu_mem_usage": True}
    assert {"torch_dtype": "fp16", "low_cpu_mem_usage": True} in sets


# --- config / tokenizer files taken over from the base --------------------------------------------------------------


def test_only_config_tokenizer_and_template_files_are_adopted_from_the_base():
    names = [
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "README.md",
    ]  # fmt: skip
    assert mq.base_files_to_copy(names) == [
        "chat_template.jinja", "config.json", "generation_config.json", "merges.txt", "tokenizer.json",
        "tokenizer_config.json", "vocab.json",
    ]  # fmt: skip


def make_folders(tmp_path, base_template="BASE-TEMPLATE"):
    base, merged = tmp_path / "base", tmp_path / "merged"
    base.mkdir()
    merged.mkdir()
    (base / "config.json").write_text(
        json.dumps({"rope_theta": 1000000.0, "torch_dtype": "bfloat16"}), encoding="utf-8"
    )
    (base / "tokenizer.json").write_text('{"vocab": 1}', encoding="utf-8")
    (base / "tokenizer_config.json").write_text(json.dumps({"chat_template": base_template}), encoding="utf-8")
    (base / "model.safetensors.index.json").write_text('{"weight_map": {"base": "x"}}', encoding="utf-8")
    (merged / "config.json").write_text(json.dumps({"rope_parameters": {}, "dtype": "float16"}), encoding="utf-8")
    (merged / "model.safetensors.index.json").write_text('{"weight_map": {"merged": "y"}}', encoding="utf-8")
    return base, merged


def test_adopting_base_files_overwrites_config_but_keeps_the_merged_weights_index(tmp_path):
    base, merged = make_folders(tmp_path)

    adopted = mq.adopt_base_files(base, merged)

    assert (merged / "config.json").read_bytes() == (base / "config.json").read_bytes()
    assert (merged / "tokenizer.json").read_bytes() == (base / "tokenizer.json").read_bytes()
    assert json.loads((merged / "model.safetensors.index.json").read_text(encoding="utf-8")) == {
        "weight_map": {"merged": "y"}
    }
    assert set(adopted["files"]) == {"config.json", "tokenizer.json", "tokenizer_config.json"}
    assert adopted["files"]["config.json"] == gates.sha256_file(base / "config.json")
    assert any(line.startswith("rope_theta") for line in adopted["config_json_differences_before_copy"])


def test_chat_template_comes_from_the_jinja_file_or_tokenizer_config(tmp_path):
    base, merged = make_folders(tmp_path)
    mq.adopt_base_files(base, merged)
    assert mq.read_chat_template(merged) == "BASE-TEMPLATE"
    (merged / "chat_template.jinja").write_text("FROM-JINJA", encoding="utf-8")
    assert mq.read_chat_template(merged) == "FROM-JINJA"
    assert mq.read_chat_template(tmp_path) is None


def test_tokenizer_record_compares_with_the_adapter_dir_copy(tmp_path):
    base, merged = make_folders(tmp_path)
    mq.adopt_base_files(base, merged)
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    record = mq.tokenizer_record(merged, adapter, "Qwen/X")
    assert record["same_as_adapter_dir"] is None and record["chat_template_sha256"] == gates.sha256_text(
        "BASE-TEMPLATE"
    )

    (adapter / "tokenizer.json").write_text('{"vocab": 1}', encoding="utf-8")
    assert mq.tokenizer_record(merged, adapter, "Qwen/X")["same_as_adapter_dir"] is True
    (adapter / "tokenizer.json").write_text('{"vocab": 2}', encoding="utf-8")
    assert mq.tokenizer_record(merged, adapter, "Qwen/X")["same_as_adapter_dir"] is False


# --- GGUF header ---------------------------------------------------------------------------------------------------


def test_chat_template_is_read_from_a_gguf_header():
    template = "{% for m in messages %}<|im_start|>{{ m.role }}{% endfor %}"
    assert mq.find_gguf_string(gguf_bytes(template), "tokenizer.chat_template") == template
    assert mq.find_gguf_string(gguf_bytes(None), "tokenizer.chat_template") is None


def test_gguf_key_text_inside_a_value_or_a_cut_header_is_not_a_hit():
    text_only = b"tokenizer.chat_template appears here but not as a key"
    assert mq.find_gguf_string(gguf_bytes(None, junk=text_only), "tokenizer.chat_template") is None
    assert mq.find_gguf_string(gguf_bytes("x" * 100)[:-80], "tokenizer.chat_template") is None


# --- shell commands -------------------------------------------------------------------------------------------------


def test_llama_cpp_is_fetched_at_one_commit_and_built_for_cpu_only():
    llama, url, ref = Path("/c/llama.cpp"), "https://example/llama.cpp", mq.LLAMA_CPP_REF
    d = str(llama)
    assert mq.git_fetch_commands(llama, url, ref) == [
        ["git", "init", "--quiet", d],
        ["git", "-C", d, "remote", "add", "origin", url],
        ["git", "-C", d, "fetch", "--quiet", "--depth", "1", "origin", ref],
        ["git", "-C", d, "checkout", "--quiet", "--detach", "FETCH_HEAD"],
    ]
    configure = mq.cmake_configure_command(llama)
    assert "-DCMAKE_BUILD_TYPE=Release" in configure and "-DGGML_CUDA=OFF" in configure
    assert configure[configure.index("-S") + 1] == d and configure[configure.index("-B") + 1] == str(llama / "build")
    build = mq.cmake_build_command(llama, 4)
    assert build[-4:] == ["--target", "llama-quantize", "-j", "4"]
    assert mq.quantize_binary(llama).as_posix().endswith("build/bin/llama-quantize")


def test_convert_and_quantize_commands():
    llama = Path("/c/llama.cpp")
    convert = mq.convert_command("python3", llama, Path("/w/merged"), Path("/w/m.f16.gguf.part"))
    assert convert[0] == "python3" and convert[1].endswith("convert_hf_to_gguf.py")
    assert convert[convert.index("--outtype") + 1] == "f16" and convert[2].endswith("merged")
    quantize = mq.quantize_command(Path("/c/q"), Path("/w/a.f16.gguf"), Path("/w/a.Q4_K_M.gguf"), "Q4_K_M", 2)
    assert quantize[-2:] == ["Q4_K_M", "2"]
    assert quantize[1].endswith("a.f16.gguf") and quantize[2].endswith("a.Q4_K_M.gguf")


# --- meta and messages ----------------------------------------------------------------------------------------------


def test_meta_has_the_documented_shape_and_survives_dump_json(tmp_path):
    names = ["code", "environment", "base", "adapter", "merge", "tokenizer", "llama_cpp", "quantization", "outputs"]
    sections = dict.fromkeys(names, {"k": 1})
    meta = mq.build_gguf_meta(
        started_at="2026-10-25T10:00:00+00:00", finished_at="2026-10-25T11:00:00+00:00", timings_s={"merge": 1.5},
        commands=["cmake --build x"], waived_mismatches={"llama-cpp": "x"}, **sections,
    )  # fmt: skip
    assert meta["schema"] == mq.META_SCHEMA and meta["commands"] == ["cmake --build x"]
    assert meta["waived_mismatches"] == {"llama-cpp": "x"}
    assert {"adapter", "merge", "llama_cpp", "quantization", "outputs", "base", "code", "environment"} <= set(meta)
    gates.dump_json(tmp_path / mq.META_NAME, meta)
    assert json.loads((tmp_path / mq.META_NAME).read_text(encoding="utf-8")) == meta


def test_next_steps_name_the_files_and_the_ollama_commands():
    text = mq.next_steps_text(Path("/drive/gguf"), ["a.Q4_K_M.gguf", "b.Q4_K_M.gguf", mq.META_NAME], "Q4_K_M")
    assert "adapters/keeper-qwen25-3b-sft-v1/" in text and "a.Q4_K_M.gguf" in text and mq.META_NAME in text
    assert "sft.ollama_models modelfiles --gguf-dir" in text
    assert "ollama create qwen2.5-3b-selfq4:v1 -f Modelfile.base" in text
    assert "ollama create keeper-qwen25-3b:sft-v1 -f Modelfile.ft" in text


def test_next_steps_are_one_command_per_line_and_return_to_the_backend_before_the_check():
    """Windows PowerShell 5.1 has no `&&`; each `cd` is relative to where the previous step left the shell."""
    text = mq.next_steps_text(Path("/drive/gguf"), ["a.gguf"], "Q4_K_M")
    lines = [line.strip() for line in text.splitlines()]
    assert "&&" not in text
    modelfiles = lines.index(
        "uv run python -m sft.ollama_models modelfiles --gguf-dir ../../adapters/keeper-qwen25-3b-sft-v1"
    )
    assert lines[modelfiles - 1] == "cd src/backend"
    assert lines[lines.index("ollama create qwen2.5-3b-selfq4:v1 -f Modelfile.base") - 1] == (
        "cd ../../adapters/keeper-qwen25-3b-sft-v1"
    )
    check = lines.index("uv run python -m sft.check_adapter")
    assert lines[check - 1] == "cd ../../src/backend" and check > lines.index(
        "ollama create keeper-qwen25-3b:sft-v1 -f Modelfile.ft"
    )


def test_defaults_follow_the_task():
    args = mq.parse_args(["--adapter-dir", "a", "--out", "o"])
    assert args.base == "Qwen/Qwen2.5-3B-Instruct" and args.quant == "Q4_K_M"
    assert args.llama_cpp_dir.as_posix().endswith("content/llama.cpp") and args.llama_cpp_ref == mq.LLAMA_CPP_REF
    assert args.keep_f16 is False and args.force is False and args.allow_mismatch == []
    # no --base-revision: the default base is pinned to the revision train_meta.json recorded for the v1 adapter
    assert args.base_revision is None and mq.is_commit_sha(mq.DEFAULT_BASE_REVISION)
    assert mq.DEFAULT_BASE_REVISION == "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    assert mq.pinned_base_revision(args.base, args.base_revision) == mq.DEFAULT_BASE_REVISION
    assert mq.pinned_base_revision(args.base, "main") == "main"
    assert mq.pinned_base_revision("Qwen/Qwen2.5-7B-Instruct", None) is None
    with pytest.raises(SystemExit):
        mq.parse_args(["--adapter-dir", "a", "--out", "o", "--allow-mismatch", "everything"])


def test_base_revision_is_resolved_to_a_commit_sha():
    """A pinned or explicit sha is used as it is; a branch / tag / main is resolved; an unresolvable name stops."""
    asked = []

    def resolver(repo, revision):
        asked.append((repo, revision))
        return None if revision == "nope" else "e" * 40

    assert mq.resolve_base_revision(mq.DEFAULT_BASE, None, resolver) == mq.DEFAULT_BASE_REVISION
    assert mq.resolve_base_revision(mq.DEFAULT_BASE, "f" * 40, resolver) == "f" * 40
    assert asked == []  # a commit needs no lookup
    assert mq.resolve_base_revision(mq.DEFAULT_BASE, "main", resolver) == "e" * 40
    assert mq.resolve_base_revision("other/model", None, resolver) == "e" * 40
    assert asked == [(mq.DEFAULT_BASE, "main"), ("other/model", None)]
    with pytest.raises(GateError, match="cannot resolve revision 'nope'"):
        mq.resolve_base_revision(mq.DEFAULT_BASE, "nope", resolver)
    assert mq.resolve_base_revision("other/model", None, lambda repo, revision: None) is None  # offline, nothing asked


def test_train_meta_consistency_problems():
    recorded = {"hf_base_for_gguf": mq.DEFAULT_BASE, "hf_base_revision": mq.DEFAULT_BASE_REVISION}
    assert mq.base_revision_problem(recorded, mq.DEFAULT_BASE, mq.DEFAULT_BASE_REVISION) is None
    assert mq.base_revision_problem(None, mq.DEFAULT_BASE, "x" * 40) is None
    assert mq.base_revision_problem(recorded, mq.DEFAULT_BASE, None) is None  # revision not known (yet)
    assert "aa8e72537993" in mq.base_revision_problem(recorded, mq.DEFAULT_BASE, "b" * 40)
    assert "Qwen/Qwen2.5-7B-Instruct" in mq.base_revision_problem(recorded, "Qwen/Qwen2.5-7B-Instruct", None)

    record = {"file": "train_meta.json", "names_this_adapter": False, "recorded_adapter_sha256": "c" * 64,
              "adapter_sha256": SHA}  # fmt: skip
    assert "cccccccccccc" in mq.adapter_problem(record) and "aaaaaaaaaaaa" in mq.adapter_problem(record)
    assert mq.adapter_problem({**record, "names_this_adapter": True}) is None
    assert mq.adapter_problem({**record, "names_this_adapter": None}) is None and mq.adapter_problem(None) is None

    assert mq.llama_cpp_problem("b" * 40, mq.LLAMA_CPP_REF) and "pinned ref" in mq.llama_cpp_problem(
        "b" * 40, mq.LLAMA_CPP_REF
    )
    assert mq.llama_cpp_problem(mq.LLAMA_CPP_REF, mq.LLAMA_CPP_REF) is None
    assert mq.llama_cpp_problem("b" * 40, "master") is None  # a branch name cannot be compared


def test_a_mismatch_stops_unless_that_kind_is_allowed(capsys):
    assert mq.enforce_match("adapter", None, []) is None
    with pytest.raises(GateError, match="--allow-mismatch adapter"):
        mq.enforce_match("adapter", "sha differs", [])
    with pytest.raises(GateError):
        mq.enforce_match("adapter", "sha differs", ["llama-cpp"])  # another kind does not waive it
    assert mq.enforce_match("adapter", "sha differs", ["adapter"]) == "sha differs"
    assert "waived" in capsys.readouterr().out


def args_for(tmp_path, *extra):
    paths = {"--adapter-dir": "adapter", "--out": "out", "--work-dir": "w", "--llama-cpp-dir": "llama.cpp"}
    return mq.parse_args([*(a for flag, name in paths.items() for a in (flag, str(tmp_path / name))), *extra])


@pytest.mark.parametrize(
    "extra, message", [(["--quant", "F16"], "--quant"), (["--ft-name", "qwen2.5-3b-instruct"], "differ")]
)
def test_main_refuses_bad_options_before_any_heavy_work(tmp_path, monkeypatch, extra, message):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    argv = ["--adapter-dir", str(tmp_path), "--out", str(tmp_path / "out"), *extra]
    with pytest.raises(GateError, match=message):
        mq.main(argv)
    assert not (tmp_path / "out").exists()


# --- control flow with fake subprocesses ---------------------------------------------------------------------------


def test_llama_cpp_is_fetched_and_built_once(monkeypatch, tmp_path):
    llama, ran = tmp_path / "llama.cpp", []

    def fake_run(cmd, commands, *, cwd=None):
        ran.append(cmd)
        commands.append(" ".join(cmd))
        if cmd[:2] == ["git", "init"]:
            llama.mkdir()
            (llama / "convert_hf_to_gguf.py").write_text("# convert", encoding="utf-8")
        if "--build" in cmd:
            binary = mq.quantize_binary(llama)
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"\x7fELF")

    monkeypatch.setattr(mq, "run", fake_run)
    monkeypatch.setattr(mq, "git_state", lambda path: {"commit": mq.LLAMA_CPP_REF, "dirty": False})
    commands: list[str] = []

    first = mq.ensure_llama_cpp(llama, mq.LLAMA_CPP_REF, commands)
    assert first["cloned_now"] and first["built_now"] and first["ref_matches_commit"] is True
    assert first["commit"] == mq.LLAMA_CPP_REF and len(commands) == 6

    again = mq.ensure_llama_cpp(llama, mq.LLAMA_CPP_REF, commands)
    assert not again["cloned_now"] and not again["built_now"] and len(commands) == 6

    other = tmp_path / "not-llama"
    other.mkdir()
    (other / "file").write_text("x", encoding="utf-8")
    with pytest.raises(GateError, match="not a llama.cpp checkout"):
        mq.ensure_llama_cpp(other, mq.LLAMA_CPP_REF, commands)


def test_an_existing_llama_cpp_at_another_commit_stops_the_run(monkeypatch, tmp_path):
    llama = tmp_path / "llama.cpp"
    binary = mq.quantize_binary(llama)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"ELF")
    (llama / "convert_hf_to_gguf.py").write_text("# convert", encoding="utf-8")
    monkeypatch.setattr(mq, "git_state", lambda path: {"commit": "b" * 40, "dirty": False})
    monkeypatch.setattr(mq, "run", lambda *a, **k: pytest.fail("nothing to fetch or build"))

    with pytest.raises(GateError, match="--allow-mismatch llama-cpp"):
        mq.ensure_llama_cpp(llama, mq.LLAMA_CPP_REF, [])

    waived = mq.ensure_llama_cpp(llama, mq.LLAMA_CPP_REF, [], ["llama-cpp"])
    assert waived["ref_matches_commit"] is False and "pinned ref" in waived["mismatch_waived"]
    assert mq.ensure_llama_cpp(llama, "master", [])["mismatch_waived"] is None  # a branch name is not comparable


@pytest.mark.parametrize("keep_f16", [False, True])
def test_gguf_stage_converts_quantizes_publishes_and_drops_the_f16(monkeypatch, tmp_path, keep_f16):
    args = args_for(tmp_path, *(["--keep-f16"] if keep_f16 else []))
    args.work_dir.mkdir()
    args.out.mkdir()
    folder = tmp_path / "merged"
    folder.mkdir()
    ran = []

    def fake_run(cmd, commands, *, cwd=None):
        ran.append(cmd)
        commands.append(" ".join(cmd))
        if cmd[1].endswith("convert_hf_to_gguf.py"):
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"f16" * 1000)
        else:  # llama-quantize <in> <out> <type> <threads>
            Path(cmd[2]).write_bytes(gguf_bytes("TEMPLATE"))

    monkeypatch.setattr(mq, "run", fake_run)
    commands: list[str] = []

    result = mq.gguf_stage(args, "ft", mq.FT_STEM, folder, Path("/q/llama-quantize"), commands)

    final = args.out / "keeper-qwen25-3b-sft-v1.Q4_K_M.gguf"
    assert final.exists() and not list(args.out.glob("*.part")) and not list(args.work_dir.glob("*.part"))
    assert result["sha256"] == gates.sha256_file(final) and result["size_bytes"] == final.stat().st_size
    assert result["f16"]["size_bytes"] == 3000 and result["source"] == "merged folder"
    assert result["chat_template_in_gguf"] and result["chat_template_sha256"] == gates.sha256_text("TEMPLATE")
    assert (args.work_dir / "keeper-qwen25-3b-sft-v1.f16.gguf").exists() is keep_f16
    assert [c[1].endswith("convert_hf_to_gguf.py") for c in ran] == [True, False] and len(commands) == 2

    ran.clear()  # a second call finds the quantized file in the work dir and converts nothing
    mq.gguf_stage(args, "ft", mq.FT_STEM, folder, Path("/q/llama-quantize"), commands)
    assert ran == []


def test_merge_stage_reuses_only_the_merge_it_was_made_from(tmp_path):
    args = args_for(tmp_path)
    merged = args.work_dir / "merged"
    merged.mkdir(parents=True)
    key = {"adapter_sha256": SHA, "base_id": mq.DEFAULT_BASE, "base_revision": "r1", "dtype": "float16"}
    gates.dump_json(merged / mq.MERGE_INFO_NAME, {**key, "device": "cpu"})

    assert mq.merge_stage(args, {"sha256": SHA}, tmp_path, "r1")["device"] == "cpu"
    with pytest.raises(GateError, match="another adapter/base"):
        mq.merge_stage(args, {"sha256": "b" * 64}, tmp_path, "r1")
    with pytest.raises(GateError, match="another adapter/base"):
        mq.merge_stage(args, {"sha256": SHA}, tmp_path, "r2")


def run_main(tmp_path, monkeypatch, *, train_meta=None, extra=(), calls=None, llama_commit=None, old_outputs=None):
    """The whole flow with fake stages; returns the output folder. `train_meta` is written next to the adapter dir.

    `calls` collects the names of the long stages that ran; `llama_commit` makes an existing llama.cpp checkout at that
    commit; `old_outputs` ({file name: bytes}) are outputs of an earlier run that already sit in the output folder.
    """
    calls = [] if calls is None else calls
    adapter = tmp_path / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text(json.dumps(ADAPTER_CONFIG), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    if train_meta is not None:
        (tmp_path / "train_meta.json").write_text(json.dumps(train_meta), encoding="utf-8")
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    (base / "config.json").write_text(
        json.dumps({"num_hidden_layers": 36, "torch_dtype": "bfloat16"}), encoding="utf-8"
    )
    out = tmp_path / "out"
    for name, data in (old_outputs or {}).items():
        out.mkdir(exist_ok=True)
        (out / name).write_bytes(data)
    if llama_commit:
        llama_dir = tmp_path / "llama.cpp"
        llama_dir.mkdir()
        (llama_dir / "convert_hf_to_gguf.py").write_text("# convert", encoding="utf-8")
        monkeypatch.setattr(mq, "git_state", lambda path: {"commit": llama_commit, "dirty": False})
        extra = ["--llama-cpp-dir", str(llama_dir), *extra]

    def fake_run(cmd, commands, *, cwd=None):
        commands.append(" ".join(cmd))
        if cmd[1].endswith("convert_hf_to_gguf.py"):
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"f16" * 100)
        else:
            Path(cmd[2]).write_bytes(gguf_bytes("TEMPLATE"))

    def stage(name, result):
        def fake(*args):
            calls.append(name)
            return result(*args) if callable(result) else result

        return fake

    tokenizer = {"chat_template_sha256": gates.sha256_text("TEMPLATE")}
    llama = {"commit": mq.LLAMA_CPP_REF, "quantize_binary": "/q/llama-quantize", "mismatch_waived": None}
    merged = {"dtype": "float16", "tokenizer": tokenizer, "device": "cpu"}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(mq, "missing_modules", lambda names: [])
    monkeypatch.setattr(mq, "check_free_space", lambda free: None)
    monkeypatch.setattr(mq, "ensure_base", stage("ensure_base", lambda base_id, revision, commands: (base, revision)))
    monkeypatch.setattr(mq, "merge_stage", stage("merge_stage", merged))
    monkeypatch.setattr(mq, "ensure_llama_cpp", stage("ensure_llama_cpp", llama))
    monkeypatch.setattr(mq, "run", fake_run)
    mq.main(["--adapter-dir", str(adapter), "--out", str(out), "--work-dir", str(tmp_path / "w"), *extra])
    return out


def trained(adapter_sha=None, base_id=mq.DEFAULT_BASE, revision=mq.DEFAULT_BASE_REVISION):
    """A train_meta.json as the training script writes it (only the keys this script reads)."""
    return {
        "adapter": {"sha256": adapter_sha or gates.sha256_text("weights")},
        "base": {"hf_base_for_gguf": base_id, "hf_base_revision": revision},
    }


def test_main_writes_the_meta_that_the_downstream_tools_read(monkeypatch, tmp_path, capsys):
    """The whole flow with fake stages: outputs, the meta paths other tools read (`outputs.<role>.file`, ...)."""
    out = run_main(tmp_path, monkeypatch, train_meta=trained(), extra=["--threads", "2"])

    meta = json.loads((out / mq.META_NAME).read_text(encoding="utf-8"))
    sha = gates.sha256_file(tmp_path / "adapter" / "adapter_model.safetensors")
    assert meta["adapter"]["sha256"] == sha and meta["llama_cpp"]["commit"] == mq.LLAMA_CPP_REF
    assert meta["adapter"]["train_meta"]["names_this_adapter"] is True and meta["waived_mismatches"] == {}
    assert meta["merge"]["dtype"] == "float16" and meta["quantization"]["type"] == "Q4_K_M"
    assert meta["base"]["revision"] == mq.DEFAULT_BASE_REVISION and meta["base"]["dtype_on_disk"] == "bfloat16"
    assert meta["tokenizer"] == {"chat_template_sha256": gates.sha256_text("TEMPLATE")}
    assert meta["environment"]["versions"]["python"]
    for role, name in (("base", "qwen2.5-3b-instruct.Q4_K_M.gguf"), ("ft", "keeper-qwen25-3b-sft-v1.Q4_K_M.gguf")):
        assert meta["outputs"][role]["file"] == name
        assert meta["outputs"][role]["sha256"] == gates.sha256_file(out / name)
    assert meta["outputs"]["same_chat_template"] is True
    assert len(meta["commands"]) == 4 and set(meta["timings_s"]) >= {"merge", "gguf_base", "gguf_ft", "total"}
    assert "ollama create keeper-qwen25-3b:sft-v1 -f Modelfile.ft" in capsys.readouterr().out

    with pytest.raises(GateError, match="already has"):  # the outputs of a finished run are never overwritten silently
        mq.main(["--adapter-dir", str(tmp_path / "adapter"), "--out", str(out), "--work-dir", str(tmp_path / "w")])


@pytest.mark.parametrize(
    "kind, train_meta",
    [
        ("adapter", trained(adapter_sha="c" * 64)),
        ("base-revision", trained(revision="d" * 40)),
        ("base-revision", trained(base_id="Qwen/Qwen2.5-7B-Instruct")),
    ],
)
def test_main_stops_on_a_train_meta_mismatch_unless_it_is_waived(monkeypatch, tmp_path, kind, train_meta):
    with pytest.raises(GateError, match=f"--allow-mismatch {kind}"):
        run_main(tmp_path, monkeypatch, train_meta=train_meta)
    assert not list((tmp_path / "out").glob("*.gguf"))

    out = run_main(tmp_path, monkeypatch, train_meta=train_meta, extra=["--allow-mismatch", kind])

    meta = json.loads((out / mq.META_NAME).read_text(encoding="utf-8"))
    assert set(meta["waived_mismatches"]) == {kind} and (out / "qwen2.5-3b-instruct.Q4_K_M.gguf").exists()


OLD_OUTPUTS = {"qwen2.5-3b-instruct.Q4_K_M.gguf": b"old base", "keeper-qwen25-3b-sft-v1.Q4_K_M.gguf": b"old ft"}


@pytest.mark.parametrize(
    "case, kwargs",
    [
        ("adapter", {"train_meta": trained(adapter_sha="c" * 64)}),
        ("base revision given as a commit", {"train_meta": trained(), "extra": ["--base-revision", "d" * 40]}),
        ("base revision given as a branch", {"train_meta": trained(), "extra": ["--base-revision", "main"]}),
        ("llama.cpp checkout", {"llama_commit": "b" * 40}),
    ],
)
def test_every_consistency_check_runs_before_forced_outputs_are_deleted_or_a_stage_starts(
    monkeypatch, tmp_path, case, kwargs
):
    """`--force` removes the previous GGUFs, but only once nothing is going to stop the run (ordering regression)."""
    monkeypatch.setattr(mq, "hf_revision", lambda repo, revision=None: "e" * 40)  # main -> another commit than trained
    calls: list[str] = []
    kwargs = {**kwargs, "extra": ["--force", *kwargs.get("extra", [])]}

    with pytest.raises(GateError, match="--allow-mismatch"):
        run_main(tmp_path, monkeypatch, calls=calls, old_outputs=OLD_OUTPUTS, **kwargs)

    out = tmp_path / "out"
    assert {p.name: p.read_bytes() for p in out.glob("*")} == OLD_OUTPUTS  # untouched, nothing new written
    assert calls == []  # no download, no merge, no llama.cpp build


def test_a_forced_run_replaces_the_previous_outputs_when_the_checks_pass(monkeypatch, tmp_path):
    calls: list[str] = []
    out = run_main(tmp_path, monkeypatch, train_meta=trained(), calls=calls, old_outputs=OLD_OUTPUTS, extra=["--force"])
    assert calls == ["ensure_base", "merge_stage", "ensure_llama_cpp"]
    assert (out / "qwen2.5-3b-instruct.Q4_K_M.gguf").read_bytes() != b"old base"


def test_a_waived_llama_cpp_mismatch_is_recorded_and_the_run_goes_on(monkeypatch, tmp_path):
    out = run_main(tmp_path, monkeypatch, llama_commit="b" * 40, extra=["--allow-mismatch", "llama-cpp"])
    meta = json.loads((out / mq.META_NAME).read_text(encoding="utf-8"))
    assert "pinned ref" in meta["waived_mismatches"]["llama-cpp"]


def test_existing_llama_cpp_problem_looks_only_at_an_existing_checkout(monkeypatch, tmp_path):
    llama = tmp_path / "llama.cpp"
    monkeypatch.setattr(mq, "git_state", lambda path: pytest.fail("no checkout, nothing to ask git"))
    assert mq.existing_llama_cpp_problem(llama, mq.LLAMA_CPP_REF) is None
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("# convert", encoding="utf-8")
    monkeypatch.setattr(mq, "git_state", lambda path: {"commit": "b" * 40, "dirty": False})
    assert "pinned ref" in mq.existing_llama_cpp_problem(llama, mq.LLAMA_CPP_REF)
    assert mq.existing_llama_cpp_problem(llama, "b" * 12) is None  # an abbreviated hex ref that matches
    assert mq.existing_llama_cpp_problem(llama, "master") is None


def test_old_torchao_is_reported_before_the_merge():
    assert mq.torchao_problem(None) is None
    assert mq.torchao_problem("0.16.0") is None and mq.torchao_problem("0.17.1+cu130") is None
    problem = mq.torchao_problem("0.10.0")
    assert problem and "pip uninstall -y torchao" in problem
