"""Adapter-effect check and model card (`sft/check_adapter.py`): a fake Ollama and a tiny fake model directory."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from evals import model_cards
from sft import check_adapter as ca
from sft import gates

OFFICIAL = "qwen2.5:3b"
BASE = "qwen2.5-3b-selfq4:v1"
FT = "keeper-qwen25-3b:sft-v1"
OFFICIAL_BLOB = "c" * 64
OTHER_BLOB = "b" * 64
TEMPLATE = "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}"
BS = chr(92)  # a Windows path separator, as in `ollama show --modelfile` output on Windows
BLOBS = "C:/Users/1/.ollama/models/blobs/sha256-"
# what /api/show reports for a Qwen2.5-3B GGUF (shortened): general.* differs by file, the rest is the architecture
MODEL_INFO = {
    "general.architecture": "qwen2", "general.file_type": 15, "general.basename": "Qwen2.5-3B-Instruct",
    "qwen2.block_count": 36, "qwen2.context_length": 32768, "qwen2.rope.freq_base": 1000000.0,
    "qwen2.attention.layer_norm_rms_epsilon": 1e-06, "tokenizer.ggml.eos_token_id": 151645,
}
DETAILS = {"family": "qwen2", "parameter_size": "3.1B", "quantization_level": "Q4_K_M"}


def digest_of(model: str) -> str:
    """A realistic digest (full sha256, as /api/tags reports it) for a tag."""
    return hashlib.sha256(model.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ fixtures


def reply_text(roll: bool, cite: bool, lang: str, tag: str) -> str:
    if lang == "en":
        return f"The corridor is silent and cold, {tag}.\n\nWhat do you do?"
    parts = [f"Ви бачите двері, {tag}. Вона тихо риплять на завісах."]
    if roll:
        parts.append("🎲 Пильність (Spot Hidden): 12 — успіх" + (" [R1]." if cite else "."))
    parts.append("Що ви робите?")
    return "\n\n".join(parts)


# (id, language, has_roll, cites)
CASES = [
    ("v-01", "uk", True, True),
    ("v-02", "uk", False, False),
    ("v-03", "en", False, False),
    ("v-04", "uk", True, False),
    ("v-05", "uk", True, True),
    ("v-06", "en", False, False),
]


def chat_row(system: str, user: str, reference: str) -> dict:
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": reference},
    ]}  # fmt: skip


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def write_safetensors(path: Path, names: list[str]) -> None:
    header = {n: {"dtype": "F16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]} for i, n in enumerate(names)}
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * (2 * len(names)))


def write_gguf(path: Path, tensors: int, payload: bytes = b"") -> None:
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, tensors, 7) + b"\0" * 32 + payload)


def gguf_meta(ft_sha: str, base_sha: str, adapter_sha: str, ft_file: str = ca.FT_GGUF) -> dict:
    """The shape training/merge_and_quantize.py writes (only the parts the card reads)."""
    return {
        "schema": 1, "kind": "merged_gguf",
        "llama_cpp": {"commit": "1234abcd" * 5, "ref": "71255fefc092b0344f2acddb891388814ae961be", "dirty": False},
        "base": {"id": "Qwen/Qwen2.5-3B-Instruct", "revision": "rev-1"},
        "adapter": {"sha256": adapter_sha, "file": "adapter_model.safetensors"},
        "merge": {"dtype": "float16", "adapter_sha256": adapter_sha},
        "environment": {"versions": {"peft": "0.20.0", "transformers": "5.0.0"}},
        "quantization": {"type": "Q4_K_M", "intermediate": "f16"},
        "outputs": {"base": {"file": ca.BASE_GGUF, "sha256": base_sha}, "ft": {"file": ft_file, "sha256": ft_sha},
                    "same_chat_template": True},
    }  # fmt: skip


@dataclass
class Env:
    root: Path
    adapter_dir: Path
    val: Path
    split: Path
    card: Path
    manifest: dict
    ft_sha: str
    base_sha: str
    adapter_sha: str

    @property
    def results(self) -> Path:
        return self.adapter_dir / "check_adapter.json"

    def argv(self, *extra: str) -> list[str]:
        return [
            "--adapter-dir", str(self.adapter_dir), "--val", str(self.val), "--split", str(self.split),
            "--card", str(self.card), "--ollama-url", "http://fake", "--n", "4", "--repeat", "2", *extra,
        ]  # fmt: skip


@pytest.fixture
def env(tmp_path: Path) -> Env:
    data = tmp_path / "data"
    data.mkdir()
    val_rows = [chat_row("SYSTEM", f"Гравець діє: {cid}", reply_text(roll, cite, lang, "ref"))
                for cid, lang, roll, cite in CASES]  # fmt: skip
    train_rows = [chat_row("SYSTEM", "Інший хід", reply_text(True, True, "uk", "train"))]
    write_jsonl(data / "val.jsonl", val_rows)
    write_jsonl(data / "train.jsonl", train_rows)
    split = tmp_path / "section_split.v1.json"
    split.write_text('{"seed": 1927}\n', encoding="utf-8", newline="\n")
    manifest = {
        "dataset": "coc7e-test", "git_sha": "d" * 40, "git_dirty": False, "prompt": {"version": "keeper-v2"},
        "files": {
            "train.jsonl": {"sha256": gates.sha256_file(data / "train.jsonl"), "examples": 1},
            "val.jsonl": {"sha256": gates.sha256_file(data / "val.jsonl"), "examples": len(CASES)},
        },
        "ids": {"train": ["t-01"], "val": [c[0] for c in CASES]},
        "split": {"sha256": gates.sha256_file(split), "seed": 1927, "corpus_chunk_ids_hash": "abc"},
        "counts": {"all": {"examples": 7}},
    }  # fmt: skip
    write_json(data / "manifest.json", manifest)

    adapter_dir = tmp_path / "adapters" / "keeper-test-sft-v1"
    names = [f"layer{i}.lora_{ab}.weight" for i in range(2) for ab in "AB"]
    write_safetensors(adapter_dir / "adapter" / "adapter_model.safetensors", names)
    write_json(adapter_dir / "adapter" / "adapter_config.json",
               {"peft_type": "LORA", "r": 16, "lora_alpha": 16, "target_modules": ["q_proj"], "other": 1})  # fmt: skip
    write_gguf(adapter_dir / ca.FT_GGUF, tensors=434, payload=b"merged-and-quantized")
    write_gguf(adapter_dir / ca.BASE_GGUF, tensors=434, payload=b"quantized-only")
    ft_sha, base_sha = (gates.sha256_file(adapter_dir / n) for n in (ca.FT_GGUF, ca.BASE_GGUF))
    adapter_sha = gates.sha256_file(adapter_dir / "adapter" / "adapter_model.safetensors")
    write_json(adapter_dir / ca.GGUF_META, gguf_meta(ft_sha, base_sha, adapter_sha))
    write_json(adapter_dir / "manifest.json", manifest)
    meta = {
        "base": {"id": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit", "revision": "r1"},
        "seed": 3407,
        "data": {"train_sha256": manifest["files"]["train.jsonl"]["sha256"],
                 "val_sha256": manifest["files"]["val.jsonl"]["sha256"], "manifest": manifest},
        "hyperparameters": {"r": 16, "alpha": 16, "lr": 1e-4, "epochs": 2},
        "versions": {"torch": "2.x", "trl": "1.13.0"},
        "hardware": {"gpu": "Tesla T4"},
        "chat_template": {"string": "TEMPLATE-STRING", "sha256": "e" * 64, "same_as_transformers": True,
                          "eos_token": "<|im_end|>", "first_rendered_example": "x" * 100},
        "gates": {"train": {"length": {"max": 10}}},
        "losses": {"train_loss_first": 2.468, "train_loss_last": 1.666, "eval_loss_first": 1.86,
                   "eval_loss_last": 1.711, "eval_loss_best": 1.711, "eval_loss_best_step": 140},
        "adapter": {"dir": "adapter", "selection": "last step", "saved_at_step": 140, "sha256": adapter_sha},
        "code": {"git": {"commit": "c" * 40, "dirty": False}},
        "gguf": None,
    }  # fmt: skip
    write_json(adapter_dir / "train_meta.json", meta)
    return Env(
        root=tmp_path, adapter_dir=adapter_dir, val=data / "val.jsonl", split=split,
        card=tmp_path / "cards" / "keeper-test-sft-v1.json", manifest=manifest, ft_sha=ft_sha, base_sha=base_sha,
        adapter_sha=adapter_sha,
    )  # fmt: skip


@pytest.fixture(autouse=True)
def fixed_git(monkeypatch):
    monkeypatch.setattr(ca, "git_state", lambda path: {"commit": "f" * 40, "dirty": False})


def user_of(messages: list[dict]) -> str:
    return messages[-1]["content"]


class FakeOllama:
    """Scripted Ollama: `base_fn(user_text, nth_call_for_that_prompt)` and `ft_fn(user_text, nth)` give the replies."""

    def __init__(self, env: Env, base_fn=None, ft_fn=None, official_modelfile=None, base_modelfile=None,
                 ft_modelfile=None, official_template=TEMPLATE, base_template=TEMPLATE, ft_template=TEMPLATE,
                 models=(OFFICIAL, BASE, FT), fail_at=None):
        self.fail_at = fail_at  # raise an OllamaError on this (1-based) chat call
        self.base_fn = base_fn or (lambda user, n: f"BASE {user}")
        self.ft_fn = ft_fn or (lambda user, n: f"FT {user}")
        self.calls: list[tuple[str, str, dict]] = []
        self.prompts: list[list[dict]] = []
        self.seen: dict[tuple[str, str], int] = {}
        self.models = models
        official_mf = f'# Modelfile generated by "ollama show"\n# FROM qwen2.5:3b\n\nFROM {BLOBS}{OFFICIAL_BLOB}\n'
        params = "stop <|im_end|>"

        def show(modelfile, default, template):
            return {"modelfile": default if modelfile is None else modelfile, "template": template,
                    "details": dict(DETAILS), "parameters": params, "model_info": copy.deepcopy(MODEL_INFO)}

        self.shows = {
            OFFICIAL: show(official_modelfile, official_mf, official_template),
            BASE: show(base_modelfile, f"FROM {BLOBS}{env.base_sha}\nPARAMETER x 1\n", base_template),
            FT: show(ft_modelfile, f"FROM {BLOBS}{env.ft_sha}\nPARAMETER x 1\n", ft_template),
        }  # fmt: skip

    def chat(self, model, messages, options):
        if self.fail_at is not None and len(self.calls) + 1 == self.fail_at:
            raise ca.OllamaError("Ollama /api/chat returned 500: model runner has unexpectedly stopped")
        user = user_of(messages)
        nth = self.seen[(model, user)] = self.seen.get((model, user), 0) + 1
        self.calls.append((model, user, options))
        self.prompts.append(messages)
        fn = self.base_fn if model == BASE else self.ft_fn
        return ca.Reply(fn(user, nth), "stop", 10)

    def show(self, model):
        return self.shows[model]

    def tags(self):
        return [{"name": m, "model": m, "digest": digest_of(m), "size": 1000 + i} for i, m in enumerate(self.models)]

    def version(self):
        return "0.99.0"


def run_main(env: Env, fake: FakeOllama, *extra: str) -> int:
    return ca.main(env.argv(*extra), client_factory=lambda url, timeout: fake)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ markers


def test_reply_markers():
    text = reply_text(True, True, "uk", "x")
    m = ca.reply_metrics(text)
    assert m["dice_lines"] == 1 and m["citations"] == 1 and m["cyrillic_share"] > 0.7
    assert ca.reply_metrics("No dice here. [R2, R3]")["citations"] == 2
    assert ca.dice_lines("text 🎲 inline\n  🎲 line\n🎲 another") == 2  # only lines that START with the dice
    assert ca.cyrillic_share("12 !?") is None and ca.cyrillic_share("abc") == 0.0
    assert ca.cyrillic_share("абв abc") == pytest.approx(0.5)


def test_load_cases_reads_language_and_rolls(env):
    cases = {c.id: c for c in ca.load_cases(env.val, env.manifest)}
    found = {i: (c.language, c.has_roll) for i, c in cases.items()}
    assert found == {cid: (lang, roll) for cid, lang, roll, _ in CASES}
    assert [m["role"] for m in cases["v-01"].prompt] == ["system", "user"]  # no assistant reply in the prompt
    assert cases["v-01"].reference.endswith("Що ви робите?")


def test_load_cases_needs_matching_ids(env):
    bad = {**env.manifest, "ids": {"val": ["only-one"]}}
    with pytest.raises(ca.CheckError, match="ids.val"):
        ca.load_cases(env.val, bad)


def test_selection_is_deterministic_and_ignores_file_order(env):
    cases = ca.load_cases(env.val, env.manifest)
    first = [c.id for c in ca.select_cases(cases, 4)]
    assert first == [c.id for c in ca.select_cases(list(reversed(cases)), 4)]
    assert len(set(first)) == 4 and len(ca.select_cases(cases, 99)) == len(cases)
    assert [c.id for c in ca.select_cases(cases, 2)] == first[:2]  # a prefix: --n only truncates


def records_for(*rows: tuple[str, bool, str, str, str]) -> list[dict]:
    """(language, has_roll, reference, base, ft) rows as generate_all would produce them."""
    out = []
    for i, (lang, roll, ref, base, ft) in enumerate(rows):
        out.append({
            "id": f"r{i}", "language": lang, "has_roll": roll,
            "reference": {"metrics": ca.reply_metrics(ref)},
            "base": ca.reply_record(ca.Reply(base, "stop")), "ft": ca.reply_record(ca.Reply(ft, "length")),
        })  # fmt: skip
    return out


def test_marker_summary_and_deltas():
    ref_roll = reply_text(True, True, "uk", "r")
    records = records_for(
        ("uk", True, ref_roll, "Просто текст без кидка.", ref_roll),
        ("uk", False, reply_text(False, False, "uk", "r"), "Ok, here is a reply in English.",
         reply_text(False, False, "uk", "r")),
        ("en", False, reply_text(False, False, "en", "r"), "Plain English.", "Plain English again."),
    )  # fmt: skip
    markers = {k: ca.summarize_markers(records, k) for k in ("base", "ft", "reference")}
    assert markers["reference"]["dice_line_rate_with_roll"] == 1.0
    assert markers["base"]["dice_line_rate_with_roll"] == 0.0 and markers["ft"]["dice_line_rate_with_roll"] == 1.0
    assert markers["ft"]["dice_line_rate_without_roll"] == 0.0
    assert markers["base"]["cyrillic_share_uk"] < markers["ft"]["cyrillic_share_uk"]
    assert markers["ft"]["cyrillic_share_en"] == 0.0
    assert markers["reference"]["citation_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert markers["ft"]["citation_rate_when_reference_cites"] == 1.0
    assert markers["base"]["citation_rate_when_reference_cites"] == 0.0
    assert markers["ft"]["truncated"] == 3 and markers["base"]["truncated"] == 0  # done_reason "length" was set on ft
    deltas = ca.marker_deltas(markers)
    assert deltas["ft_minus_base"]["dice_line_rate_with_roll"] == 1.0
    assert deltas["ft_minus_reference"]["dice_line_rate_with_roll"] == 0.0
    assert deltas["base_minus_reference"]["dice_line_rate_with_roll"] == -1.0


# ------------------------------------------------------------------------------------------------ gates


def test_defaults_name_the_self_quantized_base_and_the_merged_ft():
    args = ca.parse_args([])
    assert (args.base, args.ft, args.official) == ("qwen2.5-3b-selfq4:v1", "keeper-qwen25-3b:sft-v1", "qwen2.5:3b")


def test_effect_gate_passes_and_writes_results_and_card(env, capsys):
    fake = FakeOllama(env)
    assert run_main(env, fake) == 0
    results = load(env.results)
    assert results["passed"] is True and results["gates"]["effect"] == {
        "passed": True, "differ": 4, "n": 4, "share": 1.0, "min_share": 0.8, "same_ids": [],
    }
    assert results["gates"]["determinism"]["checked"] == 2 and results["gates"]["determinism"]["passed"] is True
    assert results["gates"]["template"] == {"passed": True, "equal": True, "official_equal": True, "verified": True}
    assert results["gates"]["weights"] == {
        "passed": True, "base_blob_matches_gguf": True, "ft_blob_matches_gguf": True, "blobs_distinct": True,
        "adapter_layers": False, "verified": True,
    }
    assert load(env.card)["status"] == "passed"
    out = capsys.readouterr().out
    assert "[base 1/4]" in out and "[base again 2/2]" in out and "[ft 4/4]" in out and "wrote" in out


def test_call_plan_and_decoding_options(env):
    fake = FakeOllama(env)
    run_main(env, fake, "--seed", "7", "--num-predict", "300")
    models = [m for m, _, _ in fake.calls]
    assert models == [BASE] * 4 + [BASE] * 2 + [FT] * 4  # one model at a time; the official model is never prompted
    assert {json.dumps(o, sort_keys=True) for _, _, o in fake.calls} == {
        json.dumps({"temperature": 0, "seed": 7, "num_ctx": 16384, "num_predict": 300}, sort_keys=True)
    }
    again = [u for m, u, _ in fake.calls[4:6]]
    assert again == [u for m, u, _ in fake.calls[:2]]  # the repeat is the first prompts of the first pass
    # the prompt is the chat without the final Keeper reply, so the reference target never reaches the model
    assert all([m["role"] for m in prompt] == ["system", "user"] for prompt in fake.prompts)


def test_effect_gate_fails_when_ft_equals_base(env, capsys):
    fake = FakeOllama(env, ft_fn=lambda user, n: f"BASE {user}")
    assert run_main(env, fake) == 1
    results = load(env.results)  # the results are written even though the gate failed
    assert results["passed"] is False and results["gates"]["effect"]["differ"] == 0
    assert results["gates"]["effect"]["passed"] is False and results["gates"]["determinism"]["passed"] is True
    card = load(env.card)
    assert card["status"] == "failed" and card["check_adapter"]["gates"]["effect"]["share"] == 0.0
    out = capsys.readouterr().out
    assert "FAIL effect: only 0/4" in out and "IGNORED" in out


def test_effect_gate_threshold_is_inclusive(env):
    # 3 of 4 differ = 0.75: below the default 0.8, and exactly at --min-differ 0.75
    same = ca.select_cases(ca.load_cases(env.val, env.manifest), 4)[0].id

    def ft(user, n):
        return f"BASE {user}" if user.endswith(same) else f"FT {user}"

    assert run_main(env, FakeOllama(env, ft_fn=ft)) == 1
    assert load(env.results)["gates"]["effect"]["same_ids"] == [same]
    assert run_main(env, FakeOllama(env, ft_fn=ft), "--min-differ", "0.75") == 0


def test_determinism_gate_fails_when_base_changes_between_runs(env, capsys):
    fake = FakeOllama(env, base_fn=lambda user, n: f"BASE {user} run {n}")
    assert run_main(env, fake) == 1
    results = load(env.results)
    det = results["gates"]["determinism"]
    assert det["passed"] is False and det["identical"] == 0 and det["checked"] == 2 and len(det["unstable_ids"]) == 2
    assert results["gates"]["effect"]["passed"] is True  # reported, but the run still fails
    assert "FAIL determinism: 0/2" in capsys.readouterr().out
    assert [c["base_stable"] for c in results["cases"]].count(False) == 2


def test_determinism_tolerance_can_be_relaxed(env):
    jittery = ca.select_cases(ca.load_cases(env.val, env.manifest), 2)[0].id  # one of the two repeated prompts

    def base(user, n):
        return f"BASE {user}" + (" jitter" if n == 2 and user.endswith(jittery) else "")

    assert run_main(env, FakeOllama(env, base_fn=base)) == 1
    assert load(env.results)["gates"]["determinism"]["unstable_ids"] == [jittery]
    assert run_main(env, FakeOllama(env, base_fn=base), "--min-stable", "0.5") == 0


def test_empty_replies_are_not_an_effect(env):
    fake = FakeOllama(env, ft_fn=lambda user, n: "   ")
    assert run_main(env, fake) == 1
    results = load(env.results)
    assert results["gates"]["effect"]["differ"] == 0
    assert results["markers"]["ft"]["empty"] == 4


@pytest.mark.parametrize(
    ("make_kwargs", "gate", "message"),
    [
        (lambda e: {"ft_template": TEMPLATE + " changed"}, "template", "FAIL template"),
        (lambda e: {"base_template": TEMPLATE + " changed"}, "template", "FAIL template"),
        (lambda e: {"official_template": TEMPLATE + " changed"}, "template", "FAIL template"),
        (lambda e: {"ft_modelfile": f"FROM {BLOBS}{e.base_sha}\n"}, "weights", "same weights blob"),
        (lambda e: {"ft_modelfile": f"FROM {BLOBS}{OTHER_BLOB}\n"}, "weights", "was not created from"),
        (lambda e: {"base_modelfile": f"FROM {BLOBS}{OTHER_BLOB}\n"}, "weights", "was not created from"),
        (lambda e: {"ft_modelfile": f"FROM {BLOBS}{e.ft_sha}\nADAPTER {BLOBS}{OTHER_BLOB}\n"}, "weights",
         "ADAPTER layer"),
    ],
)  # fmt: skip
def test_serving_gates_fail_on_a_definite_mismatch(env, capsys, make_kwargs, gate, message):
    fake = FakeOllama(env, **make_kwargs(env))
    assert run_main(env, fake) == 1
    results = load(env.results)
    assert results["gates"][gate]["passed"] is False and results["gates"]["effect"]["passed"] is True
    out = capsys.readouterr().out
    assert "FAIL " + gate in out and message in out and "the run continues" in out  # reported up front
    assert len(fake.calls) == 10  # ... and the results are still complete


def test_unreadable_serving_facts_are_reported_not_failed(env, capsys):
    fake = FakeOllama(env, base_modelfile="", ft_modelfile="", official_modelfile="")
    fake.shows[BASE].pop("template")
    assert run_main(env, fake) == 0
    gates_ = load(env.results)["gates"]
    assert gates_["template"] == {"passed": True, "equal": None, "official_equal": None, "verified": False}
    assert gates_["weights"]["verified"] is False and gates_["weights"]["passed"] is True
    assert "NOT VERIFIED" in capsys.readouterr().out


def test_missing_official_model_is_not_verified_rather_than_an_error(env, capsys):
    fake = FakeOllama(env, models=(BASE, FT))
    assert run_main(env, fake) == 0
    gates_ = load(env.results)["gates"]
    assert gates_["template"] == {"passed": True, "equal": True, "official_equal": None, "verified": False}
    assert load(env.card)["serving"]["official"] is None
    out = capsys.readouterr().out
    assert "NOT VERIFIED" in out and "qwen2.5:3b is not in Ollama" in out


@pytest.mark.parametrize("missing", [BASE, FT])
def test_missing_model_is_a_preflight_error(env, capsys, missing):
    fake = FakeOllama(env, models=tuple(m for m in (OFFICIAL, BASE, FT) if m != missing))
    assert run_main(env, fake) == 2
    assert not env.results.exists() and not env.card.exists() and fake.calls == []
    err = capsys.readouterr().err
    assert missing in err and "sft.ollama_models modelfiles" in err


def test_default_tag_resolution():
    tags = [{"name": "qwen2.5:3b"}, {"model": "keeper:latest", "name": "keeper:latest"}]
    assert ca.find_tag(tags, "qwen2.5:3b") is tags[0] and ca.find_tag(tags, "keeper") is tags[1]
    assert ca.find_tag(tags, "other") is None


def test_parse_modelfile_reads_blob_digests_from_windows_and_posix_paths():
    win = f"C:{BS}Users{BS}1{BS}.ollama{BS}models{BS}blobs{BS}sha256-"
    text = (f'# FROM qwen2.5:3b\nFROM {win}{OFFICIAL_BLOB}\nADAPTER {win}{OTHER_BLOB}\nPARAMETER stop "<|im_end|>"\n'
            f"TEMPLATE \"\"\"{{{{ .Prompt }}}}\"\"\"\n")  # fmt: skip
    assert ca.parse_modelfile(text) == {"from_blob": OFFICIAL_BLOB, "adapter_blobs": [OTHER_BLOB]}
    assert ca.parse_modelfile("FROM qwen2.5:3b\n") == {"from_blob": None, "adapter_blobs": []}
    assert ca.parse_modelfile("") == {"from_blob": None, "adapter_blobs": []}


# ------------------------------------------------------------------------------------------------ parity, status


def test_empty_template_is_a_value_not_an_unknown(env, capsys):
    fake = FakeOllama(env, ft_template="")  # the FT model lost its template; the others have one
    assert run_main(env, fake) == 1
    gate = load(env.results)["gates"]["template"]
    assert gate == {"passed": False, "equal": False, "official_equal": False, "verified": True}
    assert "FAIL template" in capsys.readouterr().out


def test_both_empty_templates_still_differ_from_the_official_one(env):
    fake = FakeOllama(env, base_template="", ft_template="")
    assert run_main(env, fake) == 1
    gate = load(env.results)["gates"]["template"]
    assert gate["equal"] is True and gate["official_equal"] is False and gate["passed"] is False


def test_parity_gate_passes_when_only_general_keys_differ(env):
    fake = FakeOllama(env)
    fake.shows[FT]["model_info"]["general.basename"] = "keeper"
    fake.shows[FT]["model_info"]["general.file_type"] = 17
    assert run_main(env, fake) == 0
    parity = load(env.results)["gates"]["parity"]
    assert parity["passed"] is True and parity["differences"] == [] and parity["verified"] is True
    assert parity["tensor_counts"] == {"base": 434, "ft": 434} and parity["tensor_counts_equal"] is True
    assert parity["compared"] == ["base", "ft"] and parity["ignored_prefixes"] == ["general."]
    assert parity["official_differences"] == []


@pytest.mark.parametrize(
    ("role", "key", "value"),
    [
        (FT, "qwen2.rope.freq_base", 10000.0),
        (BASE, "qwen2.context_length", 4096),
        (FT, "tokenizer.ggml.eos_token_id", 151643),
    ],
)
def test_parity_gate_fails_when_a_conversion_setting_drifts(env, capsys, role, key, value):
    fake = FakeOllama(env)
    fake.shows[role]["model_info"][key] = value
    assert run_main(env, fake) == 1
    results = load(env.results)
    gate = results["gates"]["parity"]
    assert gate["passed"] is False and [d["key"] for d in gate["differences"]] == [key]
    assert set(gate["differences"][0]["values"]) == {"base", "ft"}  # the pair the thesis compares: strict
    assert results["gates"]["effect"]["passed"] is True and results["status"] == "failed"
    out = capsys.readouterr().out
    assert f"FAIL parity: model_info of base and FT differs in: {key}" in out and "--parity-ignore" in out
    assert len(fake.calls) == 10  # reported up front, the run still completes


def test_differences_from_the_official_model_are_informational(env, capsys):
    """The official blob was converted long ago by another llama.cpp: its metadata may differ without any harm."""
    fake = FakeOllama(env)
    fake.shows[OFFICIAL]["model_info"]["qwen2.block_count"] = 28
    del fake.shows[OFFICIAL]["model_info"]["tokenizer.ggml.eos_token_id"]
    fake.shows[OFFICIAL]["details"]["parameter_size"] = "3B"
    assert run_main(env, fake) == 0
    results = load(env.results)
    gate = results["gates"]["parity"]
    assert gate["passed"] is True and gate["verified"] is True and gate["differences"] == []
    assert results["passed"] is True and results["status"] == "passed"  # not even passed_unverified
    keys = [d["key"] for d in gate["official_differences"]]
    assert keys == ["details.parameter_size", "qwen2.block_count", "tokenizer.ggml.eos_token_id"]
    block = gate["official_differences"][1]
    assert block["values"] == {"base": 36, "official": 28}
    missing = gate["official_differences"][2]["values"]
    assert missing == {"base": 151645, "official": "<missing>"}
    card = load(env.card)
    assert card["status"] == "passed"
    assert card["serving"]["parity"]["official_differences"] == gate["official_differences"]
    out = capsys.readouterr().out
    assert "WARNING (informational, not a failure)" in out and "3 key(s)" in out and "qwen2.block_count" in out
    assert "FAIL" not in out


def test_the_template_check_against_the_official_model_stays_strict(env):
    # unlike model_info, the template is copied from the official Modelfile: a difference is a real error
    assert run_main(env, FakeOllama(env, official_template=TEMPLATE + " changed")) == 1
    assert load(env.results)["gates"]["template"]["passed"] is False


def test_no_official_model_means_no_official_differences(env):
    assert run_main(env, FakeOllama(env, models=(BASE, FT))) == 0
    parity = load(env.results)["gates"]["parity"]
    assert parity["official_differences"] is None and parity["verified"] is True and parity["passed"] is True


def test_a_key_that_one_model_lacks_is_a_difference(env):
    fake = FakeOllama(env)
    del fake.shows[BASE]["model_info"]["qwen2.rope.freq_base"]
    assert run_main(env, fake) == 1
    (diff,) = load(env.results)["gates"]["parity"]["differences"]
    assert diff["key"] == "qwen2.rope.freq_base" and diff["values"]["base"] == "<missing>"


def test_details_are_compared_too(env):
    fake = FakeOllama(env)
    fake.shows[FT]["details"]["quantization_level"] = "Q8_0"
    assert run_main(env, fake) == 1
    assert [d["key"] for d in load(env.results)["gates"]["parity"]["differences"]] == ["details.quantization_level"]


def test_parity_ignore_accepts_a_benign_key_and_is_recorded(env):
    fake = FakeOllama(env)
    fake.shows[FT]["model_info"]["qwen2.context_length"] = 4096
    assert run_main(env, fake) == 1
    fake = FakeOllama(env)
    fake.shows[FT]["model_info"]["qwen2.context_length"] = 4096
    assert run_main(env, fake, "--parity-ignore", "qwen2.context_length") == 0
    settings = load(env.results)["settings"]
    assert settings["parity_ignore"] == ["general.", "qwen2.context_length"]


def test_different_tensor_counts_fail_the_parity_gate(env, capsys):
    write_gguf(env.adapter_dir / ca.FT_GGUF, tensors=400, payload=b"merged-and-quantized")
    ft_sha = gates.sha256_file(env.adapter_dir / ca.FT_GGUF)
    write_json(env.adapter_dir / ca.GGUF_META, gguf_meta(ft_sha, env.base_sha, env.adapter_sha))
    env.ft_sha = ft_sha
    assert run_main(env, FakeOllama(env)) == 1
    gate = load(env.results)["gates"]["parity"]
    assert gate["passed"] is False and gate["tensor_counts"] == {"base": 434, "ft": 400}
    assert "different tensor counts (base 434, ft 400)" in capsys.readouterr().out


def test_model_info_that_cannot_be_read_is_not_verified(env, capsys):
    fake = FakeOllama(env)
    fake.shows[BASE].pop("model_info")
    assert run_main(env, fake) == 0
    gate = load(env.results)["gates"]["parity"]
    assert gate["passed"] is True and gate["verified"] is False and gate["differences"] == []
    assert gate["compared"] == ["ft"] and gate["official_differences"] is None  # nothing to compare the FT model with
    assert load(env.results)["status"] == "passed_unverified"
    assert "NOT VERIFIED" in capsys.readouterr().out


def test_status_is_passed_only_when_everything_was_verified(env, capsys):
    assert run_main(env, FakeOllama(env)) == 0
    assert load(env.results)["status"] == "passed" and load(env.card)["status"] == "passed"
    assert "passed_unverified" not in capsys.readouterr().out

    assert run_main(env, FakeOllama(env, models=(BASE, FT))) == 0  # no official model: the template is not verified
    assert load(env.results)["status"] == "passed_unverified"
    card = load(env.card)
    assert card["status"] == "passed_unverified" and card["check_adapter"]["passed"] is True
    out = capsys.readouterr().out
    assert "card status is 'passed_unverified'" in out and "(template: NOT VERIFIED)" in out
    assert "--allow-failed-card" in out

    assert run_main(env, FakeOllama(env, ft_fn=lambda user, n: f"BASE {user}")) == 1
    assert load(env.card)["status"] == "failed"


def test_card_only_keeps_the_status(env):
    run_main(env, FakeOllama(env, models=(BASE, FT)))
    env.card.unlink()
    assert ca.main(env.argv("--card-only"), client_factory=lambda u, t: None) == 0
    assert load(env.card)["status"] == "passed_unverified"


def incomplete_file(env: Env) -> Path:
    return env.adapter_dir / "check_adapter.incomplete.json"


def test_an_ollama_error_midway_keeps_the_replies_so_far(env, capsys):
    fake = FakeOllama(env, fail_at=7)  # 4 base + 2 repeats succeed, the first FT call fails
    assert run_main(env, fake) == 2
    assert "model runner has unexpectedly stopped" in capsys.readouterr().err
    assert not env.results.exists()  # the partial run has its own file
    results = load(incomplete_file(env))
    assert results["incomplete"] is True and results["status"] == "incomplete" and results["passed"] is False
    assert "unexpectedly stopped" in results["error"] and results["schema"] == ca.RESULTS_SCHEMA
    cases = results["cases"]
    assert len(cases) == 4 and all("base" in c and "ft" not in c for c in cases)
    assert sum("base_repeat" in c for c in cases) == 2
    assert not env.card.exists()  # a card is only ever written for a complete run
    assert ca.main(env.argv("--card-only")) == 2  # there is no complete results file to build it from


def test_an_interrupted_run_does_not_overwrite_a_complete_results_file(env):
    assert run_main(env, FakeOllama(env)) == 0
    good, card = env.results.read_bytes(), env.card.read_bytes()  # the card cites results_file.sha256
    assert not incomplete_file(env).exists()
    assert run_main(env, FakeOllama(env, fail_at=8)) == 2
    assert env.results.read_bytes() == good and env.card.read_bytes() == card
    assert load(incomplete_file(env))["incomplete"] is True
    assert load(env.card)["check_adapter"]["results_file"]["sha256"] == gates.sha256_file(env.results)
    # a later complete run removes the stale partial file
    assert run_main(env, FakeOllama(env)) == 0
    assert not incomplete_file(env).exists()


def test_card_only_refuses_incomplete_results(env, capsys):
    run_main(env, FakeOllama(env, fail_at=3))
    capsys.readouterr()
    assert ca.main(env.argv("--card-only", "--results", str(incomplete_file(env)))) == 2
    assert "stopped midway" in capsys.readouterr().err


def test_card_only_gives_a_clear_error_for_results_of_an_older_schema(env, capsys):
    run_main(env, FakeOllama(env))
    old = load(env.results)
    old["schema"] = 1
    del old["status"]  # what schema 1 lacked
    write_json(env.results, old)
    env.card.unlink()
    assert ca.main(env.argv("--card-only")) == 2
    err = capsys.readouterr().err
    assert "results schema 1" in err and f"writes schema {ca.RESULTS_SCHEMA}" in err and "KeyError" not in err
    assert not env.card.exists()
    old.pop("schema")
    write_json(env.results, old)
    assert ca.main(env.argv("--card-only")) == 2
    assert "results schema None" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ waived checks


def test_a_waived_merge_check_is_visible_and_blocks_status_passed(env, capsys):
    meta = load(env.adapter_dir / ca.GGUF_META)
    meta["waived_mismatches"] = {"llama-cpp": "the checkout is at abc, not at the pinned 71255fe"}
    write_json(env.adapter_dir / ca.GGUF_META, meta)
    assert run_main(env, FakeOllama(env)) == 0  # every gate passed and was verified ...
    assert load(env.results)["status"] == "passed"
    card = load(env.card)
    assert card["status"] == "passed_unverified"  # ... but a consistency check of the merge was waived
    assert card["waived_mismatches"] == {"llama-cpp": "the checkout is at abc, not at the pinned 71255fe"}
    assert card["check_adapter"]["passed"] is True
    out = capsys.readouterr().out
    assert "WARNING: waived consistency check [llama-cpp]: the checkout is at abc" in out
    assert "card status is 'passed_unverified' (waived check: llama-cpp)" in out and "--allow-failed-card" in out
    errors, _ = model_cards.check_cards(card_profiles(env), ["ft", "base"], root=env.root)
    assert len(errors) == 1 and "passed_unverified" in errors[0]  # the eval harness needs --allow-failed-card


def test_an_adapter_that_is_not_the_one_in_train_meta_is_a_waived_mismatch(env, capsys):
    meta = load(env.adapter_dir / "train_meta.json")
    meta["adapter"]["sha256"] = "5" * 64
    write_json(env.adapter_dir / "train_meta.json", meta)
    assert run_main(env, FakeOllama(env)) == 0
    card = load(env.card)
    assert card["adapter"]["safetensors_sha256_matches_train_meta"] is False
    assert card["status"] == "passed_unverified" and list(card["waived_mismatches"]) == ["train_meta_adapter_sha256"]
    assert "555555555555" in card["waived_mismatches"]["train_meta_adapter_sha256"]
    out = capsys.readouterr().out
    assert "WARNING: waived consistency check [train_meta_adapter_sha256]" in out


def test_waived_mismatches_do_not_hide_a_failure_and_are_always_in_the_card(env):
    assert run_main(env, FakeOllama(env)) == 0
    card = load(env.card)
    assert card["waived_mismatches"] == {} and card["status"] == "passed"
    meta = load(env.adapter_dir / ca.GGUF_META)
    meta["waived_mismatches"] = {"adapter": "x"}
    write_json(env.adapter_dir / ca.GGUF_META, meta)
    assert run_main(env, FakeOllama(env, ft_fn=lambda user, n: f"BASE {user}")) == 1
    assert load(env.card)["status"] == "failed"  # failed stays failed
    assert ca.main(env.argv("--card-only"), client_factory=lambda u, t: None) == 1
    assert load(env.card)["waived_mismatches"] == {"adapter": "x"}


# ------------------------------------------------------------------------------------------------ eval harness


class StubLLMs:
    def __init__(self, **profiles):
        self.profiles = profiles

    def profile(self, name):
        return self.profiles[name]


def card_profiles(env):
    rel_card = env.card.relative_to(env.root).as_posix()
    return StubLLMs(
        ft=SimpleNamespace(finetuned=True, model_card=rel_card, provider="ollama", model=FT),
        base=SimpleNamespace(finetuned=False, model_card=rel_card, provider="ollama", model=BASE),
    )


def test_card_feeds_evals_model_cards(env):
    """The card as check_adapter writes it, read by evals/model_cards.py (the S9 side of the contract)."""
    assert run_main(env, FakeOllama(env)) == 0
    card = load(env.card)
    llms = card_profiles(env)
    ft, base = llms.profile("ft"), llms.profile("base")
    assert card["status"] == model_cards.PASSED
    assert model_cards.card_digest(card, ft) == digest_of(FT) and model_cards.card_digest(card, base) == digest_of(BASE)
    assert model_cards.weights_sha256(card, ft) == env.ft_sha
    assert model_cards.weights_sha256(card, base) == env.base_sha != env.ft_sha
    for field, path in model_cards.DATA_PATHS.items():
        assert model_cards._dig(card, path), field
    assert model_cards._dig(card, model_cards.DATA_PATHS["split_sha256"]) == gates.sha256_file(env.split)

    listing = {FT: digest_of(FT), BASE: digest_of(BASE)}
    assert model_cards.check_cards(llms, ["ft", "base"], root=env.root, listing=lambda: listing) == ([], [])
    listing[FT] = digest_of("re-created")  # the tag was re-created after the check
    errors, _ = model_cards.check_cards(llms, ["ft", "base"], root=env.root, listing=lambda: listing)
    assert len(errors) == 1 and "re-created or re-pulled" in errors[0]


def test_an_unverified_or_failed_card_stops_the_eval_harness(env):
    llms = card_profiles(env)
    run_main(env, FakeOllama(env, models=(BASE, FT)))  # passed_unverified
    errors, warnings = model_cards.check_cards(llms, ["ft", "base"], root=env.root)
    assert len(errors) == 1 and "passed_unverified" in errors[0] and warnings == []
    errors, warnings = model_cards.check_cards(llms, ["ft", "base"], root=env.root, allow_failed=True)
    assert errors == [] and len(warnings) == 1
    run_main(env, FakeOllama(env, ft_fn=lambda user, n: f"BASE {user}"))  # failed
    errors, _ = model_cards.check_cards(llms, ["ft", "base"], root=env.root)
    assert len(errors) == 1 and "'failed'" in errors[0]
    snap = model_cards.snapshot(SimpleNamespace(profile=llms.profile), ["ft"], root=env.root)["ft"]
    assert snap["status"] == "failed" and snap["weights_sha256"] == env.ft_sha


# ------------------------------------------------------------------------------------------------ the card


def test_card_collects_training_data_and_serving_facts(env):
    assert run_main(env, FakeOllama(env)) == 0
    card = load(env.card)
    assert card["name"] == "keeper-test-sft-v1" and card["status"] == "passed"

    adapter = card["adapter"]
    st = adapter["files"]["adapter/adapter_model.safetensors"]
    assert st["sha256"] == gates.sha256_file(env.adapter_dir / "adapter" / "adapter_model.safetensors")
    assert st["tensor_count"] == 4 and "adapter.gguf" not in adapter["files"]
    assert adapter["safetensors_sha256_matches_train_meta"] is True
    assert adapter["peft_config"]["r"] == 16 and "other" not in adapter["peft_config"]

    gguf = card["gguf"]  # the merged FT model and the self-quantized base, with the merge provenance
    ft_name, base_name = "keeper-qwen25-3b-sft-v1.Q4_K_M.gguf", "qwen2.5-3b-instruct.Q4_K_M.gguf"
    assert set(gguf["files"]) == {ft_name, base_name} and gguf["roles"] == {"ft": ft_name, "base": base_name}
    assert gguf["files"][ft_name]["sha256"] == env.ft_sha == gguf["sha256"]  # the contract of evals/model_cards.py
    assert gguf["files"][base_name]["sha256"] == env.base_sha == gguf["base_sha256"] != env.ft_sha
    assert gguf["files"][ft_name]["size"] == (env.adapter_dir / ft_name).stat().st_size
    assert gguf["files"][ft_name]["recorded_in_gguf_meta"] is True and gguf["files"][ft_name]["tensor_count"] == 434
    assert gguf["source"] == "gguf_meta.json" and gguf["quant"] == "Q4_K_M"
    assert gguf["llama_cpp_commit"] == "1234abcd" * 5 and gguf["merge_dtype"] == "float16"
    assert gguf["versions"] == {"peft": "0.20.0", "transformers": "5.0.0"} and gguf["base_revision"] == "rev-1"
    assert gguf["meta"]["llama_cpp"]["ref"] == "71255fefc092b0344f2acddb891388814ae961be"  # the meta file, verbatim
    assert gguf["merged_from_this_adapter"] is True
    assert gguf["meta_file"]["sha256"] == gates.sha256_file(env.adapter_dir / "gguf_meta.json")
    assert gguf["superseded"]["llama_cpp_commit"] == "71255fefc092b0344f2acddb891388814ae961be"
    assert "ADAPTER" in gguf["superseded"]["status"]

    training = card["training"]
    assert training["hyperparameters"]["lr"] == 1e-4 and training["seed"] == 3407
    assert training["losses"]["eval_loss_last"] == 1.711 and training["versions"]["trl"] == "1.13.0"
    assert training["chat_template"]["sha256"] == "e" * 64 and training["chat_template"]["same_as_transformers"] is True
    assert "string" not in training["chat_template"] and "first_rendered_example" not in training["chat_template"]
    assert training["gates"] == {"train": {"length": {"max": 10}}}

    data = card["data"]
    assert data["train"]["sha256"] == env.manifest["files"]["train.jsonl"]["sha256"]
    assert data["val"]["sha256"] == env.manifest["files"]["val.jsonl"]["sha256"]
    assert data["matches_train_meta"] == {"train.jsonl": True, "val.jsonl": True}
    assert data["git_commit"] == "d" * 40 and data["manifest"]["same_as_adapter_dir_copy"] is True
    assert card["split"]["sha256"] == gates.sha256_file(env.split) == card["split"]["manifest_sha256"]
    assert card["split"]["path"].endswith("section_split.v1.json")
    assert card["git"] == {"commit": "f" * 40, "dirty": False}

    serving = card["serving"]
    assert card["ollama_digests"] == {"ft": digest_of(FT), "base": digest_of(BASE)}  # evals/model_cards.py compares
    assert serving["base"]["digest"] == digest_of(BASE) and serving["ft"]["digest"] == digest_of(FT)
    assert serving["official"]["digest"] == digest_of(OFFICIAL) and serving["official"]["name"] == "qwen2.5:3b"
    assert serving["template_equal"] is True and serving["template_official_equal"] is True
    assert serving["base"]["from_blob"] == env.base_sha and serving["ft"]["from_blob"] == env.ft_sha
    assert serving["expected_sha256"] == {"base": env.base_sha, "ft": env.ft_sha}
    assert serving["base"]["details"]["quantization_level"] == "Q4_K_M" and serving["ollama_version"] == "0.99.0"

    check = card["check_adapter"]
    assert check["settings"]["seed"] == 0 and check["settings"]["temperature"] == 0 and check["settings"]["n"] == 4
    assert check["settings"]["base"] == BASE and check["settings"]["ft"] == FT
    assert check["gates"]["effect"]["passed"] is True and "deltas" in check and "markers" in check
    assert "text" not in json.dumps(check["cases"]) and len(check["cases"]) == 4  # no replies in the card
    assert check["results_file"]["sha256"] == gates.sha256_file(env.results)


def test_card_keeps_the_keys_the_eval_harness_reads(env):
    """evals/model_cards.py looks these paths up (S9); renaming one silently turns a digest check off."""
    run_main(env, FakeOllama(env))
    card = load(env.card)
    assert card["ollama_digests"] == {"base": digest_of(BASE), "ft": digest_of(FT)}  # as Ollama reports them
    gguf = card["gguf"]
    for name in (ca.FT_GGUF, ca.BASE_GGUF):
        assert set(gguf["files"][name]) >= {"sha256", "size"}
    assert gguf["llama_cpp_commit"] == "1234abcd" * 5 and gguf["quant"] == "Q4_K_M"
    assert gguf["source"] == "gguf_meta.json"
    assert card["data"]["train"]["sha256"] == env.manifest["files"]["train.jsonl"]["sha256"]
    assert card["data"]["val"]["sha256"] == env.manifest["files"]["val.jsonl"]["sha256"]
    assert set(card["split"]) >= {"path", "sha256"} and len(card["split"]["sha256"]) == 64
    assert card["serving"]["ft"]["digest"] == digest_of(FT) and card["serving"]["base"]["digest"] == digest_of(BASE)


def test_card_is_utf8_lf_json(env):
    run_main(env, FakeOllama(env))
    raw = env.card.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    raw.decode("utf-8")
    assert b"\r" not in env.results.read_bytes()


def test_parameter_and_system_differences_are_warned_about(env, capsys):
    fake = FakeOllama(env)
    fake.shows[FT]["parameters"] = "stop <|im_end|>\ntemperature 0.2"
    assert run_main(env, fake) == 0  # informational: not a gate
    assert load(env.card)["serving"]["parameters_equal"] is False
    assert "differ in PARAMETER or SYSTEM" in capsys.readouterr().out


def test_card_only_rebuilds_the_card_without_ollama(env, capsys):
    run_main(env, FakeOllama(env))
    first = load(env.card)
    env.card.unlink()

    def forbidden(url, timeout):
        raise AssertionError("--card-only must not talk to Ollama")

    assert ca.main(env.argv("--card-only"), client_factory=forbidden) == 0
    second = load(env.card)
    assert {k: v for k, v in second.items() if k != "generated_at"} == {
        k: v for k, v in first.items() if k != "generated_at"
    }
    assert "wrote" in capsys.readouterr().out


def test_card_only_keeps_a_failed_status_and_exit_code(env):
    run_main(env, FakeOllama(env, ft_fn=lambda user, n: f"BASE {user}"))
    env.card.unlink()
    assert ca.main(env.argv("--card-only"), client_factory=lambda u, t: None) == 1
    assert load(env.card)["status"] == "failed"


def test_card_only_needs_results_for_the_same_val_file(env, capsys):
    assert ca.main(env.argv("--card-only")) == 2
    assert "saved results" in capsys.readouterr().err
    run_main(env, FakeOllama(env))
    saved = load(env.results)
    saved["data"]["val_sha256"] = "0" * 64
    write_json(env.results, saved)
    assert ca.main(env.argv("--card-only")) == 2


def test_card_only_refuses_ggufs_that_changed_since_the_run(env, capsys):
    run_main(env, FakeOllama(env))
    write_gguf(env.adapter_dir / ca.FT_GGUF, tensors=434, payload=b"another merge")
    new_sha = gates.sha256_file(env.adapter_dir / ca.FT_GGUF)
    write_json(env.adapter_dir / ca.GGUF_META, gguf_meta(new_sha, env.base_sha, env.adapter_sha))  # other run
    assert ca.main(env.argv("--card-only")) == 2
    assert "are not the ones" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ gguf_meta.json


def test_gguf_meta_of_another_merge_is_refused(env, capsys):
    write_json(env.adapter_dir / ca.GGUF_META, gguf_meta("9" * 64, "8" * 64, env.adapter_sha))
    fake = FakeOllama(env)
    assert run_main(env, fake) == 2
    assert "gguf_meta.json describes" in capsys.readouterr().err and fake.calls == []


def test_gguf_meta_of_another_adapter_is_refused(env, capsys):
    write_json(env.adapter_dir / ca.GGUF_META, gguf_meta(env.ft_sha, env.base_sha, "7" * 64))
    assert run_main(env, FakeOllama(env)) == 2
    assert "merged from an adapter" in capsys.readouterr().err


def test_gguf_names_follow_gguf_meta_for_another_quantization(env):
    renamed = "keeper-qwen25-3b-sft-v1.Q5_K_M.gguf"
    (env.adapter_dir / ca.FT_GGUF).rename(env.adapter_dir / renamed)
    write_json(env.adapter_dir / ca.GGUF_META, gguf_meta(env.ft_sha, env.base_sha, env.adapter_sha, renamed))
    assert run_main(env, FakeOllama(env)) == 0
    gguf = load(env.card)["gguf"]
    assert renamed in gguf["files"] and gguf["roles"]["ft"] == renamed


def test_gguf_meta_may_use_other_key_names_and_have_no_digests(env, capsys):
    write_json(env.adapter_dir / ca.GGUF_META, {"llama_cpp_commit": "abc123", "merged_dtype": "bfloat16"})
    assert run_main(env, FakeOllama(env)) == 0
    gguf = load(env.card)["gguf"]
    assert gguf["llama_cpp_commit"] == "abc123" and gguf["merge_dtype"] == "bfloat16"
    assert gguf["files"][ca.FT_GGUF]["recorded_in_gguf_meta"] is None
    assert "records no sha256" in capsys.readouterr().out


def test_gguf_meta_without_a_commit_is_flagged(env, capsys):
    write_json(env.adapter_dir / ca.GGUF_META, {"files": {"ft": env.ft_sha, "base": env.base_sha}})
    assert run_main(env, FakeOllama(env)) == 0
    assert load(env.card)["gguf"]["llama_cpp_commit"] is None
    assert "no llama.cpp commit" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({"llama_cpp_commit": "aaa"}, "aaa"),
        ({"llama_cpp": {"commit": "bbb"}}, "bbb"),
        ({"llama_cpp": "ccc"}, "ccc"),
        ({"build": {"llama_cpp": {"git_sha": "ddd"}}}, "ddd"),
        ({"llama_cpp": {}}, None),
        ({}, None),
    ],
)
def test_llama_cpp_commit_is_found_under_the_usual_keys(meta, expected):
    assert ca.llama_cpp_commit(meta) == expected


# ------------------------------------------------------------------------------------------------ refusing bad inputs


def test_val_file_that_is_not_the_training_data_is_refused(env, capsys):
    meta = load(env.adapter_dir / "train_meta.json")
    meta["data"]["val_sha256"] = "1" * 64
    write_json(env.adapter_dir / "train_meta.json", meta)
    fake = FakeOllama(env)
    assert run_main(env, fake) == 2
    assert "not the file the adapter was trained with" in capsys.readouterr().err and fake.calls == []


def test_edited_val_file_fails_the_manifest_check(env, capsys):
    with env.val.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(chat_row("S", "u", "a")) + "\n")
    assert run_main(env, FakeOllama(env)) == 2
    assert "does not match manifest" in capsys.readouterr().err


def test_split_file_that_differs_from_the_manifest_is_refused(env, capsys):
    env.split.write_text('{"seed": 1}\n', encoding="utf-8", newline="\n")
    assert run_main(env, FakeOllama(env)) == 2
    assert "differs from the split" in capsys.readouterr().err


@pytest.mark.parametrize(
    "missing",
    [ca.FT_GGUF, ca.BASE_GGUF, ca.GGUF_META, "adapter/adapter_model.safetensors", "train_meta.json"],
)
def test_missing_files_are_reported(env, capsys, missing):
    (env.adapter_dir / missing).unlink()
    assert run_main(env, FakeOllama(env)) == 2
    err = capsys.readouterr().err
    assert "not found" in err and ("merge_and_quantize" in err or Path(missing).name in err)


def test_gguf_header_and_safetensors_count(tmp_path):
    good = tmp_path / "a.gguf"
    write_gguf(good, 504)
    assert ca.gguf_header(good) == {"version": 3, "tensor_count": 504, "metadata_kv_count": 7}
    bad = tmp_path / "b.gguf"
    bad.write_bytes(b"not a gguf file at all, really")
    with pytest.raises(ca.CheckError, match="not a GGUF"):
        ca.gguf_header(bad)
    st = tmp_path / "m.safetensors"
    write_safetensors(st, ["x", "y", "z"])
    assert ca.safetensors_tensor_count(st) == 3
    st.write_bytes(b"short")
    assert ca.safetensors_tensor_count(st) is None


# ------------------------------------------------------------------------------------------------ the HTTP client


def make_client(handler) -> ca.OllamaClient:
    return ca.OllamaClient("http://ollama.test:11434", timeout=5, transport=httpx.MockTransport(handler))


def test_client_chat_sends_deterministic_options_and_parses_the_reply():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Привіт 🎲"},
                                         "done": True, "done_reason": "length", "eval_count": 5})

    reply = make_client(handler).chat("m", [{"role": "user", "content": "hi"}], {"temperature": 0, "seed": 0})
    assert reply == ca.Reply("Привіт 🎲", "length", 5)
    assert seen["url"] == "http://ollama.test:11434/api/chat"
    assert seen["body"] == {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False,
                            "options": {"temperature": 0, "seed": 0}, "keep_alive": 0}


def test_client_show_tags_version():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/show":
            echo = json.loads(request.content)
            return httpx.Response(200, json={"template": "T", "modelfile": "FROM x", "echo": echo})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "a:1", "digest": "d"}]})
        return httpx.Response(200, json={"version": "0.12.0"})

    client = make_client(handler)
    assert client.show("a:1")["echo"] == {"model": "a:1"} and client.tags() == [{"name": "a:1", "digest": "d"}]
    assert client.version() == "0.12.0"


def test_client_errors_are_readable():
    def not_found(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'nope' not found"})

    with pytest.raises(ca.OllamaError, match=r"404.*model 'nope' not found"):
        make_client(not_found).show("nope")
    assert make_client(not_found).version() is None  # the version is optional

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ca.OllamaError, match="cannot connect to Ollama"):
        make_client(refuse).tags()

    def no_message(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"done": True})

    with pytest.raises(ca.OllamaError, match="message.content"):
        make_client(no_message).chat("m", [], {})


def test_ollama_down_gives_exit_code_2(env, capsys):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def factory(url, timeout):
        return ca.OllamaClient(url, timeout, transport=httpx.MockTransport(refuse))

    assert ca.main(env.argv(), client_factory=factory) == 2
    assert "cannot connect to Ollama" in capsys.readouterr().err and not env.results.exists()
