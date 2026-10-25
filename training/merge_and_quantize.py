r"""Merge the Phase 4 LoRA into the full-precision base and quantize BOTH models to GGUF. Colab CPU runtime, no GPU.

Why: this Ollama rejects `ADAPTER` ("LoRA adapters are no longer supported"), so the adapter cannot be applied at
serving time. The fallback of docs/critique.md C4 is used: merge the LoRA into the full-precision base and quantize the
base the very same way, so the two Ollama models (base and fine-tuned) differ only by the training.

Colab (Runtime -> Change runtime type -> CPU; a GPU runtime burns quota for nothing). Cell 1, Drive and code:

    from google.colab import drive
    drive.mount("/content/drive")
    !git clone --depth 1 https://github.com/xgrsai/gm-assistant.git /content/gm-assistant

Cell 2, dependencies (does not touch torch; llama.cpp's own requirements files are deliberately NOT installed, they
pin torch / transformers):

    !pip install -q -r /content/gm-assistant/training/requirements-merge.txt
    !pip uninstall -y torchao   # Colab's torchao 0.10 makes PEFT refuse to load the adapter; the merge doesn't need it

Cell 3, merge + convert + quantize (clones and builds llama.cpp on the first run; expect an hour or so on a CPU
runtime: an estimate, not a measurement):

    !python /content/gm-assistant/training/merge_and_quantize.py \
        --adapter-dir /content/drive/MyDrive/sft/out/keeper-qwen25-3b-sft-v1/adapter \
        --out /content/drive/MyDrive/sft/out/keeper-qwen25-3b-sft-v1/gguf

Order of work (every check stops the run before the next, more expensive, stage):
  1. the adapter files are read and hashed; if <adapter-dir>/../train_meta.json names another adapter sha256, the
     run stops (`--allow-mismatch adapter` waives it; every waived problem is recorded in the meta);
  2. the base (`--base`, default Qwen/Qwen2.5-3B-Instruct, the original full-precision model) is downloaded as an HF
     folder at ONE commit: `--base-revision`, else for the default base the revision that train_meta.json recorded
     (DEFAULT_BASE_REVISION), else the current main resolved to its sha. The meta records the sha, never a branch name.
     A base id or revision that differs from the `base.hf_base_for_gguf` / `hf_base_revision` of train_meta.json
     stops the run (`--allow-mismatch base-revision`);
  3. merge: the base is loaded in float16 on the CPU, the adapter is applied with PEFT `PeftModel.from_pretrained`,
     `merge_and_unload(safe_merge=True)`, and the weights are saved to <work-dir>/merged. The base's config and
     tokenizer files are then copied over byte for byte (so both GGUFs are converted from the same config, tokenizer
     and chat template, whatever the transformers version would have re-serialised).
     Checks: PEFT attached layers x targets LoRA modules; the adapter's lora_B tensors are not all zero and are really
     loaded; the saved folder holds exactly the base's tensor names (an extra lm_head.weight would change the GGUF);
     and (C4: an adapter that merged as a no-op is a failure) on the SAVED files, with BOTH sides rounded to float16
     (the base is bf16 on disk), a LoRA target tensor differs from the base in at least 1% of the sampled values while
     a non-target control tensor is unchanged. A failed check exits non-zero;
  4. llama.cpp is cloned at `--llama-cpp-ref` (default: the commit that converted the v1 adapter) and `llama-quantize`
     is built (cmake, CPU only, Release; both skipped when already there). An existing checkout at another commit
     stops the run (`--allow-mismatch llama-cpp`). That check, the adapter check and the base-revision check all run
     FIRST, before anything is downloaded, built, written or (with --force) deleted;
  5. for the base folder and then for the merged folder: `convert_hf_to_gguf.py --outtype f16`, `llama-quantize` to
     `--quant` (default Q4_K_M, no imatrix), the f16 intermediate is deleted (unless --keep-f16), the quantized file is
     copied to --out and verified by sha256: `qwen2.5-3b-instruct.<QUANT>.gguf`, `keeper-qwen25-3b-sft-v1.<QUANT>.gguf`;
  6. `gguf_meta.json` goes to --out (llama.cpp commit, library versions, base revision, adapter sha256 and config,
     merge checks, sha256 and size of both GGUFs, script hash and git state), then the next steps are printed.

The LoRA was trained against the NF4 4-bit base and is merged into the float16 base: the usual QLoRA practice, a small
train/serve mismatch that the thesis states (`merge.note` in the meta).

The pure parts (names, commands, checks, meta assembly) are unit-tested in src/backend/tests/test_merge_quantize.py;
the heavy parts (model download, merge, llama.cpp build, conversion) were NOT run when this was written.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # a copy of gates.py next to this script (upload without the repo)
sys.path.insert(0, str(HERE.parent / "src" / "backend"))  # the repo checkout
try:
    from sft import gates  # noqa: E402
except ImportError:  # no repo, gates.py copied next to the script
    import gates  # noqa: E402

DEFAULT_BASE = "Qwen/Qwen2.5-3B-Instruct"  # original full-precision model: what the LoRA is merged into
# The revision of DEFAULT_BASE that train_meta.json recorded for the v1 adapter (base.hf_base_revision): used when
# --base is the default and --base-revision is not given, so that the merge does not follow a moving `main`.
DEFAULT_BASE_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
TRAINED_AGAINST = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"  # the NF4 base the adapter was trained on
DEFAULT_QUANT = "Q4_K_M"
BASE_STEM = "qwen2.5-3b-instruct"
FT_STEM = "keeper-qwen25-3b-sft-v1"
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp"
# The commit that converted the v1 adapter on Colab (its convert script imported fine next to transformers 5.17.0).
LLAMA_CPP_REF = "71255fefc092b0344f2acddb891388814ae961be"
META_NAME = "gguf_meta.json"
MERGE_INFO_NAME = "merge_info.json"  # marker inside <work-dir>/merged: what that folder was made from
META_SCHEMA = 1
MERGE_DTYPE = "float16"
SAMPLE_ELEMENTS = 65536  # elements compared per tensor in the weight check (strided; whole tensor when smaller)
MIN_FREE_GB = 30  # base 6 + merged 6 + one f16 GGUF 6 + two Q4 GGUFs 4 + HF cache/download slack
GGUF_HEAD_BYTES = 64 * 2**20  # the GGUF key/value block (incl. the tokenizer) sits at the start of the file
GGUF_TYPE_STRING = 8
CMAKE_FLAGS = [  # CPU only; the curl / server / test targets are not needed for llama-quantize
    "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_SERVER=OFF",
]  # fmt: skip
ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")
CONTROL_TENSOR = "model.norm.weight"  # not a LoRA target: the merge must leave it untouched
MIN_CHANGED_FRACTION = 0.01  # a LoRA target tensor must differ from the base in at least this share of the sample
MISMATCH_CHOICES = ("adapter", "base-revision", "llama-cpp")  # checks that --allow-mismatch can wave through
BASE_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.jinja"]  # weights, configs, tokenizer files (no README)
REQUIRED_MODULES = ["torch", "transformers", "peft", "safetensors", "huggingface_hub", "numpy"]
PACKAGES = ["torch", "transformers", "peft", "safetensors", "huggingface_hub", "tokenizers", "accelerate", "numpy"]
ADAPTER_CONFIG_KEYS = (
    "peft_type", "task_type", "r", "lora_alpha", "lora_dropout", "bias", "use_rslora", "use_dora", "fan_in_fan_out",
    "init_lora_weights", "rank_pattern", "alpha_pattern", "modules_to_save", "base_model_name_or_path", "peft_version",
)  # fmt: skip
TRAIN_SERVE_NOTE = (
    f"The LoRA was trained against the NF4 4-bit base ({TRAINED_AGAINST}) and merged into the full-precision base "
    f"weights in {MERGE_DTYPE} (standard QLoRA practice). The base and the fine-tuned model then go through the same "
    "convert_hf_to_gguf + llama-quantize pipeline from the same base revision, so they differ only by the training."
)


# ---------------------------------------------------------------------------------------------------------------------
# Pure helpers (unit-tested; no torch, no network, no subprocess)
# ---------------------------------------------------------------------------------------------------------------------


def normalize_quant(value: str) -> str:
    """`q4_k_m` -> `Q4_K_M`; anything that is not a llama.cpp quantization name (or is a float type) is refused."""
    quant = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9_]+", quant):
        raise gates.GateError(f"--quant {value!r} is not a llama.cpp quantization type (e.g. Q4_K_M, Q5_K_M, Q8_0)")
    if quant in {"F16", "BF16", "F32"}:
        raise gates.GateError(f"--quant {quant} is not a quantization (the f16 intermediate is built anyway)")
    return quant


def f16_file_name(stem: str) -> str:
    return f"{stem}.f16.gguf"


def gguf_file_name(stem: str, quant: str) -> str:
    return f"{stem}.{quant}.gguf"


def is_commit_sha(ref: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{40}", ref) is not None


def ref_matches(commit: str | None, ref: str) -> bool | None:
    """Is the checkout at `ref`? Only decidable for a (possibly abbreviated) hex commit; None for branches and tags."""
    if commit is None or re.fullmatch(r"[0-9a-f]{7,40}", ref) is None:
        return None
    return commit.startswith(ref)


def pinned_base_revision(base_id: str, requested: str | None) -> str | None:
    """The revision to use: the requested one, else the pinned one for the default base, else None (resolve main)."""
    if requested:
        return requested
    return DEFAULT_BASE_REVISION if base_id == DEFAULT_BASE else None


def base_revision_problem(train_meta_base: dict[str, Any] | None, base_id: str, revision: str | None) -> str | None:
    """Is the base we merge into the one train_meta.json recorded (`base.hf_base_for_gguf` / `hf_base_revision`)?"""
    if not train_meta_base:
        return None
    recorded_id, recorded_revision = train_meta_base.get("hf_base_for_gguf"), train_meta_base.get("hf_base_revision")
    if recorded_id and recorded_id != base_id:
        return f"train_meta.json records the HF base {recorded_id}, but --base is {base_id}"
    if recorded_revision and revision and recorded_revision != revision:
        return (
            f"train_meta.json records base revision {recorded_revision[:12]}, but {base_id} is used at {revision[:12]}"
        )
    return None


def llama_cpp_problem(commit: str | None, ref: str) -> str | None:
    if ref_matches(commit, ref) is False:
        return f"the llama.cpp checkout is at {commit}, not at the pinned ref {ref}"
    return None


def revision_from_snapshot_path(path: str | Path) -> str | None:
    """HF cache folders are `.../snapshots/<commit sha>`: the folder name is the revision that was downloaded."""
    name = Path(path).name
    return name if is_commit_sha(name) else None


def missing_modules(names: Sequence[str], finder: Callable[[str], Any] = find_spec) -> list[str]:
    return [name for name in names if finder(name) is None]


# PEFT refuses to inject LoRA layers when an older torchao is installed (Colab ships 0.10.0, PEFT 0.20 wants >= 0.16),
# although this merge never uses torchao. Removing the package is the fix.
TORCHAO_MIN = (0, 16, 0)


def torchao_problem(version: str | None) -> str | None:
    if version is None:
        return None
    parts = tuple(int(p) for p in re.findall(r"\d+", version)[:3])
    if parts >= TORCHAO_MIN:
        return None
    return (f"torchao {version} is installed, and PEFT refuses to load the adapter with torchao older than "
            f"{'.'.join(map(str, TORCHAO_MIN))} (this merge does not use torchao): run `pip uninstall -y torchao`, "
            "then rerun this script")


def installed_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def gpu_runtime_warning(which: Callable[[str], str | None] = shutil.which) -> str | None:
    if which("nvidia-smi"):
        return (
            "a GPU is attached to this runtime. This script needs none: switch to a CPU runtime "
            "(Runtime -> Change runtime type) so that compute units are not spent"
        )
    return None


def check_free_space(free_bytes: int, need_gb: float = MIN_FREE_GB) -> None:
    if free_bytes < need_gb * 2**30:
        raise gates.GateError(
            f"only {free_bytes / 2**30:.1f} GB free in the work dir, about {need_gb:.0f} GB are needed "
            "(base + merged folder + one f16 GGUF at a time + the quantized files)"
        )


def validate_adapter_config(config: dict[str, Any]) -> None:
    if config.get("peft_type") != "LORA":
        raise gates.GateError(f"adapter_config.json: peft_type is {config.get('peft_type')!r}, only LORA is merged")
    targets = config.get("target_modules")
    if not isinstance(targets, list) or not targets:
        raise gates.GateError("adapter_config.json: target_modules must be a non-empty list of module names")
    unknown = sorted(set(targets) - set(ATTN_MODULES) - set(MLP_MODULES))
    if unknown:
        raise gates.GateError(f"adapter_config.json: unsupported target modules {unknown} (Qwen2 projections only)")


def adapter_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    summary = {key: config.get(key) for key in ADAPTER_CONFIG_KEYS}
    summary["target_modules"] = sorted(config.get("target_modules") or [])
    r, alpha = config.get("r"), config.get("lora_alpha")
    summary["scaling"] = alpha / r if isinstance(r, int) and r and isinstance(alpha, int | float) else None
    return summary


def adapter_matches_train_meta(train_meta: dict[str, Any] | None, sha256: str) -> bool | None:
    """Does train_meta.json name this adapter file? None when there is no train_meta or it records no hash."""
    adapter = (train_meta or {}).get("adapter")
    recorded = adapter.get("sha256") if isinstance(adapter, dict) else None
    return None if not recorded else recorded == sha256


def expected_lora_modules(num_layers: int, target_modules: Sequence[str]) -> int:
    """LoRA modules PEFT must attach: every target projection in every decoder layer."""
    return num_layers * len(set(target_modules))


def weight_name(layer: int, module: str) -> str:
    if module not in ATTN_MODULES and module not in MLP_MODULES:
        raise gates.GateError(f"unsupported module {module!r}")
    block = "self_attn" if module in ATTN_MODULES else "mlp"
    return f"model.layers.{layer}.{block}.{module}.weight"


def probe_tensor_names(num_layers: int, target_modules: Sequence[str]) -> dict[str, Any]:
    """Base weights to compare after the merge: an early and a late LoRA target, plus an untouched control tensor."""
    targets = sorted(set(target_modules))
    if not targets or num_layers < 1:
        raise gates.GateError("nothing to probe: no target modules or no layers")
    first = "q_proj" if "q_proj" in targets else targets[0]
    last = "down_proj" if "down_proj" in targets else targets[-1]
    names = [weight_name(0, first), weight_name(num_layers - 1, last)]
    return {"targets": list(dict.fromkeys(names)), "control": CONTROL_TENSOR}


def sample_step(count: int, sample: int = SAMPLE_ELEMENTS) -> int:
    """Stride that leaves about `sample` of `count` elements. Odd, so that it does not line up with the power-of-two
    row widths of the weight matrices (which would sample the same few columns over and over)."""
    step = max(1, count // sample)
    return step + 1 if step > 1 and step % 2 == 0 else step


def shard_for(weight_map: dict[str, str] | None, name: str) -> str:
    """The safetensors file holding `name`: from the checkpoint index, or the single `model.safetensors`."""
    if weight_map is None:
        return "model.safetensors"
    if name not in weight_map:
        raise gates.GateError(f"tensor {name} is not in the checkpoint index (keys look like {sorted(weight_map)[:2]})")
    return weight_map[name]


def tensor_diff(base: Sequence[float], merged: Sequence[float]) -> dict[str, Any]:
    """Compare equally long samples of one weight tensor. Non-finite merged values are counted, not averaged."""
    if len(base) != len(merged) or not len(base):
        raise gates.GateError(f"cannot compare samples of different or zero length ({len(base)} vs {len(merged)})")
    changed = nonfinite = 0
    max_diff = total = 0.0
    for before, after in zip(base, merged, strict=True):
        if not (math.isfinite(before) and math.isfinite(after)):
            nonfinite += 1
            continue
        diff = abs(after - before)
        if diff > 0:
            changed += 1
            total += diff
            max_diff = max(max_diff, diff)
    return {
        "sampled": len(base), "changed": changed, "changed_fraction": round(changed / len(base), 6),
        "max_abs_diff": max_diff, "mean_abs_diff": total / len(base), "nonfinite": nonfinite,
    }  # fmt: skip


def as_fp16(values: Sequence[float]) -> list[float]:
    """Round each value to float16 (IEEE round-to-nearest-even, like `tensor.half()`); out of range becomes +-inf.

    The base weights are stored in bfloat16 and the merged ones are float16. Compared as they are, tiny weights (below
    about 1.5e-5, float16 subnormals) differ after the bf16 -> fp16 cast ALONE, which would make a no-op merge look
    like a change. Both sides are therefore rounded to float16 first, as the GGUF converter does.
    """
    out = []
    for value in values:
        try:
            out.append(struct.unpack("<e", struct.pack("<e", value))[0])
        except OverflowError:
            out.append(math.copysign(math.inf, value))
    return out


def changed_share(diff: dict[str, Any]) -> float:
    return diff["changed"] / diff["sampled"] if diff["sampled"] else 0.0


def merge_check(
    targets: dict[str, dict[str, Any]], control_name: str, control: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """(problems, warnings) of the saved merged weights vs the base. Any problem must stop the run (critique C4).

    `targets` / `control` are `tensor_diff` results of float16-rounded samples (see `as_fp16`): identical weights then
    give exactly 0 changed values, so the control tensor must show none, and a real LoRA delta moves a large share of a
    target tensor (at least MIN_CHANGED_FRACTION of the sample).
    """
    problems: list[str] = []
    warnings: list[str] = []
    if not targets:
        problems.append("no LoRA target tensor was compared")
    for name, diff in {**targets, control_name: control}.items():
        if diff["nonfinite"]:
            problems.append(f"{name}: {diff['nonfinite']} non-finite values (NaN/inf) in the merge or the base")
    moved = [name for name, diff in targets.items() if changed_share(diff) >= MIN_CHANGED_FRACTION]
    if targets and not moved:
        best = max(changed_share(diff) for diff in targets.values())
        problems.append(
            f"NO-OP MERGE: no compared LoRA target tensor differs from the base in at least {MIN_CHANGED_FRACTION:.0%} "
            f"of its sampled float16 values (best {best:.4%}; 0% = identical). The adapter did not change the weights "
            "(adapter weights not loaded, or LoRA B is zero), so base and FT would be the same model"
        )
    elif len(moved) < len(targets):
        weak = sorted(set(targets) - set(moved))
        warnings.append(f"LoRA target tensors (almost) identical to the base (partial merge?): {weak}")
    if control["changed"]:
        problems.append(
            f"control tensor {control_name} differs from the base ({control['changed']} of {control['sampled']} "
            f"sampled float16 values, max |diff| {control['max_abs_diff']:.3g}): the merge touched weights it should "
            "not, or the two folders are different models"
        )
    return problems, warnings


def tensor_set_problems(base_names: Iterable[str], merged_names: Iterable[str]) -> list[str]:
    """The merged checkpoint must hold exactly the base's tensors (an extra lm_head.weight changes the GGUF's set)."""
    base, merged = set(base_names), set(merged_names)
    problems = []
    if merged - base:
        problems.append(f"{len(merged - base)} tensors only in the merged checkpoint, e.g. {sorted(merged - base)[:3]}")
    if base - merged:
        problems.append(
            f"{len(base - merged)} tensors missing in the merged checkpoint, e.g. {sorted(base - merged)[:3]}"
        )
    return problems


def lora_load_problems(
    *, expected_modules: int, found_modules: int, file_nonzero_b: int, model_nonzero_b: int
) -> list[str]:
    """Did PEFT attach and load the whole adapter? A LoRA whose B matrices stayed at zero merges as a no-op."""
    problems = []
    if found_modules != expected_modules:
        problems.append(
            f"PEFT attached {found_modules} LoRA modules, the adapter config implies {expected_modules} "
            "(decoder layers x target modules): the adapter does not fit this base"
        )
    if file_nonzero_b == 0:
        problems.append("every lora_B tensor in adapter_model.safetensors is zero: the adapter is untrained or empty")
    elif model_nonzero_b != file_nonzero_b:
        problems.append(
            f"{file_nonzero_b} lora_B tensors are non-zero in the adapter file but {model_nonzero_b} in the loaded "
            "model: adapter weights were not (all) loaded (key mismatch? see the PEFT warning above)"
        )
    return problems


def load_kwarg_candidates(dtype: Any) -> list[dict[str, Any]]:
    """`from_pretrained` keyword sets to try in order: transformers 5 names it `dtype`, 4 named it `torch_dtype`."""
    return [
        {"dtype": dtype, "low_cpu_mem_usage": True},
        {"dtype": dtype},
        {"torch_dtype": dtype, "low_cpu_mem_usage": True},
    ]


def base_files_to_copy(names: Iterable[str]) -> list[str]:
    """Config / tokenizer / chat-template files of the base folder (not the weights and not their shard index)."""
    return sorted(
        name
        for name in names
        if (name.endswith(".json") and name != "model.safetensors.index.json") or name.endswith((".txt", ".jinja"))
    )


def find_gguf_string(head: bytes, key: str) -> str | None:
    """Value of a string key in the first bytes of a GGUF file (header KV block), or None if it is not there.

    KV layout (GGUF v2/v3): u64 key length, key bytes, u32 value type (8 = string), u64 length, bytes.
    """
    key_bytes = key.encode("utf-8")
    needle = struct.pack("<Q", len(key_bytes)) + key_bytes
    at = head.find(needle)
    while at >= 0:
        pos = at + len(needle)
        if pos + 12 <= len(head):
            (value_type,) = struct.unpack_from("<I", head, pos)
            (length,) = struct.unpack_from("<Q", head, pos + 4)
            start = pos + 12
            if value_type == GGUF_TYPE_STRING and start + length <= len(head):
                return head[start : start + length].decode("utf-8", errors="replace")
        at = head.find(needle, at + 1)
    return None


# --- shell commands (built here, run by `run`) ---------------------------------------------------------------------


def git_fetch_commands(llama_dir: Path, url: str, ref: str) -> list[list[str]]:
    """A depth-1 checkout of one commit / tag / branch (GitHub serves a single SHA)."""
    d = str(llama_dir)
    return [
        ["git", "init", "--quiet", d],
        ["git", "-C", d, "remote", "add", "origin", url],
        ["git", "-C", d, "fetch", "--quiet", "--depth", "1", "origin", ref],
        ["git", "-C", d, "checkout", "--quiet", "--detach", "FETCH_HEAD"],
    ]


def quantize_binary(llama_dir: Path) -> Path:
    return llama_dir / "build" / "bin" / "llama-quantize"


def cmake_configure_command(llama_dir: Path) -> list[str]:
    return ["cmake", "-S", str(llama_dir), "-B", str(llama_dir / "build"), *CMAKE_FLAGS]


def cmake_build_command(llama_dir: Path, jobs: int) -> list[str]:
    return ["cmake", "--build", str(llama_dir / "build"), "--target", "llama-quantize", "-j", str(jobs)]


def convert_command(python: str, llama_dir: Path, model_dir: Path, outfile: Path) -> list[str]:
    script = str(llama_dir / "convert_hf_to_gguf.py")
    return [python, script, str(model_dir), "--outfile", str(outfile), "--outtype", "f16"]


def quantize_command(binary: Path, source: Path, target: Path, quant: str, threads: int) -> list[str]:
    return [str(binary), str(source), str(target), quant, str(threads)]


# --- meta and messages ---------------------------------------------------------------------------------------------


def build_gguf_meta(
    *,
    started_at: str,
    finished_at: str,
    timings_s: dict[str, float],
    code: dict[str, Any],
    environment: dict[str, Any],
    base: dict[str, Any],
    adapter: dict[str, Any],
    merge: dict[str, Any],
    tokenizer: dict[str, Any],
    llama_cpp: dict[str, Any],
    quantization: dict[str, Any],
    outputs: dict[str, Any],
    commands: list[str],
    waived_mismatches: dict[str, str],
) -> dict[str, Any]:
    """Assemble `gguf_meta.json`. Pure: the script gathers the facts, this only fixes the shape."""
    return {
        "schema": META_SCHEMA,
        "kind": "merged_gguf",
        "started_at": started_at,
        "finished_at": finished_at,
        "timings_s": timings_s,
        "code": code,
        "environment": environment,
        "base": base,
        "adapter": adapter,
        "merge": merge,
        "tokenizer": tokenizer,
        "llama_cpp": llama_cpp,
        "quantization": quantization,
        "outputs": outputs,
        "commands": commands,
        "waived_mismatches": waived_mismatches,
    }


def next_steps_text(out_dir: Path, files: Sequence[str], quant: str, ft_stem: str = FT_STEM) -> str:
    listing = "\n".join(f"       {name}" for name in files)
    return (
        f"\nDone. Next steps, on the machine with Ollama (start in the repo root; one command per line, which also "
        f"works in Windows PowerShell 5.1):\n"
        f"  1. Download from {out_dir} into the git-ignored adapters/{ft_stem}/:\n{listing}\n"
        f"  2. Write the Modelfiles:\n"
        f"       cd src/backend\n"
        f"       uv run python -m sft.ollama_models modelfiles --gguf-dir ../../adapters/{ft_stem}\n"
        f"  3. Create the Ollama models, in the folder where that command wrote Modelfile.base / Modelfile.ft:\n"
        f"       cd ../../adapters/{ft_stem}\n"
        f"       ollama create qwen2.5-3b-selfq4:v1 -f Modelfile.base\n"
        f"       ollama create keeper-qwen25-3b:sft-v1 -f Modelfile.ft\n"
        f"  4. Then the adapter-effect check (C4):\n"
        f"       cd ../../src/backend\n"
        f"       uv run python -m sft.check_adapter\n"
        f"Both models are {quant}, produced by the same pipeline; see {META_NAME} for hashes and versions.\n"
    )


# ---------------------------------------------------------------------------------------------------------------------
# Environment facts and small IO helpers
# ---------------------------------------------------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@contextmanager
def timed(timings: dict[str, float], name: str) -> Iterator[None]:
    start = time.monotonic()
    try:
        yield
    finally:
        timings[name] = round(time.monotonic() - start, 1)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run(cmd: list[str], commands: list[str], *, cwd: Path | None = None) -> None:
    """Run a command with its output streamed to the notebook; the command line is kept for the meta."""
    line = shlex.join(cmd)
    commands.append(line)
    log("$ " + line)
    subprocess.run(cmd, check=True, cwd=cwd)


def git_state(path: Path) -> dict:
    """Commit and dirtiness of the git checkout containing `path` (both None outside a checkout / without git)."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(dirty)}


def code_provenance() -> dict:
    """Which code produced the GGUFs: hashes of this script and of gates.py, plus the git state if there is one."""
    return {
        "merge_and_quantize.py_sha256": gates.sha256_file(Path(__file__)),
        "gates.py_sha256": gates.sha256_file(Path(gates.__file__)),
        "git": git_state(HERE),
    }


def hf_revision(repo_id: str, revision: str | None = None) -> str | None:
    """Commit sha of a branch / tag of the repo on the Hub (default: main); None when it cannot be asked (offline)."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo_id, revision=revision).sha
    except Exception:  # noqa: BLE001
        return None


def resolve_base_revision(
    base_id: str, requested: str | None, resolve: Callable[[str, str | None], str | None] | None = None
) -> str | None:
    """The commit sha to download: pinned / requested if it is one, else a branch / tag / main resolved on the Hub.

    A requested branch or tag that cannot be resolved stops the run (the meta must record a commit, not a name);
    None is returned only when nothing was requested and the Hub is unreachable (the cache folder name then tells).
    """
    wanted = pinned_base_revision(base_id, requested)
    if wanted and is_commit_sha(wanted):
        return wanted
    sha = (resolve or hf_revision)(base_id, wanted)
    if sha is None and wanted:
        raise gates.GateError(f"cannot resolve revision {wanted!r} of {base_id} to a commit (offline?)")
    return sha


def enforce_match(kind: str, problem: str | None, allowed: Sequence[str]) -> str | None:
    """Stop on a mismatch unless `--allow-mismatch <kind>` was given; returns the waived problem (for the meta)."""
    if problem is None:
        return None
    if kind not in allowed:
        raise gates.GateError(f"{problem}. Fix the input, or pass --allow-mismatch {kind} to go on (it is recorded)")
    log(f"WARNING (waived by --allow-mismatch {kind}): {problem}")
    return problem


def existing_llama_cpp_problem(llama_dir: Path, ref: str) -> str | None:
    """Is an EXISTING llama.cpp checkout at `ref`? None when there is no checkout yet (it will be cloned at `ref`)."""
    if not (llama_dir / "convert_hf_to_gguf.py").exists():
        return None
    return llama_cpp_problem(git_state(llama_dir)["commit"], ref)


def adapter_problem(train_meta: dict[str, Any] | None) -> str | None:
    """`inspect_adapter`'s train_meta record names another adapter file than the one given."""
    if train_meta and train_meta["names_this_adapter"] is False:
        return (
            f"{train_meta['file']} records adapter sha256 {train_meta['recorded_adapter_sha256'][:12]}, "
            f"but adapter_model.safetensors is {train_meta['adapter_sha256'][:12]}"
        )
    return None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def host_info() -> dict:
    info: dict = {"platform": platform.platform(), "cpu_count": os.cpu_count(), "gpu_used": False}
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (AttributeError, ValueError, OSError):  # not Linux
        info["ram_gb"] = None
    return info


def file_info(path: Path) -> dict:
    return {"file": path.name, "size_bytes": path.stat().st_size, "sha256": gates.sha256_file(path)}


# ---------------------------------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------------------------------


def inspect_adapter(adapter_dir: Path) -> dict:
    """Read and hash the adapter and read the train_meta.json next to it (checked by `adapter_problem` / main)."""
    config_path, weights_path = adapter_dir / "adapter_config.json", adapter_dir / "adapter_model.safetensors"
    missing = [p.name for p in (config_path, weights_path) if not p.is_file()]
    if missing:
        raise gates.GateError(f"{adapter_dir}: missing {missing} (--adapter-dir is the PEFT 'adapter' folder)")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_adapter_config(config)
    sha = gates.sha256_file(weights_path)

    candidates = (adapter_dir.parent / "train_meta.json", adapter_dir / "train_meta.json")
    train_meta_path = next((p for p in candidates if p.is_file()), None)
    train_meta = None
    if train_meta_path is not None:
        recorded = json.loads(train_meta_path.read_text(encoding="utf-8"))
        train_meta = {
            "file": train_meta_path.name,
            "sha256": gates.sha256_file(train_meta_path),
            "names_this_adapter": adapter_matches_train_meta(recorded, sha),
            "recorded_adapter_sha256": (recorded.get("adapter") or {}).get("sha256"),
            "adapter_sha256": sha,
            "base": recorded.get("base"),  # {"hf_base_for_gguf": ..., "hf_base_revision": ...}
        }
    return {
        "dir_name": adapter_dir.name,
        "file": weights_path.name,
        "sha256": sha,
        "size_bytes": weights_path.stat().st_size,
        "config": adapter_config_summary(config),
        "config_sha256": gates.sha256_file(config_path),
        "train_meta": train_meta,
        "target_modules": sorted(config["target_modules"]),
    }


def ensure_base(base_id: str, revision: str | None, commands: list[str]) -> tuple[Path, str | None]:
    """The full-precision base as a local HF folder at one commit; returns it with the commit sha that was downloaded.

    `revision` is a commit sha from `resolve_base_revision` (None only when the Hub could not be asked: the cache folder
    name, which is the commit, is recorded then).
    """
    from huggingface_hub import snapshot_download

    log(f"base {base_id} @ {revision or 'unknown revision (offline?)'}: downloading (about 6 GB)")
    commands.append(f"snapshot_download({base_id!r}, revision={revision!r}, allow_patterns={BASE_PATTERNS})")
    path = Path(snapshot_download(base_id, revision=revision, allow_patterns=BASE_PATTERNS))
    if not list(path.glob("*.safetensors")):
        raise gates.GateError(f"{path}: no *.safetensors weights after the download")
    downloaded = revision_from_snapshot_path(path) or revision
    if revision and downloaded != revision:
        raise gates.GateError(f"asked for {base_id} @ {revision} but the cache folder is {path.name}")
    return path, downloaded


def adapter_tensor_stats(adapter_file: Path) -> dict[str, int]:
    from safetensors import safe_open

    a = b = b_nonzero = 0
    with safe_open(str(adapter_file), framework="pt", device="cpu") as fh:
        keys = list(fh.keys())
        for key in keys:
            if ".lora_A." in key:
                a += 1
            elif ".lora_B." in key:
                b += 1
                b_nonzero += bool((fh.get_tensor(key) != 0).any())
    return {"tensors": len(keys), "lora_A": a, "lora_B": b, "lora_B_nonzero": b_nonzero}


def checkpoint_weight_map(folder: Path) -> dict[str, str] | None:
    """`weight_map` of the folder's safetensors index (tensor name -> shard), None for a single-file checkpoint."""
    index = folder / "model.safetensors.index.json"
    return json.loads(index.read_text(encoding="utf-8"))["weight_map"] if index.exists() else None


def checkpoint_tensor_names(folder: Path) -> set[str]:
    weight_map = checkpoint_weight_map(folder)
    if weight_map is not None:
        return set(weight_map)
    from safetensors import safe_open

    with safe_open(str(folder / "model.safetensors"), framework="pt", device="cpu") as fh:
        return set(fh.keys())


def read_tensor_sample(folder: Path, name: str) -> list[float]:
    """A strided sample of one weight tensor of a safetensors checkpoint folder, as Python floats (dtype-exact)."""
    from safetensors import safe_open

    shard = folder / shard_for(checkpoint_weight_map(folder), name)
    with safe_open(str(shard), framework="pt", device="cpu") as fh:
        if name not in set(fh.keys()):
            raise gates.GateError(f"{shard}: no tensor {name}")
        flat = fh.get_tensor(name).flatten().float()  # bf16 / fp16 -> float32 is exact
    return flat[:: sample_step(flat.numel())].tolist()


def compare_with_base(base_dir: Path, merged_dir: Path, num_layers: int, target_modules: Sequence[str]) -> dict:
    """Weight check on the SAVED merged folder against the base folder: what gets converted is what is compared.

    Both sides are rounded to float16 first (`as_fp16`): the base is stored in bfloat16, the merge in float16.
    """
    probes = probe_tensor_names(num_layers, target_modules)

    def diff(name: str) -> dict[str, Any]:
        return tensor_diff(as_fp16(read_tensor_sample(base_dir, name)), as_fp16(read_tensor_sample(merged_dir, name)))

    targets = {name: diff(name) for name in probes["targets"]}
    control = diff(probes["control"])
    problems, warnings = merge_check(targets, probes["control"], control)
    for warning in warnings:
        log("WARNING: " + warning)
    for name, d in {**targets, probes["control"]: control}.items():
        log(f"  {name}: {d['changed']}/{d['sampled']} sampled values differ, max |diff| {d['max_abs_diff']:.3g}")
    return {
        "sample_elements": SAMPLE_ELEMENTS, "compared_as": MERGE_DTYPE, "min_changed_fraction": MIN_CHANGED_FRACTION,
        "targets": targets, "control": {probes["control"]: control}, "problems": problems, "warnings": warnings,
    }  # fmt: skip


def load_base_fp16(base_dir: Path, torch: Any) -> tuple[Any, dict[str, str]]:
    """The base in float16 on the CPU; tries the `from_pretrained` keyword sets of load_kwarg_candidates in order."""
    from transformers import AutoModelForCausalLM

    errors = []
    for kwargs in load_kwarg_candidates(getattr(torch, MERGE_DTYPE)):
        try:
            model = AutoModelForCausalLM.from_pretrained(str(base_dir), **kwargs)
        except (TypeError, ValueError) as exc:  # this transformers version does not know a keyword
            errors.append(f"{sorted(kwargs)}: {exc}")
            continue
        loaded = {str(p.dtype) for p in model.parameters()}
        if loaded != {f"torch.{MERGE_DTYPE}"}:
            raise gates.GateError(f"the base loaded as {sorted(loaded)}, not {MERGE_DTYPE} (it would not fit in RAM)")
        return model, {k: str(v) for k, v in kwargs.items()}
    raise gates.GateError("could not load the base with any keyword set: " + " | ".join(errors))


def adopt_base_files(base_dir: Path, merged_dir: Path) -> dict:
    """Overwrite the merged folder's config / tokenizer files with the base's, byte for byte.

    Both GGUFs are then converted from the same config, tokenizer and chat template; whatever the installed transformers
    wrote (e.g. a different rope or dtype layout in config.json) cannot make the two models differ. The differences
    that the saved config had are recorded first.
    """
    names = base_files_to_copy(p.name for p in base_dir.iterdir() if p.is_file())
    saved = merged_dir / "config.json"
    differences = None
    if saved.exists() and (base_dir / "config.json").exists():
        base_config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
        differences = gates.config_diff(base_config, json.loads(saved.read_text(encoding="utf-8")))
    for name in names:
        shutil.copyfile(base_dir / name, merged_dir / name)
    return {
        "files": {name: gates.sha256_file(merged_dir / name) for name in names},
        "config_json_differences_before_copy": differences,
    }


def read_chat_template(folder: Path) -> str | None:
    """The chat template of a model folder: `chat_template.jinja`, else the `chat_template` key of tokenizer_config."""
    jinja = folder / "chat_template.jinja"
    if jinja.exists():
        return jinja.read_text(encoding="utf-8")
    config = folder / "tokenizer_config.json"
    template = json.loads(config.read_text(encoding="utf-8")).get("chat_template") if config.exists() else None
    return template if template is None or isinstance(template, str) else json.dumps(template, sort_keys=True)


def tokenizer_record(merged_dir: Path, adapter_dir: Path, base: str) -> dict:
    """Which tokenizer / chat template the merged folder (and so the fine-tuned GGUF) carries."""
    merged_sha = gates.sha256_file(merged_dir / "tokenizer.json")
    adapter_file = adapter_dir / "tokenizer.json"
    adapter_sha = gates.sha256_file(adapter_file) if adapter_file.exists() else None
    template = read_chat_template(merged_dir)
    return {
        "source": f"{base} (files copied from the base folder)",
        "tokenizer_json_sha256": merged_sha,
        "adapter_dir_tokenizer_json_sha256": adapter_sha,
        "same_as_adapter_dir": None if adapter_sha is None else adapter_sha == merged_sha,
        "chat_template_sha256": None if template is None else gates.sha256_text(template),
    }


def merge_stage(args: argparse.Namespace, adapter: dict, base_dir: Path, base_revision: str | None) -> dict:
    """Load the base in float16 on the CPU, apply the adapter, merge, save, adopt the base's files, check the result."""
    merged_dir = args.work_dir / "merged"
    key = {
        "adapter_sha256": adapter["sha256"], "base_id": args.base, "base_revision": base_revision,
        "dtype": MERGE_DTYPE,
    }  # fmt: skip
    marker = merged_dir / MERGE_INFO_NAME
    if marker.exists() and not args.force:
        info = json.loads(marker.read_text(encoding="utf-8"))
        if all(info.get(k) == v for k, v in key.items()):
            log(f"merge: {merged_dir} already holds this merge (checked), skipping")
            return info
        raise gates.GateError(f"{merged_dir} was made from another adapter/base; pass --force or use a new --work-dir")
    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    import torch
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer

    num_layers = int(json.loads((base_dir / "config.json").read_text(encoding="utf-8"))["num_hidden_layers"])
    targets = adapter["target_modules"]

    log(f"merge: loading the base in {MERGE_DTYPE} on the CPU")
    model, load_kwargs = load_base_fp16(base_dir, torch)

    log(f"merge: applying the adapter ({adapter['file']}, sha256 {adapter['sha256'][:12]})")
    file_stats = adapter_tensor_stats(Path(args.adapter_dir) / adapter["file"])
    peft_model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    layers = [m for m in peft_model.modules() if isinstance(m, LoraLayer)]
    lora_modules = len(layers)
    model_nonzero_b = sum(bool((m.lora_B["default"].weight != 0).any()) for m in layers)
    del layers  # they reference the base layers: drop them so that the model can be freed after the save
    problems = lora_load_problems(
        expected_modules=expected_lora_modules(num_layers, targets), found_modules=lora_modules,
        file_nonzero_b=file_stats["lora_B_nonzero"], model_nonzero_b=model_nonzero_b,
    )  # fmt: skip
    if problems:
        raise gates.GateError("adapter did not load: " + "; ".join(problems))

    log(f"merge: {lora_modules} LoRA modules loaded; merge_and_unload(safe_merge=True)")
    with torch.no_grad():
        merged = peft_model.merge_and_unload(safe_merge=True)
    log(f"merge: saving to {merged_dir}")
    merged.save_pretrained(str(merged_dir), max_shard_size="2GB")
    del merged, peft_model, model
    gc.collect()
    adopted = adopt_base_files(base_dir, merged_dir)

    log("merge: tensor names of the saved folder vs the base")
    base_names, merged_names = checkpoint_tensor_names(base_dir), checkpoint_tensor_names(merged_dir)
    name_problems = tensor_set_problems(base_names, merged_names)
    if name_problems:
        raise gates.GateError("the merged checkpoint has another tensor set than the base: " + "; ".join(name_problems))
    log("merge: weight check on the saved folder (merged vs base)")
    weights = compare_with_base(base_dir, merged_dir, num_layers, targets)
    if weights["problems"]:
        raise gates.GateError("; ".join(weights["problems"]))
    info = {
        **key,
        "device": "cpu",
        "load_kwargs": load_kwargs,
        "safe_merge": True,
        "lora_modules_merged": lora_modules,
        "adapter_tensors": file_stats,
        "lora_B_nonzero_in_model": model_nonzero_b,
        "tensor_counts": {"base": len(base_names), "merged": len(merged_names)},
        "weights_check": weights,
        "trained_against": TRAINED_AGAINST,
        "note": TRAIN_SERVE_NOTE,
        "adopted_from_base": adopted,
        "tokenizer": tokenizer_record(merged_dir, Path(args.adapter_dir), args.base),
        "merged_at": now(),
    }
    gates.dump_json(marker, info)  # written last: a folder without it is never reused
    return info


def ensure_llama_cpp(llama_dir: Path, ref: str, commands: list[str], allowed: Sequence[str] = ()) -> dict:
    """A llama.cpp checkout (cloned at `ref` when missing) with `llama-quantize` built (CPU only, Release).

    An EXISTING checkout at another commit than `ref` stops the run (`--allow-mismatch llama-cpp` waives it): both
    models must come from the commit the meta names, and a stale build there would not match a fresh clone.
    """
    cloned = False
    if not (llama_dir / "convert_hf_to_gguf.py").exists():
        if llama_dir.exists() and any(llama_dir.iterdir()):
            raise gates.GateError(f"{llama_dir} exists but is not a llama.cpp checkout; remove it or use another path")
        log(f"llama.cpp: fetching {ref} from {LLAMA_CPP_URL}")
        for cmd in git_fetch_commands(llama_dir, LLAMA_CPP_URL, ref):
            run(cmd, commands)
        cloned = True
    state = git_state(llama_dir)
    matches = ref_matches(state["commit"], ref)
    waived = enforce_match("llama-cpp", llama_cpp_problem(state["commit"], ref), allowed)
    binary = quantize_binary(llama_dir)
    built = False
    if not binary.exists():
        log("llama.cpp: building llama-quantize (CPU only; a few minutes)")
        run(cmake_configure_command(llama_dir), commands)
        run(cmake_build_command(llama_dir, os.cpu_count() or 2), commands)
        built = True
    if not binary.exists():
        raise gates.GateError(f"{binary} was not produced by the cmake build")
    return {
        "commit": state["commit"], "dirty": state["dirty"], "ref": ref, "ref_matches_commit": matches,
        "url": LLAMA_CPP_URL, "cloned_now": cloned, "built_now": built, "mismatch_waived": waived,
        "convert_hf_to_gguf.py_sha256": gates.sha256_file(llama_dir / "convert_hf_to_gguf.py"),
        "quantize_binary": str(binary), "cmake_flags": CMAKE_FLAGS,
    }  # fmt: skip


def publish(source: Path, target: Path) -> dict:
    """Copy a file to the output folder (Drive) through a .part name and verify the copy by sha256."""
    expected = gates.sha256_file(source)
    part = target.with_name(target.name + ".part")
    shutil.copyfile(source, part)
    actual = gates.sha256_file(part)
    if actual != expected:
        part.unlink()
        raise gates.GateError(f"the copy of {source.name} is corrupt (sha256 {actual[:12]}, expected {expected[:12]})")
    part.replace(target)
    return {"file": target.name, "size_bytes": target.stat().st_size, "sha256": expected}


def gguf_stage(args: argparse.Namespace, role: str, stem: str, folder: Path, binary: Path, commands: list[str]) -> dict:
    """HF folder -> f16 GGUF -> quantized GGUF in the work dir -> copied to --out; the f16 intermediate is dropped."""
    quant = normalize_quant(args.quant)
    f16, quantized = args.work_dir / f16_file_name(stem), args.work_dir / gguf_file_name(stem, quant)
    f16_info = None
    if not quantized.exists():
        if not f16.exists():
            log(f"{role}: convert_hf_to_gguf {folder.name} -> f16")
            part = f16.with_name(f16.name + ".part")
            run(convert_command(sys.executable, args.llama_cpp_dir, folder, part), commands)
            part.replace(f16)
        f16_info = file_info(f16)
        log(f"{role}: llama-quantize -> {quant}")
        part = quantized.with_name(quantized.name + ".part")
        run(quantize_command(binary, f16, part, quant, args.threads), commands)
        part.replace(quantized)
    else:
        log(f"{role}: {quantized.name} already in the work dir, skipping the conversion")
    if not args.keep_f16 and f16.exists():
        f16.unlink()
    with quantized.open("rb") as fh:
        template = find_gguf_string(fh.read(GGUF_HEAD_BYTES), "tokenizer.chat_template")
    log(f"{role}: copying {quantized.name} to {args.out}")
    published = publish(quantized, args.out / quantized.name)
    return {
        **published,
        "role": role,
        "source": "base snapshot" if role == "base" else "merged folder",
        "f16": f16_info,
        "chat_template_in_gguf": template is not None,
        "chat_template_sha256": gates.sha256_text(template) if template is not None else None,
    }


# ---------------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
        help="PEFT adapter folder: adapter_config.json, adapter_model.safetensors, tokenizer files",
    )
    p.add_argument("--out", type=Path, required=True, help="output folder (on Drive): two GGUFs + gguf_meta.json")
    p.add_argument("--base", default=DEFAULT_BASE, help="full-precision HF model the LoRA is merged into")
    p.add_argument(
        "--base-revision",
        help=f"HF commit (or branch / tag) of --base. Default for the default base: {DEFAULT_BASE_REVISION}, the one "
        "train_meta.json recorded; for another --base: its current main. The meta always records the commit sha",
    )
    p.add_argument("--quant", default=DEFAULT_QUANT, help="llama-quantize type for BOTH models")
    p.add_argument("--llama-cpp-dir", type=Path, default=Path("/content/llama.cpp"))
    p.add_argument("--llama-cpp-ref", default=LLAMA_CPP_REF, help="commit / tag to clone when --llama-cpp-dir is empty")
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/content/merge_work"),
        help="local scratch space (merged folder, f16 GGUFs); not Drive: it is slow for big files",
    )
    p.add_argument("--base-name", default=BASE_STEM, help="file stem of the base GGUF")
    p.add_argument("--ft-name", default=FT_STEM, help="file stem of the fine-tuned GGUF")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 2, help="threads for llama-quantize")
    p.add_argument("--keep-f16", action="store_true", help="keep the f16 GGUF intermediates in the work dir")
    p.add_argument("--force", action="store_true", help="redo the merge and overwrite existing outputs")
    p.add_argument(
        "--allow-mismatch",
        action="append",
        default=[],
        choices=MISMATCH_CHOICES,
        help="go on despite a failed consistency check (repeatable; the waived problem is recorded in the meta): "
        "adapter = train_meta.json names another adapter file; base-revision = train_meta.json recorded another "
        "base / revision; llama-cpp = an existing llama.cpp checkout is not at --llama-cpp-ref",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    # No GPU is needed; hiding it keeps torch from touching one if the runtime has it (torch is imported later).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    args = parse_args(argv)
    started, t0 = now(), time.monotonic()
    timings: dict[str, float] = {}  # seconds per stage, filled by `timed`
    quant = normalize_quant(args.quant)
    if args.base_name == args.ft_name:
        raise gates.GateError("--base-name and --ft-name must differ")
    adapter = inspect_adapter(args.adapter_dir)
    waived: dict[str, str] = {}  # consistency problems the author chose to go on with (--allow-mismatch)
    if problem := enforce_match("adapter", adapter_problem(adapter["train_meta"]), args.allow_mismatch):
        waived["adapter"] = problem
    if warning := gpu_runtime_warning():
        log("WARNING: " + warning)
    missing = missing_modules(REQUIRED_MODULES)
    if missing:
        raise gates.GateError(f"missing Python packages {missing}: pip install -r training/requirements-merge.txt")
    if problem := torchao_problem(installed_version("torchao")):
        raise gates.GateError(problem)
    stems = (args.base_name, args.ft_name)
    finals = [args.out / gguf_file_name(stem, quant) for stem in stems]
    exists = [p.name for p in finals if p.exists()]
    if exists and not args.force:
        raise gates.GateError(f"{args.out} already has {exists}: pass --force to overwrite or use a new --out")

    # Every check that needs no download and no compute comes BEFORE anything is deleted or written, and before the
    # long stages: a run that is going to stop must not have removed the previous outputs or spent ten minutes first.
    train_base = (adapter["train_meta"] or {}).get("base")  # what the training run recorded about its HF base
    revision = resolve_base_revision(args.base, args.base_revision)
    if problem := enforce_match(
        "base-revision", base_revision_problem(train_base, args.base, revision), args.allow_mismatch
    ):
        waived["base-revision"] = problem
    if problem := enforce_match(
        "llama-cpp", existing_llama_cpp_problem(args.llama_cpp_dir, args.llama_cpp_ref), args.allow_mismatch
    ):
        waived["llama-cpp"] = problem

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    check_free_space(shutil.disk_usage(args.work_dir).free)
    if args.force:  # start from scratch: the merge is redone by merge_stage, the GGUFs here
        for stem in stems:
            for name in (f16_file_name(stem), gguf_file_name(stem, quant)):
                (args.work_dir / name).unlink(missing_ok=True)
        for final in finals:
            final.unlink(missing_ok=True)

    commands: list[str] = []
    with timed(timings, "download_base"):
        base_dir, base_revision = ensure_base(args.base, revision, commands)
    if revision is None and (  # the Hub could not be asked before: the downloaded commit is known only now
        problem := enforce_match(
            "base-revision", base_revision_problem(train_base, args.base, base_revision), args.allow_mismatch
        )
    ):
        waived["base-revision"] = problem
    with timed(timings, "merge"):
        merge = merge_stage(args, adapter, base_dir, base_revision)
    with timed(timings, "llama_cpp"):
        llama = ensure_llama_cpp(args.llama_cpp_dir, args.llama_cpp_ref, commands, args.allow_mismatch)
    if llama["mismatch_waived"]:
        waived["llama-cpp"] = llama["mismatch_waived"]

    outputs: dict[str, Any] = {}
    for role, stem, folder in (("base", args.base_name, base_dir), ("ft", args.ft_name, args.work_dir / "merged")):
        with timed(timings, f"gguf_{role}"):
            outputs[role] = gguf_stage(args, role, stem, folder, Path(llama["quantize_binary"]), commands)
    templates = {outputs[role]["chat_template_sha256"] for role in ("base", "ft")}
    outputs["same_chat_template"] = None if None in templates else len(templates) == 1
    if outputs["same_chat_template"] is not True:
        log(f"WARNING: GGUF chat templates: same={outputs['same_chat_template']} (None = not found in the GGUF header)")

    base_config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    meta = build_gguf_meta(
        started_at=started, finished_at=now(), timings_s={**timings, "total": round(time.monotonic() - t0, 1)},
        code=code_provenance(),
        environment={"versions": package_versions(), "host": host_info()},
        base={
            "id": args.base, "revision": base_revision, "num_hidden_layers": base_config.get("num_hidden_layers"),
            "dtype_on_disk": base_config.get("dtype") or base_config.get("torch_dtype"),
            "trained_against": TRAINED_AGAINST,
        },
        adapter={k: v for k, v in adapter.items() if k != "target_modules"},
        merge={k: v for k, v in merge.items() if k != "tokenizer"},
        tokenizer=merge["tokenizer"],
        llama_cpp=llama,
        quantization={
            "type": quant, "intermediate": "f16", "imatrix": None, "threads": args.threads, "kept_f16": args.keep_f16,
            "applied_to": [args.base_name, args.ft_name], "same_pipeline": True,
        },
        outputs=outputs, commands=commands, waived_mismatches=waived,
    )  # fmt: skip
    gates.dump_json(args.out / META_NAME, meta)
    log(f"wrote {args.out / META_NAME}")
    print(next_steps_text(args.out, [p.name for p in finals] + [META_NAME], quant, args.ft_name))


if __name__ == "__main__":
    try:
        main()
    except gates.GateError as exc:
        sys.exit(f"CHECK FAILED: {exc}")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"COMMAND FAILED (exit {exc.returncode}): {shlex.join(str(a) for a in exc.cmd)}")
