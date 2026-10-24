"""QLoRA fine-tuning of the Keeper narrator (Phase 4) with Unsloth + TRL. Runs on Colab / Kaggle (one T4), not locally.

    python training/train_unsloth.py \
        --train artifacts/sft/coc7e-v1/train.jsonl --val artifacts/sft/coc7e-v1/val.jsonl \
        --out /content/drive/MyDrive/gm-assistant/out/keeper-qwen25-3b-sft-v1

Order of work (everything before `trainer.train()` is a gate that stops the run, see docs/critique.md C3):
  1. train/val sha256 must equal the ones in manifest.json (refuse on mismatch);
  2. every example is rendered with the tokenizer's own chat template, so the data's system prompt is used (Qwen's
     template injects "You are Qwen..." only when no system message is given), and the EXACT tokenizer measures it:
     any example longer than --max-len stops the run (TRL would silently cut off the answer);
  3. masking: `train_on_responses_only` (tags by model family), then the labels are overwritten so that ONLY THE FINAL
     assistant reply is supervised. The system prompt holds the knowledge base / rolls of the current turn only, so the
     history replies are not valid targets under it. Masking gate: each example has exactly one supervised span, it
     ends at the last token and decodes to the final reply + EOS; the tokenized example ends with exactly one EOS;
     the labels the data collator really feeds the loss equal the gated ones;
  4. training with checkpoints every --save-steps into <out>/checkpoints (Colab sessions disconnect: re-run the same
     command with --resume);
  5. the LoRA adapter (the LAST step, not the best one) goes to <out>/adapter, `train_meta.json` to <out>, and, with
     --llama-cpp-dir, the adapter is converted to GGUF (<out>/adapter.gguf) for Ollama's ADAPTER.

`--check-data` runs steps 1-2 only (needs `transformers`, no GPU, no unsloth): use it before paying for a GPU session.
The pure checks live in src/backend/sft/gates.py (stdlib only, unit-tested); this file is the GPU glue.
Written against the pins in training/requirements-train.txt; it was not run when it was written.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

# Must be set before torch initialises CUDA: fewer fragmentation OOMs on long sequences.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # a copy of gates.py next to this script (upload without the repo)
sys.path.insert(0, str(HERE.parent / "src" / "backend"))  # the repo checkout
try:
    from sft import gates  # noqa: E402
except ImportError:  # no repo, gates.py copied next to the script
    import gates  # noqa: E402

DEFAULT_BASE = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"  # 4-bit weights to train on (small download)
DEFAULT_HF_BASE = "Qwen/Qwen2.5-3B-Instruct"  # full-precision twin: its config is what the GGUF conversion needs
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PACKAGES = ["torch", "transformers", "unsloth", "unsloth_zoo", "trl", "peft", "bitsandbytes", "accelerate", "datasets"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=DEFAULT_BASE, help="model to train on (loaded in 4 bit)")
    p.add_argument("--hf-base", default=DEFAULT_HF_BASE, help="full-precision HF id of the same model (GGUF export)")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--manifest", type=Path, help="default: manifest.json next to --train")
    p.add_argument("--out", type=Path, required=True, help="output dir; put it on Drive so it survives a disconnect")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--max-len",
        type=int,
        default=8192,
        help="token budget per example (serving history is up to 6000 tokens + system prompt + knowledge base); "
        "the length gate stops the run if any example is longer. Lower it only if memory forces you to",
    )
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=1, help="per device")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in <out>/checkpoints")
    p.add_argument("--check-data", action="store_true", help="run the data gates (steps 1-2) and exit")
    p.add_argument("--skip-memory-probe", action="store_true",
                   help="do not run one training step on the longest example before training")
    p.add_argument("--llama-cpp-dir", type=Path, help="llama.cpp checkout: also convert the adapter to GGUF")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------------------------------------------------
# Data gates (no GPU)
# ---------------------------------------------------------------------------------------------------------------------


def verify_data(args: argparse.Namespace) -> tuple[dict, dict[str, str]]:
    manifest_path = args.manifest or args.train.parent / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"manifest not found: {manifest_path} (pass --manifest)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = gates.verify_data_hashes(manifest, {"train.jsonl": args.train, "val.jsonl": args.val})
    print(f"manifest ok: train {hashes['train.jsonl'][:12]}, val {hashes['val.jsonl'][:12]}")
    return manifest, hashes


def render_split(tokenizer, rows: list[dict], split: str, max_len: int) -> tuple[list[str], list[int], dict]:
    """Render with the tokenizer's chat template, then run the structure, system-prompt and length gates."""
    gates.check_messages(rows, split)
    texts = [
        gates.finalize_text(
            tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False),
            tokenizer.bos_token,
            tokenizer.eos_token,
        )
        for row in rows
    ]
    gates.check_rendered(rows, texts, tokenizer.eos_token, split)
    lengths = [len(tokenizer(text)["input_ids"]) for text in texts]
    stats = gates.check_lengths(lengths, max_len, split)
    print(f"length gate ok [{split}]: {stats} (max_len={max_len})")
    return texts, lengths, stats


# ---------------------------------------------------------------------------------------------------------------------
# Environment facts for train_meta.json
# ---------------------------------------------------------------------------------------------------------------------


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def hardware_info(torch) -> dict:
    info: dict = {
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu=props.name,
            vram_gb=round(props.total_memory / 2**30, 1),
            capability=list(torch.cuda.get_device_capability(0)),
        )
    return info


def hf_revision(repo_id: str) -> str | None:
    """Commit sha of the repo's main branch on the Hub (None when offline: the meta records it rather than failing)."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo_id).sha
    except Exception:  # noqa: BLE001
        return None


def base_revision(model, base: str) -> str | None:
    revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    return revision or hf_revision(base)


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
    """Which code produced the adapter: hashes of this script and of gates.py, plus the git state if there is one."""
    return {
        "train_unsloth.py_sha256": gates.sha256_file(Path(__file__)),
        "gates.py_sha256": gates.sha256_file(Path(gates.__file__)),
        "git": git_state(HERE),
    }


def template_string(tokenizer) -> str:
    template = tokenizer.chat_template
    return template if isinstance(template, str) else json.dumps(template, sort_keys=True)


def memory_probe(model, trainer, train_lens: list[int], torch) -> dict:
    """One real training step (trainer.training_step) on the LONGEST train example: peak memory depends on that length.

    It goes through the trainer, not model(**batch): Unsloth's memory savings (chunked cross-entropy without the full
    vocabulary logits, its gradient checkpointing, mixed precision via the accelerator) live on the trainer's path, so
    a bare forward pass overstates the memory a real step needs by several GB.
    """
    if not torch.cuda.is_available():
        return {"skipped": "no CUDA"}
    longest = max(range(len(train_lens)), key=train_lens.__getitem__)
    batch = trainer.data_collator([trainer.train_dataset[longest]])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        loss = trainer.training_step(model, batch)
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        sweep = memory_sweep(model, trainer, batch, torch)
        sys.exit(
            f"OUT OF MEMORY on a real training step with the longest train example ({train_lens[longest]} tokens, "
            f"batch size 1). Peak memory of the same step on the example's last N tokens: {sweep}. "
            f"GPU total {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB. Send this line to the "
            "maintainer; see training/README.md."
        )
    peak = torch.cuda.max_memory_allocated()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return {
        "example_index": longest,
        "tokens": train_lens[longest],
        "loss": float(loss),
        "max_memory_allocated_gb": round(peak / 2**30, 2),
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
    }


def memory_sweep(model, trainer, batch: dict, torch, lengths: tuple[int, ...] = (1024, 2048, 3072)) -> dict:
    """Peak GB of one training step on the last N tokens of `batch` (keeps the supervised final reply): the growth
    with N shows whether attention (quadratic) or logits/activations (linear) take the memory."""
    seq = batch["input_ids"].shape[-1]
    out = {}
    for n in lengths:
        if n >= seq:
            continue
        cut = {k: (v[..., -n:] if hasattr(v, "shape") and v.shape[-1] == seq else v) for k, v in batch.items()}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            trainer.training_step(model, cut)
            out[n] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        except torch.cuda.OutOfMemoryError:
            out[n] = "OOM"
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------------------------------------------------
# GGUF export
# ---------------------------------------------------------------------------------------------------------------------


def export_gguf(llama_cpp_dir: Path, adapter_dir: Path, hf_base: str, out_file: Path) -> dict:
    """`convert_lora_to_gguf.py --base <HF base config dir>`; failure is reported, not fatal (the adapter is saved)."""
    script = llama_cpp_dir / "convert_lora_to_gguf.py"
    if not script.exists():
        return {"error": f"{script} not found"}
    revision = hf_revision(hf_base)
    try:
        from huggingface_hub import snapshot_download

        # only the config / tokenizer files are needed, not the weights; pinned to the revision recorded below
        base_dir = snapshot_download(
            hf_base, revision=revision, allow_patterns=["*.json", "merges.txt", "*.model", "*.tiktoken"]
        )
        cmd = [sys.executable, str(script), "--base", base_dir, "--outfile", str(out_file)]
        cmd += ["--outtype", "f16", str(adapter_dir)]
        subprocess.run(cmd, check=True)
    except Exception as exc:  # noqa: BLE001 - recorded in the meta, the run output is already on disk
        return {"error": f"{type(exc).__name__}: {exc}", "hf_base_revision": revision}
    return {
        "path": out_file.name,
        "sha256": gates.sha256_file(out_file),
        "outtype": "f16",
        "converted_from": hf_base,
        "hf_base_revision": revision,
        "llama_cpp": git_state(llama_cpp_dir),
    }


# ---------------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------------


def check_data_only(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    verify_data(args)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    for split, path in (("train", args.train), ("val", args.val)):
        render_split(tokenizer, gates.load_jsonl(path), split, args.max_len)
    # The real run uses the unsloth-loaded tokenizer; its template sha256 is recorded in train_meta.json and compared.
    print(f"chat template sha256 (transformers AutoTokenizer): {gates.sha256_text(template_string(tokenizer))}")
    print("data gates passed")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.check_data:
        check_data_only(args)
        return

    tags = gates.family_for_base(args.base)
    manifest, hashes = verify_data(args)
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_dir, adapter_dir = args.out / "checkpoints", args.out / "adapter"
    latest = gates.latest_checkpoint(ckpt_dir)
    has_checkpoints = latest is not None

    run_config = {
        "base": args.base, "train_sha256": hashes["train.jsonl"], "val_sha256": hashes["val.jsonl"], "seed": args.seed,
        "max_len": args.max_len, "r": args.r, "alpha": args.alpha, "lr": args.lr, "epochs": args.epochs,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum, "warmup_steps": args.warmup_steps,
    }  # fmt: skip
    config_path = args.out / "run_config.json"
    if has_checkpoints and not args.resume:
        sys.exit(f"{ckpt_dir} already has checkpoints: pass --resume to continue, or use a new --out")
    if args.resume and has_checkpoints:
        # run_config.json is also copied into every checkpoint, so copying only checkpoints/ (Kaggle) is enough
        saved_path = next((p for p in (config_path, latest / "run_config.json") if p.exists()), None)
        if saved_path is None:
            sys.exit(f"--resume: no run_config.json in {args.out} or in {latest.name}; cannot tell which run this is")
        diff = gates.config_diff(json.loads(saved_path.read_text(encoding="utf-8")), run_config)
        if diff:
            sys.exit("--resume with a different configuration would mix two runs: " + "; ".join(diff))
    elif args.resume:
        print("--resume: no checkpoint found, starting from scratch")
    gates.dump_json(config_path, run_config)
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Unsloth patches transformers / trl / peft on import: it must come first.
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    # isort: split
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base, max_seq_length=args.max_len, dtype=None, load_in_4bit=True  # dtype=None: fp16 on a T4
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.r, lora_alpha=args.alpha, lora_dropout=0, bias="none", target_modules=LORA_TARGETS,
        use_gradient_checkpointing="unsloth", random_state=args.seed,
    )  # fmt: skip

    train_rows, val_rows = gates.load_jsonl(args.train), gates.load_jsonl(args.val)
    train_texts, train_lens, train_len_stats = render_split(tokenizer, train_rows, "train", args.max_len)
    val_texts, val_lens, val_len_stats = render_split(tokenizer, val_rows, "val", args.max_len)
    print("first rendered example:\n" + train_texts[0][:1500] + "\n---")

    # Only a "text" column: with a "messages" column TRL would apply its own chat template and ignore our rendering.
    train_ds, val_ds = Dataset.from_dict({"text": train_texts}), Dataset.from_dict({"text": val_texts})

    use_bf16 = is_bfloat16_supported()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(ckpt_dir),
            dataset_text_field="text",
            max_length=args.max_len,
            packing=False,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=args.warmup_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            seed=args.seed,
            data_seed=args.seed,
            report_to="none",
        ),
    )
    # train_on_responses_only unmasks EVERY assistant span, history included. The history replies were written for an
    # earlier turn's knowledge base / rolls, which the system prompt (current turn only) does not contain, so we keep
    # only the final reply: same tags, but labels recomputed from the LAST response tag onwards.
    trainer = train_on_responses_only(trainer, instruction_part=tags.instruction_part, response_part=tags.response_part)
    response_ids = tokenizer(tags.response_part, add_special_tokens=False)["input_ids"]

    def final_reply_labels(example: dict) -> dict:
        return {"labels": gates.mask_all_but_last_response(example["input_ids"], response_ids)}

    keep = ["input_ids", "attention_mask", "labels"]
    for attr in ("train_dataset", "eval_dataset"):
        ds = getattr(trainer, attr).map(final_reply_labels)
        setattr(trainer, attr, ds.select_columns([c for c in keep if c in ds.column_names]))

    # Gates on what the trainer will really see.
    eos_id = tokenizer.eos_token_id

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False)

    gate_report = {"train": {"length": train_len_stats}, "val": {"length": val_len_stats}}
    seen = (
        ("train", trainer.train_dataset, train_lens, train_rows),
        ("val", trainer.eval_dataset, val_lens, val_rows),
    )
    gated_labels: dict[str, list[list[int]]] = {}
    for split, ds, expected_lens, rows in seen:
        input_ids = [list(ids) for ids in ds["input_ids"]]
        labels = [list(row) for row in ds["labels"]]
        if [len(ids) for ids in input_ids] != expected_lens:  # shorter = truncated, longer = something was appended
            sys.exit(f"{split}: the trainer's token counts differ from the length gate's measurement")
        gates.check_token_ends(input_ids, eos_id, split)
        gate_report[split]["masking"] = gates.check_masking(labels, split)
        replies = [row["messages"][-1]["content"] for row in rows]
        gates.check_final_response_only(labels, input_ids, decode, replies, tokenizer.eos_token, split)
        gated_labels[split] = labels
        print(f"masking gate ok [{split}] (final reply only): {gate_report[split]['masking']}")

    # The labels column is only a claim; the collator decides what the loss sees. The first batch of the eval
    # dataloader (sequential sampler) holds the first gated val examples. If the trainer's collator recomputes labels
    # or fails, fall back to an explicit Seq2Seq collator (pads labels with -100, keeps ours) and check again.
    def check_collator() -> None:
        batch_labels = next(iter(trainer.get_eval_dataloader()))["labels"].tolist()
        gates.collator_labels_match(batch_labels, gated_labels["val"][: len(batch_labels)])

    collator_report = {"type": type(trainer.data_collator).__name__, "fallback": False}
    try:
        check_collator()
    except Exception as exc:  # noqa: BLE001 - a GateError, or any error raised by the trainer's own collator
        print(f"WARNING: collator check failed ({type(exc).__name__}: {exc}); installing DataCollatorForSeq2Seq")
        trainer.data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
        check_collator()
        collator_report = {"type": type(trainer.data_collator).__name__, "fallback": True}
    gate_report["collator"] = collator_report
    print(f"collator gate ok: {collator_report}")

    first_labels = [t for t in gated_labels["train"][0] if t != gates.IGNORE_INDEX]
    supervised_preview = decode(first_labels)
    print("supervised tokens of example 0 (must be the final answer only):\n" + supervised_preview[:600] + "\n---")

    template = template_string(tokenizer)
    template_sha = gates.sha256_text(template)
    plain_sha = None
    try:  # the template of a plain transformers tokenizer, which --check-data used
        plain_sha = gates.sha256_text(template_string(AutoTokenizer.from_pretrained(args.base)))
    except Exception as exc:  # noqa: BLE001
        print(f"note: could not load the plain tokenizer to compare chat templates: {exc}")
    if plain_sha is not None and plain_sha != template_sha:
        print("WARNING: the unsloth tokenizer's chat template differs from transformers'; --check-data was not exact")

    class SaveRunConfig(TrainerCallback):
        """Copy run_config.json into each checkpoint so that a checkpoint dir alone identifies its run."""

        def on_save(self, train_args, state, control, **kwargs):
            folder = Path(train_args.output_dir) / f"checkpoint-{state.global_step}"
            if folder.is_dir():
                shutil.copyfile(config_path, folder / "run_config.json")

    trainer.add_callback(SaveRunConfig())

    meta_base = {
        "base": {"id": args.base, "revision": base_revision(model, args.base), "hf_base_for_gguf": args.hf_base,
                 "hf_base_revision": hf_revision(args.hf_base), "family": tags.family, "load_in_4bit": True,
                 "bf16": use_bf16},
        "data": {
            "train_file": args.train.name, "val_file": args.val.name, "train_sha256": hashes["train.jsonl"],
            "val_sha256": hashes["val.jsonl"], "train_examples": len(train_rows), "val_examples": len(val_rows),
            "manifest": manifest,
        },
        "hyperparameters": {
            **{k: v for k, v in run_config.items() if k not in ("base", "train_sha256", "val_sha256", "seed")},
            "target_modules": LORA_TARGETS, "lora_dropout": 0, "optim": "adamw_8bit", "lr_scheduler": "cosine",
            "weight_decay": 0.01, "response_tags": [tags.instruction_part, tags.response_part],
            "loss_on": "final assistant reply only (history replies masked)",
            "eval_steps": args.eval_steps, "save_steps": args.save_steps,
            "resumed": bool(args.resume and has_checkpoints),
        },
        "seed": args.seed,
        "versions": package_versions(),
        "hardware": hardware_info(torch),
        "chat_template": {
            "string": template, "sha256": template_sha, "sha256_transformers_autotokenizer": plain_sha,
            "same_as_transformers": None if plain_sha is None else plain_sha == template_sha,
            "source": f"unsloth tokenizer of {args.base}", "add_generation_prompt": False,
            "bos_token": tokenizer.bos_token, "eos_token": tokenizer.eos_token,
            "first_rendered_example": train_texts[0][:3000], "supervised_preview": supervised_preview[:600],
        },
        "gates": gate_report,
    }  # fmt: skip

    if args.skip_memory_probe:
        gate_report["memory_probe"] = {"skipped": "--skip-memory-probe"}
    else:
        gate_report["memory_probe"] = memory_probe(model, trainer, train_lens, torch)
    print(f"memory probe: {gate_report['memory_probe']}")

    result = trainer.train(resume_from_checkpoint=True if args.resume and has_checkpoints else None)
    final_eval = trainer.evaluate()
    losses = gates.summarize_losses(trainer.state.log_history)
    losses.update(train_loss=result.metrics.get("train_loss"), final_eval_loss=final_eval.get("eval_loss"))
    print(f"losses: {({k: v for k, v in losses.items() if k != 'eval_loss_curve'})}")
    if not losses["eval_loss_decreased"]:
        print("WARNING: eval loss did not go down between the first and the last evaluation; inspect the curve")

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    adapter_file = adapter_dir / "adapter_model.safetensors"
    adapter_sha = gates.sha256_file(adapter_file) if adapter_file.exists() else None
    extra = {
        "adapter": {
            "dir": adapter_dir.name,
            "sha256": adapter_sha,
            "selection": "last step (no load_best_model_at_end)",
            "saved_at_step": trainer.state.global_step,
            "eval_loss_best": losses["eval_loss_best"],
            "eval_loss_best_step": losses["eval_loss_best_step"],
        },
        "code": code_provenance(),
        "gguf": None,
    }
    meta_path = args.out / "train_meta.json"
    gates.dump_json(meta_path, gates.build_meta(**meta_base, losses=losses, extra=extra))

    if args.llama_cpp_dir:
        extra["gguf"] = export_gguf(args.llama_cpp_dir, adapter_dir, args.hf_base, args.out / "adapter.gguf")
        gates.dump_json(meta_path, gates.build_meta(**meta_base, losses=losses, extra=extra))
        print(f"gguf: {extra['gguf']}")
    else:
        print("GGUF not produced (no --llama-cpp-dir); see training/README.md, step 'adapter to GGUF'")
    print(f"done: {meta_path}")


if __name__ == "__main__":
    try:
        main()
    except gates.GateError as exc:
        sys.exit(f"GATE FAILED: {exc}")
