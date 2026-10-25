from pathlib import Path

from evals.harness import sha16
from keeper.experiments import ExperimentRegistry
from keeper.llm import ModelFactory, ModelProfile
from keeper.prompts import PromptSet

BACKEND = Path(__file__).resolve().parents[1]
# The keys every profile dump in a recorded run config has (artifacts/eval/runs/*/config.json, SFT inputs, run logs).
LEGACY_PROFILE_KEYS = {"provider", "model", "temperature", "max_tokens", "thinking_level", "context_window", "num_ctx",
                       "finetuned", "base", "quant"}


def test_profiles_and_arms_load():
    factory = ModelFactory.from_yaml(BACKEND / "config" / "models.yaml")
    arms = ExperimentRegistry.from_dir(BACKEND / "experiments", "play-gemini")
    spec = arms.resolve()
    assert factory.profile(spec.generator).provider == "google_genai"
    assert factory.profile(spec.aux).thinking_level != "minimal"  # rejected by gemini-3.8-flash
    assert len(spec.spec_hash()) == 16


def test_rate_limiter_does_not_stall_first_turn():
    limiter = ModelFactory({}, google_rpm=6)._limiters["google_genai"]
    assert limiter.acquire(blocking=False) and limiter.acquire(blocking=False)  # adjudicate + narrate
    assert not limiter.acquire(blocking=False)


def test_prompt_hash_is_line_ending_independent(tmp_path):
    (tmp_path / "a.md.j2").write_bytes(b"line1\r\nline2\r\n")
    (tmp_path / "b.md.j2").write_bytes(b"line1\nline2\n")
    prompts = PromptSet(tmp_path)
    assert prompts.hash("a") == prompts.hash("b")


def test_snapshot_leaves_out_unset_model_card():
    plain = ModelProfile(provider="ollama", model="m")
    assert plain.model_card is None and set(plain.snapshot()) == LEGACY_PROFILE_KEYS
    assert plain.snapshot() == plain.model_dump(exclude={"model_card"})
    tuned = ModelProfile(provider="ollama", model="m", finetuned=True, model_card="training/model_cards/x.json")
    assert tuned.snapshot() == {**plain.snapshot(), "finetuned": True, "model_card": "training/model_cards/x.json"}


def test_profiles_of_recorded_runs_dump_as_before():
    factory = ModelFactory.from_yaml(BACKEND / "config" / "models.yaml")
    for name, profile in factory.profiles.items():
        assert set(profile.snapshot()) == LEGACY_PROFILE_KEYS | ({"model_card"} if profile.model_card else set()), name
    # The profiles of the phase3 runs: changing any of them (or a ModelProfile default) would change those runs'
    # fingerprints, which must not happen silently. Value computed before model cards were introduced.
    recorded = ["gemini-aux", "gemini-flash", "gemini-judge", "qwen25-3b-ollama", "qwen25-3b-ollama-aux"]
    assert sha16({name: factory.profile(name).snapshot() for name in recorded}) == "b58dc834d5299cc3"
