from pathlib import Path

from keeper.experiments import ExperimentRegistry
from keeper.llm import ModelFactory
from keeper.prompts import PromptSet

BACKEND = Path(__file__).resolve().parents[1]


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
