"""Orchestration smoke test for run_experiment.py: no GPU, no real model.

compute_soo_distance() and classify_scenario() are monkeypatched with fakes
(one of which fails on purpose) so this exercises only the orchestration
logic -- preflight, CSV writing, incremental flush, error handling, and
resume -- against a 3-scenario slice in a temp results dir. Run directly:

    python src/test_run_experiment.py
"""

import contextlib
import csv
import shutil
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# This dev machine has no torch/transformers (the real run is on a Colab
# GPU). Stub them just enough to satisfy import-time attribute access in
# hooks.py/soo_distance.py/behavior.py -- nothing in this test ever calls
# into them for real, since load_model/compute_soo_distance/classify_scenario
# are all monkeypatched below.
try:
    import torch  # noqa: F401
except ImportError:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.no_grad = contextlib.nullcontext
    torch_stub.manual_seed = lambda seed: None
    functional_stub = types.ModuleType("torch.nn.functional")
    functional_stub.mse_loss = lambda a, b: None
    nn_stub = types.ModuleType("torch.nn")
    nn_stub.functional = functional_stub
    torch_stub.nn = nn_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_stub
    sys.modules["torch.nn.functional"] = functional_stub

    transformers_stub = types.ModuleType("transformers")

    class _UnusedInTest:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("stub: should not be called in orchestration smoke test")

    transformers_stub.AutoModelForCausalLM = _UnusedInTest
    transformers_stub.AutoTokenizer = _UnusedInTest
    transformers_stub.BitsAndBytesConfig = _UnusedInTest
    sys.modules["transformers"] = transformers_stub

from data.generate_scenarios import get_scaled_scenarios
from src import run_experiment as rx


@dataclass
class FakeSOOResult:
    primary_distance: float
    exploratory_distances: dict


@dataclass
class FakeBehaviorResult:
    label: str
    detected_room: str
    reason: str
    raw_response_text: str


def fake_compute_soo_distance(model, tokenizer, self_prompt, other_prompt, **kwargs):
    return FakeSOOResult(primary_distance=0.1, exploratory_distances={10: 0.1, 15: 0.1, 20: 0.1, 25: 0.1})


def make_fake_classify_scenario(fail_on_id: str):
    def fake_classify_scenario(model, tokenizer, scenario):
        if scenario.scenario_id == fail_on_id:
            raise RuntimeError("simulated generation failure")
        return FakeBehaviorResult(
            label="honest", detected_room="Room A", reason="fake", raw_response_text="I would recommend Room A."
        )

    return fake_classify_scenario


def main():
    tmp_dir = Path(tempfile.mkdtemp(prefix="run_experiment_smoke_"))
    scenarios = get_scaled_scenarios()[:3]
    fail_id = scenarios[1].scenario_id

    orig = {
        "SCENARIOS_PATH": rx.SCENARIOS_PATH,
        "RESULTS_DIR": rx.RESULTS_DIR,
        "RESULTS_CSV": rx.RESULTS_CSV,
        "ERRORS_LOG": rx.ERRORS_LOG,
        "METADATA_JSON": rx.METADATA_JSON,
        "EXPECTED_SCENARIO_COUNT": rx.EXPECTED_SCENARIO_COUNT,
        "load_model": rx.load_model,
        "compute_soo_distance": rx.compute_soo_distance,
        "classify_scenario": rx.classify_scenario,
    }

    try:
        rx.SCENARIOS_PATH = tmp_dir / "scenarios.json"
        rx.RESULTS_DIR = tmp_dir / "results"
        rx.RESULTS_CSV = rx.RESULTS_DIR / "experiment_results.csv"
        rx.ERRORS_LOG = rx.RESULTS_DIR / "experiment_errors.log"
        rx.METADATA_JSON = rx.RESULTS_DIR / "experiment_metadata.json"
        rx.EXPECTED_SCENARIO_COUNT = len(scenarios)

        import json
        from dataclasses import asdict

        rx.SCENARIOS_PATH.write_text(json.dumps([asdict(s) for s in scenarios]))

        rx.load_model = lambda: (None, None)
        rx.compute_soo_distance = fake_compute_soo_distance
        rx.classify_scenario = make_fake_classify_scenario(fail_id)

        # --- Run 1: scenario[1] fails, scenario[0] and scenario[2] complete ---
        rx.run()

        with rx.RESULTS_CSV.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2, f"expected 2 completed rows after run 1, got {len(rows)}"
        ids_after_run1 = {r["scenario_id"] for r in rows}
        assert fail_id not in ids_after_run1, "failed scenario must not get a completed row"
        assert rx.ERRORS_LOG.exists(), "errors log must be written on failure"
        assert fail_id in rx.ERRORS_LOG.read_text()
        assert rx.METADATA_JSON.exists(), "metadata json must be written"
        print("[PASS] run 1: error handling + incremental save")

        # --- Run 2 (resume): fix the fake failure, only scenario[1] should run ---
        rx.classify_scenario = make_fake_classify_scenario(fail_on_id="__none__")
        rx.run()

        with rx.RESULTS_CSV.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3, f"expected 3 completed rows after resume, got {len(rows)}"
        ids_after_run2 = [r["scenario_id"] for r in rows]
        assert len(set(ids_after_run2)) == 3, "resume must not duplicate completed rows"
        assert set(ids_after_run2) == {s.scenario_id for s in scenarios}
        print("[PASS] run 2: resume skips completed, retries failed, no duplicates")

        # --- Preflight: wrong scenario count must stop before model load ---
        rx.EXPECTED_SCENARIO_COUNT = 999
        try:
            rx.run()
            raise AssertionError("expected SystemExit on scenario count mismatch")
        except SystemExit:
            print("[PASS] preflight rejects wrong scenario count")

        print("\nAll orchestration smoke tests passed.")
    finally:
        for key, value in orig.items():
            setattr(rx, key, value)
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
