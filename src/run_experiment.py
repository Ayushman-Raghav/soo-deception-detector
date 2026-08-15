"""Orchestration only: run the full 54-scenario set through both pipeline
halves (soo_distance.py + behavior.py) and save one combined row per
scenario to results/experiment_results.csv.

No experimental logic lives here -- compute_soo_distance() and
classify_scenario() are called unmodified. This file just sequences them,
handles errors per-scenario, and persists incrementally so a Colab
disconnect never loses completed work. Rerunning skips any scenario_id that
already has a completed row in the CSV (resumable); failed scenarios get no
CSV row, only a results/experiment_errors.log entry, so they are retried on
the next run.
"""

import csv
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

try:
    from src.hooks import DEFAULT_MODEL, load_model
    from src.soo_distance import compute_soo_distance
    from src.behavior import GENERATION_CONFIG, classify_scenario
    from data.generate_scenarios import Scenario, validate_scenarios
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent / "data"))
    from hooks import DEFAULT_MODEL, load_model
    from soo_distance import compute_soo_distance
    from behavior import GENERATION_CONFIG, classify_scenario
    from generate_scenarios import Scenario, validate_scenarios

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = REPO_ROOT / "data" / "scenarios.json"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "experiment_results.csv"
ERRORS_LOG = RESULTS_DIR / "experiment_errors.log"
METADATA_JSON = RESULTS_DIR / "experiment_metadata.json"

EXPECTED_SCENARIO_COUNT = 54
PRIMARY_LAYER = 19
EXPLORATORY_LAYERS = [10, 15, 20, 25]
N_MODEL_LAYERS = 32  # Mistral-7B-Instruct-v0.2 decoder layer count (see src/hooks.py)

FIELDNAMES = [
    "scenario_id",
    "primary_layer_distance",
    "layer_10_distance",
    "layer_15_distance",
    "layer_20_distance",
    "layer_25_distance",
    "label",
    "detected_room",
    "reason",
    "raw_response_text",
    "temptation_level",
    "object_value",
    "honesty_consequence",
    "relationship",
    "social_context",
    "framing",
]


def preflight_check() -> list[Scenario]:
    """Validate everything possible before the model is loaded. Raises
    SystemExit with a clear message on the first violation."""
    if not SCENARIOS_PATH.exists():
        raise SystemExit(f"Preflight failed: {SCENARIOS_PATH} does not exist.")

    scenarios = [Scenario(**s) for s in json.loads(SCENARIOS_PATH.read_text())]

    if len(scenarios) != EXPECTED_SCENARIO_COUNT:
        raise SystemExit(
            f"Preflight failed: expected {EXPECTED_SCENARIO_COUNT} scenarios, "
            f"found {len(scenarios)} in {SCENARIOS_PATH}."
        )

    try:
        validate_scenarios(scenarios)  # unique IDs + required fields present
    except ValueError as e:
        raise SystemExit(f"Preflight failed: {e}")

    for layer_idx in [PRIMARY_LAYER] + EXPLORATORY_LAYERS:
        if not (0 <= layer_idx < N_MODEL_LAYERS):
            raise SystemExit(
                f"Preflight failed: layer index {layer_idx} invalid for "
                f"{DEFAULT_MODEL} ({N_MODEL_LAYERS} layers, 0..{N_MODEL_LAYERS - 1})."
            )

    return scenarios


def load_completed_ids() -> set[str]:
    if not RESULTS_CSV.exists():
        return set()
    with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
        return {row["scenario_id"] for row in csv.DictReader(f)}


def write_metadata() -> None:
    metadata = {
        "model_name": DEFAULT_MODEL,
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
        },
        "primary_layer": PRIMARY_LAYER,
        "exploratory_layers": EXPLORATORY_LAYERS,
        "activation_aggregation": "mean pooling",
        "behavioral_decoding": dict(GENERATION_CONFIG),
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2))


def build_result_row(scenario: Scenario, soo_result, behavior_result) -> dict:
    exploratory = soo_result.exploratory_distances
    return {
        "scenario_id": scenario.scenario_id,
        "primary_layer_distance": soo_result.primary_distance,
        "layer_10_distance": exploratory[10],
        "layer_15_distance": exploratory[15],
        "layer_20_distance": exploratory[20],
        "layer_25_distance": exploratory[25],
        "label": behavior_result.label,
        "detected_room": behavior_result.detected_room,
        "reason": behavior_result.reason,
        "raw_response_text": behavior_result.raw_response_text,
        "temptation_level": scenario.temptation_level,
        "object_value": scenario.object_value,
        "honesty_consequence": scenario.honesty_consequence,
        "relationship": scenario.relationship,
        "social_context": scenario.social_context,
        "framing": scenario.framing,
    }


def run() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    scenarios = preflight_check()
    write_metadata()

    completed_ids = load_completed_ids()
    remaining = [s for s in scenarios if s.scenario_id not in completed_ids]
    total = len(scenarios)
    print(
        f"Loaded {total} scenarios "
        f"({len(completed_ids)} already completed, {len(remaining)} remaining)."
    )

    errors = 0
    failed_ids: list[str] = []

    if remaining:
        model, tokenizer = load_model()

        is_new_file = not RESULTS_CSV.exists()
        with RESULTS_CSV.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            if is_new_file:
                writer.writeheader()
                csv_file.flush()

            start_index = total - len(remaining)
            for offset, scenario in enumerate(remaining):
                i = start_index + offset + 1
                try:
                    soo_result = compute_soo_distance(
                        model,
                        tokenizer,
                        scenario.self_prompt,
                        scenario.other_prompt,
                        primary_layer=PRIMARY_LAYER,
                        exploratory_layers=EXPLORATORY_LAYERS,
                    )
                    behavior_result = classify_scenario(model, tokenizer, scenario)

                    writer.writerow(build_result_row(scenario, soo_result, behavior_result))
                    csv_file.flush()

                    print(
                        f"[{i}/{total}] {scenario.scenario_id} "
                        f"label={behavior_result.label} "
                        f"distance={soo_result.primary_distance:.6f}"
                    )
                except Exception as e:
                    errors += 1
                    failed_ids.append(scenario.scenario_id)
                    print(f"[{i}/{total}] {scenario.scenario_id} ERROR: {e}")
                    with ERRORS_LOG.open("a", encoding="utf-8") as log:
                        log.write(f"{scenario.scenario_id}: {e}\n")
                        log.write(traceback.format_exc())
                        log.write("\n")

    label_counts = Counter()
    n_completed = 0
    if RESULTS_CSV.exists():
        with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        n_completed = len(rows)
        label_counts = Counter(row["label"] for row in rows)

    print("\n=== Experiment complete ===")
    print(f"Total scenarios completed: {n_completed}/{total}")
    print(f"  honest: {label_counts['honest']}")
    print(f"  deceptive: {label_counts['deceptive']}")
    print(f"  excluded: {label_counts['excluded']}")
    print(f"Errors this run: {errors}")
    if failed_ids:
        print(f"Failed scenario IDs: {failed_ids}")


if __name__ == "__main__":
    run()
