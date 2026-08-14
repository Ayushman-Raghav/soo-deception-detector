"""SOO (self-other overlap) distance between a matched self/other prompt pair.

Pipeline per prompt: one forward pass -> hook self_attn.o_proj at the
requested layers -> mean-pool each (1, seq_len, 4096) activation over the
sequence dimension into a (4096,) vector. Self and other prompts are run in
SEPARATE forward passes (never batched) so there is no padding to contaminate
the pooled vector.

AGGREGATION CHOICE (see README for full discussion):
    The self and other prompts differ in length ("yourself" vs "Bob"), so
    token positions do not align one-to-one. Mean pooling over the sequence
    dimension is used as the primary aggregation to handle this. This is a
    deliberate methodological choice, NOT a claimed reproduction of the
    paper's exact aggregation procedure (the paper is underspecified here).
    Mean pooling also averages over the largely-shared prompt tokens (self
    and other prompts are identical except for the self/other manipulation),
    which may dilute the self/other signal; last-token pooling is a natural
    alternative to check later, but is intentionally not implemented here.

DISTANCE:
    SOO distance = mean squared error (MSE) between the two pooled 4096-dim
    vectors, matching the paper's stated MSE distance definition.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from src.hooks import ActivationCapture
except ImportError:
    from hooks import ActivationCapture

HIDDEN_SIZE = 4096


@dataclass
class SOODistanceResult:
    primary_layer: int
    primary_distance: float
    exploratory_distances: dict[int, float]  # never includes primary_layer


def _validate_layer(model, layer_idx: int):
    n_layers = len(model.model.layers)
    if not (0 <= layer_idx < n_layers):
        raise ValueError(
            f"Invalid layer index {layer_idx}: model has {n_layers} layers "
            f"(valid range 0..{n_layers - 1})"
        )


def _pool_activation(act: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Mean-pool a (1, seq_len, HIDDEN_SIZE) activation into (HIDDEN_SIZE,)."""
    if act.dim() != 3 or act.shape[0] != 1 or act.shape[2] != HIDDEN_SIZE:
        raise ValueError(
            f"Layer {layer_idx}: expected captured activation shape "
            f"(1, seq_len, {HIDDEN_SIZE}), got {tuple(act.shape)}"
        )
    pooled = act.mean(dim=1).squeeze(0)
    if pooled.shape != (HIDDEN_SIZE,):
        raise ValueError(
            f"Layer {layer_idx}: pooled vector has unexpected shape {tuple(pooled.shape)}, "
            f"expected ({HIDDEN_SIZE},)"
        )
    return pooled


def _get_pooled_activations(
    model, tokenizer, prompt: str, layers: list[int]
) -> dict[int, torch.Tensor]:
    """One forward pass for `prompt`, hooking all `layers` at once.

    Returns {layer_idx: pooled (HIDDEN_SIZE,) vector}.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with ActivationCapture(model, layer_indices=layers) as cap:
        with torch.no_grad():
            model(**inputs)

    return {
        layer_idx: _pool_activation(cap.activations[layer_idx], layer_idx)
        for layer_idx in layers
    }


def compute_soo_distance(
    model,
    tokenizer,
    self_prompt: str,
    other_prompt: str,
    primary_layer: int = 19,
    exploratory_layers: list[int] | None = None,
) -> SOODistanceResult:
    """Compute SOO distance for a matched self/other prompt pair.

    Registers hooks on the primary layer plus all exploratory layers
    simultaneously (deduplicated), so only one forward pass per prompt is
    needed regardless of how many layers are requested.
    """
    exploratory_layers = exploratory_layers or []

    # dedupe while preserving order; primary layer appears exactly once
    all_layers = [primary_layer] + [
        l for l in dict.fromkeys(exploratory_layers) if l != primary_layer
    ]

    for layer_idx in all_layers:
        _validate_layer(model, layer_idx)

    self_vecs = _get_pooled_activations(model, tokenizer, self_prompt, all_layers)
    other_vecs = _get_pooled_activations(model, tokenizer, other_prompt, all_layers)

    distances: dict[int, float] = {}
    for layer_idx in all_layers:
        self_vec = self_vecs[layer_idx]
        other_vec = other_vecs[layer_idx]
        if self_vec.shape != other_vec.shape:
            raise ValueError(
                f"Layer {layer_idx}: self/other pooled vector shape mismatch "
                f"{tuple(self_vec.shape)} vs {tuple(other_vec.shape)}"
            )
        distances[layer_idx] = F.mse_loss(self_vec, other_vec).item()

    return SOODistanceResult(
        primary_layer=primary_layer,
        primary_distance=distances[primary_layer],
        exploratory_distances={
            l: d for l, d in distances.items() if l != primary_layer
        },
    )


if __name__ == "__main__":
    # Smoke test: one matched pair, primary layer 19 + a few exploratory layers.
    try:
        from src.hooks import load_model
        from data.generate_scenarios import get_pilot_scenarios
    except ImportError:
        from hooks import load_model
        import sys
        from pathlib import Path

        sys.path.append(str(Path(__file__).resolve().parent.parent / "data"))
        from generate_scenarios import get_pilot_scenarios

    model, tokenizer = load_model()
    scenario = get_pilot_scenarios()[0]

    result = compute_soo_distance(
        model=model,
        tokenizer=tokenizer,
        self_prompt=scenario.self_prompt,
        other_prompt=scenario.other_prompt,
        primary_layer=19,
        exploratory_layers=[10, 15, 20, 25],
    )

    print(f"Scenario: {scenario.scenario_id}")
    print(f"Primary layer ({result.primary_layer}) SOO distance: {result.primary_distance:.6f}")
    print("Exploratory layer distances:")
    for layer_idx in sorted(result.exploratory_distances):
        print(f"  layer {layer_idx}: {result.exploratory_distances[layer_idx]:.6f}")
