"""Forward hooks on Mistral self-attention o_proj for activation capture.

Run this file directly on a CUDA machine (e.g. Colab T4) to load the model
and print its structure:

    python src/hooks.py

This confirms the o_proj module path, decoder layer count/numbering, and the
shape of the captured activation tensor BEFORE soo_distance.py is written
against them, per the project's methodology-first constraint.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def load_model(model_name: str = DEFAULT_MODEL, device_map: str = "auto"):
    """Load Mistral in 4-bit. Assumes CUDA is available (Colab T4)."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map=device_map
    )
    model.eval()
    return model, tokenizer


def get_o_proj(model, layer_idx: int):
    """Return the self_attn.o_proj module for the given decoder layer index."""
    return model.model.layers[layer_idx].self_attn.o_proj


class ActivationCapture:
    """Context manager that hooks self_attn.o_proj on the given layers and
    records their forward-pass output activations.

    Usage:
        with ActivationCapture(model, layer_indices=[19]) as cap:
            model(**inputs)
        act = cap.activations[19]  # shape (batch, seq_len, hidden_size)

    The captured tensor is the OUTPUT of o_proj: the self-attention block's
    contribution to the residual stream at that layer, before the residual
    add. Hooks are removed on __exit__.
    """

    def __init__(self, model, layer_indices: list[int]):
        self.model = model
        self.layer_indices = list(layer_indices)
        self.activations: dict[int, torch.Tensor] = {}
        self._handles = []

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            self.activations[layer_idx] = output.detach()

        return hook

    def __enter__(self):
        for idx in self.layer_indices:
            handle = get_o_proj(self.model, idx).register_forward_hook(
                self._make_hook(idx)
            )
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles:
            h.remove()
        self._handles = []


def print_model_structure(model):
    layers = model.model.layers
    print(f"Number of decoder layers: {len(layers)} (indices 0..{len(layers) - 1})")
    o_proj0 = layers[0].self_attn.o_proj
    print("o_proj module path: model.model.layers[i].self_attn.o_proj")
    print(
        f"o_proj[0]: {type(o_proj0).__name__}, "
        f"in_features={o_proj0.in_features}, out_features={o_proj0.out_features}, "
        f"bias={o_proj0.bias is not None}"
    )


if __name__ == "__main__":
    model, tokenizer = load_model()
    print_model_structure(model)

    test_prompt = "Which room would you recommend, Room A or Room B?"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with ActivationCapture(model, layer_indices=[19]) as cap:
        with torch.no_grad():
            model(**inputs)

    act = cap.activations[19]
    print(f"\nTest prompt: {test_prompt!r}")
    print(f"Tokenized seq_len: {inputs['input_ids'].shape[1]}")
    print(f"Layer 19 o_proj output activation shape: {tuple(act.shape)}")
    print("  interpreted as (batch_size, seq_len, hidden_size)")
