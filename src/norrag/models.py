"""Load the Gemma 3 and Borealis instruction-tuned models and tokenizers locally with Hugging Face transformers, pinned to a fixed revision.

What goes here:
- pick_device() -> str: "mps" on an Apple Silicon Mac, otherwise "cuda" or "cpu".
- load_model(model_id, revision=None, dtype="bfloat16") -> (model, tokenizer): download the model
  (first time only; it is cached in ~/.cache/huggingface) and load it ready for scoring (model.eval()).
- model_revision(model_id, revision=None) -> str: the exact commit hash of the downloaded model, saved in
  the run metadata and pinned in the config.

Notes: the 12B checkpoints are image+text models (their model cards load them with
Gemma3ForConditionalGeneration); we only give them text. The 270M and 1B models are text-only.
Downloading Gemma needs the Hugging Face token in .env.
Used by: scripts/07_run_qa.py, demo/demo.ipynb.
"""

from pathlib import Path

import torch
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

from norrag.config import get_secret


def pick_device():
    """Return "mps" on an Apple Silicon Mac, "cuda" on a computer with an NVIDIA GPU, otherwise "cpu"."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_id, revision=None, dtype="bfloat16"):
    """Load a model and its tokenizer, ready for scoring; the first call downloads them to ~/.cache/huggingface."""
    token = get_secret("HF_TOKEN")  # Gemma needs it: you must accept Google's licence on Hugging Face first
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
    # AutoModelForCausalLM picks the right class: Gemma3ForCausalLM for the text-only 270M models,
    # Gemma3ForConditionalGeneration for the image+text 12B models
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,  # None means the newest version
        dtype=getattr(torch, dtype),  # "bfloat16" -> torch.bfloat16, the precision Gemma 3 was trained in
        device_map=pick_device(),  # load the weights straight onto the GPU
        token=token,
    )
    model.eval()  # scoring mode (from_pretrained already does this; written out to be explicit)
    return model, tokenizer


def model_revision(model_id, revision=None):
    """Return the exact commit hash of a downloaded model (None if it is not downloaded)."""
    # The cache keeps each version in a folder named after its commit hash: .../snapshots/<hash>/config.json
    config_file = try_to_load_from_cache(model_id, "config.json", revision=revision)
    if not isinstance(config_file, str):  # not downloaded (yet)
        return None
    return Path(config_file).parent.name
