"""Load the Gemma 3 and Borealis instruction-tuned models and tokenizers locally with Hugging Face transformers, pinned to a fixed revision.

What goes here:
- pick_device() -> str: "mps" on an Apple Silicon Mac, otherwise "cuda" or "cpu".
- load_model(model_id, revision=None, dtype="bfloat16") -> (model, tokenizer): download the model
  (first time only; it is cached in ~/.cache/huggingface) and load it ready for scoring (model.eval()).
- model_revision(model) -> str: the exact commit hash that was loaded, saved in the run metadata.

Notes: the 12B checkpoints are image+text models (their model cards load them with
Gemma3ForConditionalGeneration); we only give them text. The 270M and 1B models are text-only.
Downloading Gemma needs the Hugging Face token in .env.
Used by: scripts/05_run_qa.py, demo/demo.ipynb.
"""
