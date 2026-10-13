"""Load configs/experiment.yaml, resolve project paths and read secrets (OpenAI key, Hugging Face token) from the environment or a local .env file.

What goes here:
- PROJECT_ROOT: the repo folder, so paths work no matter which folder a script is started from.
- load_config(path=None) -> dict: read configs/experiment.yaml into a dictionary.
- get_path(config, name) -> Path: turn a name under `paths:` in the config (e.g. "data_processed")
  into a full path, and create the folder if it does not exist.
- get_secret(name) -> str or None: read a key such as OPENAI_API_KEY or HF_TOKEN from .env or the
  environment. Adapted from the portable-setup cell in the course notebooks; say so in a comment.

Used by: every script and notebook, usually as the first thing they do.
"""
