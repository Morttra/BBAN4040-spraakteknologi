"""Load configs/experiment.yaml, resolve project paths and read secrets (OpenAI key, Hugging Face token) from the environment or a local .env file."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

#Defining the repo folder as 3 folders up from this file (src/norrag/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2] 

def load_config(path=None):
    """Read the experiment settings into a dictionary (configs/experiment.yaml unless another path is given)."""
    if path is None:
        path = PROJECT_ROOT / "configs" / "experiment.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_path(config, name):
    """Creates complete paths from "paths" in the config file loaded in load_config()"""
    path = PROJECT_ROOT / config["paths"][name] # e.g. .../BBAN4040-spraakteknologi/data/processed
    path.mkdir(parents=True, exist_ok=True) # create the folder (and any parent folders) if missing
    return path

def get_secret(name, default=None): 
    # Adapted from the "portable-setup" cell in the course's martra-notebooks (github.com/arildbn/bban4040)
    """Read a key such as OPENAI_API_KEY or HF_TOKEN from .env or the environment."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get(name, default)
    return key.strip() if isinstance(key, str) else key #remove space before or after key, if there