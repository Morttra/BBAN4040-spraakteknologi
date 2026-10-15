"""Read and write JSONL records (retrieval runs, QA outputs, judgements) together with run metadata such as model, revision, prompt template and settings.

JSONL means one JSON object per line. New lines can be appended as results come in, so a long run that
stops can continue where it left off, and you can open the file in VS Code and read it.

Used by: scripts 04, 05 and 07, and the notebooks that read results.
"""

import json
import platform
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from norrag.config import PROJECT_ROOT

# Packages whose versions can change results; their versions are saved in every .meta.json
KEY_PACKAGES = ["torch", "transformers", "sentence-transformers", "bm25s", "scikit-learn", "openai"]


def write_jsonl(path, rows, append=False):
    """Save a list of dictionaries, one per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # create the folder if it is missing
    mode = "a" if append else "w"  # "a" adds to the end of the file, "w" starts a new file
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_jsonl(path):
    """Read a JSONL file back into a list of dictionaries, one per line."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():  # skip empty lines
                rows.append(json.loads(line))
    return rows


def read_jsonl_df(path):
    """Read a JSONL file into a pandas DataFrame, one row per line, for analysis."""
    return pd.DataFrame(read_jsonl(path))

def done_ids(path, key="qid"):
    """Return the set of ids already in a file, so a restarted script can skip them.

    key is one field name ("qid"), or several (["qid", "chunk_id"]) when an id is a combination.
    """
    path = Path(path)
    if not path.exists():  # nothing done yet
        return set()
    if isinstance(key, str):
        return {row[key] for row in read_jsonl(path)}
    return {tuple(row[k] for k in key) for row in read_jsonl(path)}

def git_commit():
    """Return the current git commit hash, ending in "-dirty" if tracked files have unsaved changes (None outside git)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # git not installed, or not a git repo
        return None
    return commit + "-dirty" if changes else commit


def package_version(name):
    """Return the installed version of a package, or None if it is not installed."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def write_meta(path, config, **extra):
    """Save a .meta.json next to a results file: time, git commit, package versions, settings and any extra details.

    Example: write_meta("results/raw/qa/gemma-4b__rag_bm25.jsonl", config, model_id="google/gemma-3-4b-it")
    writes results/raw/qa/gemma-4b__rag_bm25.meta.json. This documents model, version, prompts and settings.
    """
    meta_path = Path(path).with_suffix(".meta.json")
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "packages": {name: package_version(name) for name in KEY_PACKAGES},
        **extra,  # e.g. model_id, revision, condition, prompt_template
        "config": config,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta_path
