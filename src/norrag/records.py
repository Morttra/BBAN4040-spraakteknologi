"""Read and write JSONL records (retrieval runs, QA outputs, judgements) together with run metadata such as model, revision, prompt template and settings.

JSONL means one JSON object per line. New lines can be appended as results come in, so a long run that
stops can continue where it left off, and you can open the file in VS Code and read it.

What goes here:
- write_jsonl(path, rows, append=False): save a list of dictionaries, one per line.
- read_jsonl(path) -> list of dicts, and read_jsonl_df(path) -> pandas DataFrame for analysis.
- done_ids(path, key="qid") -> set: the ids already in a file, so a restarted script can skip them.
- write_meta(path, config, **extra): save a small .meta.json next to a results file with the time,
  the git commit, the model id and revision and the settings used. This is how we document model,
  version, prompts and settings, as the course requires.

Used by: scripts 04, 05 and 07, and the notebooks that read results.
"""
