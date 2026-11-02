"""Turn an NRK quiz question into a search query (for example the full text, the last sentence only, or the question plus its options).

Many NRK questions start with a little story ("Du blir vekket av forsvarsministeren ...") before the
actual question, and those extra words can pull the search in the wrong direction.

What goes here:
- build_query(question, options, mode) -> str, where mode is one of
      "full"            the whole question text
      "last_sentence"   only the final sentence, usually the actual question
      "with_options"    the question followed by the answer options
  The same query goes to both retrievers.

Used by: scripts/04_retrieve.py, the demo.
Setting (`retrieval.query_mode`): "full" (decided); the final sentence alone is often too short to
search with ("Hvilken?") and the sentence split breaks on "ca." and "2.".
"""

import re

MODES = ["full", "last_sentence", "with_options"]


def build_query(question, options, mode):
    """Build the search query for one question in the chosen mode."""
    if mode == "full":
        return question
    if mode == "last_sentence":
        sentences = re.split(r"(?<=[.!?])\s+", question.strip())  # split after . ! or ? followed by a space
        return sentences[-1]
    if mode == "with_options":
        return question + " " + " ".join(options)
    raise ValueError(f"Unknown query mode {mode!r}; use one of {MODES}.")
