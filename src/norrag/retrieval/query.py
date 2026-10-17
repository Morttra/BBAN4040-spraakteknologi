"""Turn an NRK quiz question into a search query (for example the full text, the last sentence only, or the question plus its options).

Many NRK questions start with a little story ("Du blir vekket av forsvarsministeren ...") before the
actual question, and those extra words can pull the search in the wrong direction.

What goes here:
- build_query(question_row, mode) -> str, where mode is one of
      "full"            the whole question text
      "last_sentence"   only the final sentence, usually the actual question
      "with_options"    the question followed by the answer options
  The same query goes to both retrievers.

Used by: scripts/04_retrieve.py, the demo.
Decision (`retrieval.query_mode`): which mode to use.
"""
