# Annotations

Human relevance labels for evaluation layer 1, together with the annotation guidelines and file format.

For 50 questions, each of you labels whether the chunks the retrievers found are relevant. The labels
are used to compare the retrievers and to check the LLM judge (layer 2), which uses the same definition.

## Files

- `sheet_<name>.xlsx`: your sheet, one row per (question, chunk) pair, 469 rows. Each row shows
  the question, its options, the correct answer and the chunk. Fill in `relevant`, and `comment` if
  needed, and save it in the same place as an Excel file (.xlsx).
- `key.csv`: which question, chunk and retrievers each row belongs to. Don't open it before both of
  you are done: the sheets are blind, so you can't tell which retriever found a chunk.

Both are written by `scripts/05_make_annotation_sample.py`.

## What counts as relevant

- **1 (relevant):** the chunk states the answer to this question, or lets you work it out with a
  simple step and no outside knowledge.
- **0 (not relevant):** everything else.

Judge from the chunk alone, with the question and the correct answer from the sheet. What you know
about the topic yourself doesn't count. The article title at the start of the chunk is part of the chunk.

## Borderline cases

| Case | Label | Example |
|---|---|---|
| The answer appears, but in another context | 0 | "Hvilket år ble Norge selvstendig?" and a chunk about something else that happens to mention 1905 |
| The answer needs a simple step | 1 | "Hvor mange år fylte Julius i 2019?" and a chunk saying he was born in 1979 |
| The chunk only rules out some options | 0 | It shows that two of four options are wrong, but not which of the other two is right |
| The answer is written differently | 1 | The answer is "NVE" and the chunk says "Norges vassdrags- og energidirektorat" |
| The answer is only clear with the title | 1 | "I hvilken fjord kan du se både Preikestolen og Kjeragbolten?" and a chunk titled "Lysefjorden" whose text says "fjorden er kjent for Preikestolen og Kjeragbolten" |

## How to label

1. Label alone, and don't discuss any items until both sheets are done. Otherwise the agreement
   between you (Cohen's kappa) doesn't measure anything.
2. Put 1 or 0 in every row. If you are unsure, pick the label that fits the definition best and write
   a short note in `comment`.
3. It takes about 2.5–4 hours. Take breaks; labelling gets sloppy when you are tired.

## Disagreements

DECISION: how rows where the two of you disagree get their final label. Decide this before you
compare the sheets.
