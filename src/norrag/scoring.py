"""Log-likelihood multiple-choice scoring: compute the model's log-probability of each answer option, as in NorEval.

The idea: the model does not write an answer. For each option we ask how likely it is that this text
comes next after the prompt, and pick the most likely option. NorEval (through lm-evaluation-harness)
scores NRK-Quiz-QA this way, so our numbers can be compared with theirs.

What goes here:
- option_logprobs(model, tokenizer, prompt, continuations) -> list of float: for each continuation,
  the sum of the log-probabilities of its tokens given the prompt.
- predict(logprobs, lengths=None) -> int: the index of the best option. With lengths, each score is
  divided by the option's length in characters first (NorEval's acc_norm), so long options are not
  punished for being long.

Used by: pipeline.py.
Decisions (`qa`): score the option text (NorEval prompts p0/p1) or the option letter (p2/p3), and
whether to wrap prompts in the models' chat template.
"""
