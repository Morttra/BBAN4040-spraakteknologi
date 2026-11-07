"""Log-likelihood multiple-choice scoring: compute the model's log-probability of each answer option, as in NorEval.

The idea: the model does not write an answer. We ask how likely each option letter is as the next token
after the prompt, and pick the most likely one. NorEval (through lm-evaluation-harness) scores
NRK-Quiz-QA this way, so our numbers can be compared with theirs.

What goes here:
- option_logprobs(model, tokenizer, prompt, labels, chat_template=True) -> list of float: for each
  option letter, its log-probability as the next token. With chat_template, the prompt is wrapped as a
  user message, except its last paragraph ("Svar:"), which becomes the start of the model's reply.
  Otherwise the model tends to begin with a full sentence ("Svaret er ...") instead of a letter.
- predict(logprobs) -> int: the index of the best option.

Every option is a single letter, so one pass through the model scores all of them, and long options are
neither favoured nor punished (NorEval's acc and acc_norm give the same result).
Used by: pipeline.py.
Settings (`qa`): scoring (option_letter, NorEval p3) and chat_template.
"""

import torch


def option_logprobs(model, tokenizer, prompt, labels, chat_template=True):
    """Log-probability of each option letter as the next token after the prompt (one pass through the model)."""
    if chat_template:
        # Everything but the last paragraph is the user's message; "Svar:" starts the model's reply
        message, answer_start = prompt.rsplit("\n\n", 1)
        text = tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=False,
                                             add_generation_prompt=True) + answer_start
        inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")  # text already starts with <bos>
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    letters = [" " + label for label in labels]  # a space after "Svar:": " B", as in NorEval
    letter_ids = [tokenizer(letter, add_special_tokens=False)["input_ids"] for letter in letters]
    if any(len(ids) != 1 for ids in letter_ids):
        raise ValueError(f"Every option letter must be a single token: {letters}")
    with torch.no_grad():  # scoring only: skip the bookkeeping that training needs
        # logits_to_keep=1: only the scores for the token after the prompt, not for every position
        logits = model(**inputs.to(model.device), logits_to_keep=1).logits[0, -1]
    log_probs = torch.log_softmax(logits.float(), dim=-1)  # one log-probability per token in the vocabulary
    return [log_probs[ids[0]].item() for ids in letter_ids]
