# Decision log

Dated log of methodological decisions and changes from the M1 sketch, used when writing M2 and the method section.

# Change in wikipedia dump:
In the beginning, we based our project on the Wikipedia dump from data.norge.no, which was dated around 20th March 2019. The 16th of October 2026 we discovered that a new version of the NRK-Quiz was released the same day. When checking which years the quiz reference, we found that quite a number of questions were more recent than 2019. The solution was to use a November 2023 dump of Wikipedia available from Hugging Face. This also gave us more articles and questions to work with. With the 2023 dump there are only 13 questions not covered by the dataset. 473 questions mentioned 2019 or later years, so changing the dataset was the right move. Version 2 of the quiz had also fixed som inaccuracies. 

It was discovered that the 2023 version was missing a lot of important numbers from the conversion process from HTML.

Solution was fetching the full 7.5 GB HTML dump from 20th March 2025 and converting the data by making a set of rules to strip HTML formatting from the text. Also added IDs to the articles and marked them nn or nb, like earlier. 

Later we decided to limit the project to bokmål only. This is what the stemmer is trained for and reduces data and compute a bit, as well as reducing noise from something not relevant for the purpose of the project. 

# Embedding/Chunking:
100 word chunks, as 53% of articles is less than 100 words. 74% of articles are less than 200 words. These researchers used 100 word chunks on wikipedia (https://arxiv.org/pdf/2004.04906v2/1000) and it worked well. We keep the title in each chunk, and use a 20 word overlap window as a starting point. The sentence-BERT model used is nb-sbert-v2-base.

# Prompt:
When choosing how to format the prompts going into the llms to answer questions we based the formatting on NorEvals format options for the NRK quiz. First we decided the p0 format was right as the questions are prompted without options. The reasoning was that this was the closest format to the use case we were exploring. After considering that part of the quiz is completely dependent on having options to choose from we rather decided to use the p3 format, which present one options line for line in the format "A: answer". This lets the llm answer a single letter in its reponse, which greatly reduces speed when three models are answering 3516 questions each. 

We should be aware that models can lean toward one letter. A test would be to take a subsample of the questions, do one run, then reorder the options and run again before comparison. 

When testing on a question with and without wrapping the prompt in accordance to how the models are instruction trained we say that with the wrapping the models were inclined to start the answer with "S", most likely as in "Svaret er...". After that we tried moving "Svar:" to the model response before testing both 12B models on a 50 question subset. This keeps the wording from NorEval. Changing the prompt more than this would mean deviating form NorEval's formatting.

After further testing it was revealed that frasing the sentence "Bakgrunn fra Wikipedia:" before the chunks was enough to make the models refuse answering the questions if the answer weren't in the chunks. 