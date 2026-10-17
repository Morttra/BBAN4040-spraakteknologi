# Decision log

Dated log of methodological decisions and changes from the M1 sketch, used when writing M2 and the method section.

# Change in wikipedia dump:
In the beginning, we based our project on the Wikipedia dump from data.norge.no, which was dated around 20th March 2019. The 16th of October 2026 we discovered that a new version of the NRK-Quiz was released the same day. When checking which years the quiz reference, we found that quite a number of questions were more recent than 2019. The solution was to use a November 2023 dump of Wikipedia available from Hugging Face. This also gave us more articles and questions to work with. With the 2023 dump there are only 13 questions not covered by the dataset. 473 questions mentioned 2019 or later years, so changing the dataset was the right move. Version 2 of the quiz had also fixed som inaccuracies. 

It was discovered that the 2023 version was missing a lot of important numbers from the conversion process from HTML.

Solution was fetching the full 7.5 GB HTML dump from 20th March 2025 and converting the data by making a set of rules to strip HTML formatting from the text. Also added IDs to the articles and marked them nn or nb, like earlier. 
Result:
Saved 4,790 questions to data/processed/questions.parquet
Wikipedia (nb): 659026 pages [2:07:59, 85.82 pages/s] 
Wikipedia (nb): skipped 11 pages without HTML
Wikipedia (nn): 174589 pages [19:00, 153.02 pages/s]
Wikipedia (nn): skipped 2 pages without HTML
Saved 818,758 articles to data/processed/articles.parquet

# Embedding/Chunking:
100 word chunks, as 53% of articles is less than 100 words. 74% of articles are less than 200 words. These researchers used 100 word chunks on wikipedia (https://arxiv.org/pdf/2004.04906v2/1000) and it worked well. We keep the title in each chunk, and use a 20 word overlap window as a starting point. The sentence-BERT model used is nb-sbert-v2-base.