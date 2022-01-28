# Fake News Prediction using NLP and Classification


### Background


Reddit is a social news website and forum where content is socially curated and promoted by site members through voting. Posts are organized by subject into user-created boards called "communities" or "subreddits", which cover a wide variety of topics. Ranked among the most popular mobile social apps in the United States, Reddit has more than 48 million monthly active users (as of June 2021). Its prominence and popularity, data source for researchers

### Problem Statement

Four in five Singaporeans say that they can confidently spot fake news, but when put to the test, about 90 per cent mistakenly identified at least one out of five fake headlines as being real. Thus, while one could be confident about his ability to detect fake news, one may not be able to do in reality.
<font size="1">[Source](https://www.straitstimes.com/singapore/4-in-5-singaporeans-confident-in-spotting-fake-news-but-90-per-cent-wrong-when-put-to-the)</font>  

People fall into the trap of reading fake news because they were written like real news to deliberately  misinform or deceive readers. Fake news is not just making people believe false things, it’s also making people less likely to consume or accept information.
<font size="1">[Source](https://www.theatlantic.com/ideas/archive/2019/06/fake-news-republicans-democrats/591211/)</font>  

Fake news is a prevalent and harmful problem in our modern society, often misleading the general public on important topics such as healthcare and defense. This can lead to long standing societal issues which are a detriment to nations worldwide. More than making people believe false things, the rise of fake news is making it harder for people to see the truth.


In view of this menace, we aim to develop a model that is discerning enough to separate real news from fake news, so that government bodies can weed out the fake news, thus creating a secure, and more misinformation-resilient society in the long run. We want to create a model that will help identify real and fake news just based on the titles.
Titles are a strong differentiating factor between fake and real news. In general, fake news has very little information or substance in the article content but packs a ton of information into the titles. Titles are often the determining factor on whether someone will click on or read the article.
<font size="1">[Source](https://medium.com/the-nela-research-blog/fake-news-starts-with-the-title-ad7b63bf79c0)</font>  

To tackle the problem, we will foucus on text-based news from subreddit `r/TheOnion` and `r/nottheonion`, using natural language processing and machine learning models to predict whether an article is from `r/TheOnion` (fake news) or from `r/nottheonion` (real news). 

- **Primary** Stakeholders: Reddit Users, General Public
- **Secondary** Stakeholders: Reddit Moderators, Government

### Dataset

<img src="images/onion or not.jpg" width="400"/>

For this project, we have selected the following subreddits that are seemingly similar; In actual fact, only one thread lists real news while the other thread lists fake news.
- [r/TheOnion](https://www.reddit.com/r/TheOnion/): a subreddit that lists **satire** news or **fake** stories that are ridiculous, written in such a way that makes it seem plausible.
- [r/NotTheOnion](https://www.reddit.com/r/nottheonion/): a subreddit that lists **real** news stories so absurd that one could honestly believe they were from `TheOnion`

### Goal 

Our team aims to develop a model using natural language processing and machine learning models to predict whether an article is from `r/TheOnion` (fake news) or `r/NotTheOnion` (real news). 
Helping government bodies/regular citizens to identify the fake news, thus creating a secure, and more misinformation-resilient society.

After scraping posts from the two subreddits: `TheOnion` and `NotTheOnion`, we will use Natural Language Processing (NLP) to train a classification model to identify which subreddit a given post came from, based on the title of a post.

### Evaluation of Success
- Model Score
- High F1 Score (low occurrence of False Positives (FP) and False Negatives (FN), we want to minimize both since the impact of both FPs and FNs are equally severe)

### Methodology

- **[0. Data Scraping from Reddit](#0)**

   To get the necessary posts for this project, I scrapped at least 5000 posts before 1 Jan 2022 from each subreddit.
  
- **[1. Import Scraped Data & Libraries](#1)**
- **[2. Data Cleaning & EDA](#2)**
    
    The following was done during data cleaning,
      - Removed Links
      - Removed non-Alphabetical/Numberical and Single Character Words
      - Lemmatization and Tokenization
      - Dropped Duplicates

    Word Cloud after data cleaning:

  <img src="images/wordcloud.jpeg" width="400"/>
    
- **[3. Tokenization & Top Words](#3)**
- **[4. Train Test Split](#4)**
- **[5. Building & Fitting Models](#5)**


  | # | Model | Score |
  | --- | --- | --- |
  | 1 | Count Vectorizer + Multinomial Naive Bayes	| 0.811 |
  | 2 | TFIDF + Multinomial Naive Bayes | 0.793 |
  | 3 | Count Vectorizer + Logistic Regression | 0.795 |
  | 4 | TFIDF + Logistic Regression | 0.797 |
  | 5 | Count Vectorizer + Random Forest	| 0.754 |
  | 6 | TFIDF + Random Forest	| 0.754 |


- **[6. Best Model](#6)**

    ##### Based on Accuracy

    The model that gives the best score is `Count Vectorizer` and `Multinomial Naive Bayes` model, with the parameters:
    - `ngram_range`: (1,2)
    - `alpha`: 1.95

    The best score is 0.812 (81.2% accuracy), which is a 62.4% improvement in comparison to our baseline model, which has an accuracy of 0.5 (50% accuracy).

- **[7. Interpreting Model Coefficient](#7)**

  ##### Based on Interpretability
  An interpretable model helps us fundamentally understand the value and accuracy of our findings.

  To interpret model coefficients, we have chosen Count Vectorizer and Logistic Regression Model. We were able to interpret model coefficients and identify the top words that contribute the most positively to the following subreddits. Using the model coefficients that we have found, we are able to find the effect (no. of times) of a word occurence on the model classification.

    <img src="images/model_coeff.jpeg" width="400"/>

  - `r/NotTheOnion`: 
      1. nation: 10.91 times 
      2. onion: 7.92 times 
      3. breaking: 5.37 times 


  - `r/TheOnion`: 
      1. content: 11.36 times 
      2. colorado: 6.49 times 
      3. bitcoin: 5.64 times 

  However, our goal here is to create an accurate model with a low f1 score, given the impact of both FPs and FNs are equally detrimental. Hence, we want to find a model that gives the best score. <br>

- **[8. Conclusion](#7)**

  ##### Recommendations
  
    <img src="images/models.jpg" width="400"/>
  
    `Count Vectorizer` and `Multinomial Naive Bayes` model 
    - gives the best F1-Score (minimised False Positives and False Negatives)
    - simple, easy to implement
    - good and accurate text classification prediction 

  #### Problems
  - The model is overfitted to the training set - the train score (0.984) is significantly higher than the Test Score (0.824), which amounts to a ~16% difference in accuracy. However, since we are optimizing for accuracy, more importantly we want choose a model with the highest percentage of correct predictions.
  - Our model is limited to English words only, and thus, non-English posts were dropped completely.
  - Recentness of the terms used - New words and acronyms are created every day. Model might not be suitable for future classifications.
  - Shift in topic of concern/discussion - Model might not be a good representative of the content for future subreddit posts. Topic of interest that are being discussed changes with current events and time periods (i.e `NotTheOnion` tends to reflect current global issues). 

  A well-rounded model should take into account the other features or recognize the type of content discussed in a subreddit thread, not just words.


  #### Possible Enhancements
  - Consider non-text posts (i.e. images, video)
  - Explore other features of a post (i.e. subtext, comments, upvotes)
  - Analysing post authors, as they may play a critical role in the authenticity of a post's content. We can look at authors posting history, number of posts made, posting patterns or the type content posted, – because an author that have posted fake news before will more likely be posting fake news again.
  - For this project, we assumed that the classification is binary, but in reality, it might be more than just fake or real news, there can be a differentiation between Fake News and Sattire News.
  - Content-based analysis - further exploration is needed to better understand the content of the posts, not just individual words), with that in mind, we may explore other NLP methods (e.g. BERT, a transformer-based machine learning technique for NLP pre-training developed by Google, it is pre-trained on a large corpus of unlabelled text including the entire Wikipedia (\~2.5 billion words) and BookCorpus (\~800 million words). BERT is a “deeply bidirectional” model, it considers both the left and the right side of a token’s context before making a prediction.
  <font size="1">[Source](https://www.analyticsvidhya.com/blog/2019/09/demystifying-bert-groundbreaking-nlp-framework/)</font>  


