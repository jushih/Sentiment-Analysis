# Boardgame Sentiment Analysis

This project explores board game reviews on boardgamegeek.com with the goal of predicting ratings based on the comments users leave in their reviews. Sentiment in this case is represented by a game rating on a scale of 1-10 assigned by the user. Predictive analysis was performed using the NLTK and sklearn packages. I summarize the main results of the analysis here.

The dataset used was compiled by Matt Borthwick of the Portland Data Science Group. It contains the fields:

* **gameID** - Unique identifier for each board game 
* **rating** - Game rating assigned by user
* **comment** - User review about the game 

### Data Exploration

<img src="https://github.com/jushih/Sentiment-Analysis/blob/master/images/histogram.png" />

Data exploration was conducted on a sample of 847 reviews while the full set contains around 800,000 reviews. User ratings are concentrated in the 6-9 range, with fewer negative and perfect ratings. Most users assign ratings of integer values, but some use decimal values such as 7.5 or 7.22. For the purpose of predictive analysis, ratings will be rounded to the nearest integer.

Next we examine user comments. Data cleaning is performed to convert the text to lowercase and remove punctuation. The nltk stopwords list is used to filter out generic words that don't contain sentiment. The cleaned text is saved into a separate column.

![cleaned](/images/cleaned_text.png)

With the cleaned comments, I examine the top 100 most frequently used words in the reviews. Some common words such as 'game' and 'player' don't contain sentiment, and these are added to the stopwords list. Below are the top 10 words after stopwords have been removed. 

| Top 10 Words |
| ------------ |
| like         |
| fun          |
| good         |
| much         |
| time         |
| get          |
| great        |
| better       |
| dont         |
| well         |
  
I write a function that generates a histogram of how often a word appears with a rating and take a look at several words. Some words like "great" are strongly associated with one sentiment, while other words such as "hate" are found in both positive and negative reviews though it is a negative term. In reviews, the user may compare games to one another and express a variety of sentiments, or use modifiers ("didn't like") to express the opposite sentiment, adding to the complexity of the analysis.

![appearance](/images/great_hate.png)

A word cloud of positive (rating > 8) and negative (rating < 3) reviews is generated. While the positive word cloud contains mostly positive words, the negative word cloud contains a mix of words that are not necessarily negative.

![wordcloud](/images/wordcloud.png)

### Predictive Modeling

To prepare the data for modeling, I split the cleaned corpus of words into a training and test set.

```python
X_train, X_test, y_train, y_test = train_test_split(corpus.cleaned, corpus.rating, test_size=0.20)
```

I then set up a pipeline using sklearn where I define and tune the model.

```python
X_train, X_test, y_train, y_test = train_test_split(corpus.cleaned, corpus.rating, test_size=0.20)

model_nb3 = Pipeline([
    ('count_vectorizer', CountVectorizer( ngram_range=(1, 2), min_df=10, lowercase = True, 
    stop_words = stopwords.words('english'))), 
    ('tfidf_transformer',  TfidfTransformer()), 
    ('classifier', MultinomialNB()) ])

model_nb3.fit(X_train,y_train.astype('int'))
```

The pipeline does the following:

* `count_vectorizer` - Breaks up the text into a matrix with each word (called "token" in NLP) being the column of the matrix and the value being the count of occurences. 
* `ngram_range` - Optional parameter to extract the text in groups of 2 or more words together. This is useful because the modifiers such as 'not' can be used to change the following word's meaning.
* `stopwords` - Removes any words from the stopwords list created in the data exploration step.
* `lowercase` - Converts all text into lowercase.
* `tfidf_transformer` - Weighs terms by importance to help with feature selection.
* `classifier` - I try two types of models suited to multi-class classification, Multinomial NB and LinearSVC.

Model performance is determined by calculating Root Mean Square Error, **RMSE**.

I start with a Baseline model that predicts every review to be the mean rating. Since ratings are heavily clustered around the mean, the goal is to make the model perform better than the baseline prediction in being able to identify positive and negative sentiment and assign ratings accordingly. The baseline model has an RMSE of 1.68.

The **Multinomial Naive Bayes** model calculates the probability that a comment will belong in a class based on word counts. In our case, the classes are the ratings of 1 through 10. The **LinearSVC** model uses a support vector machine algorithm to determine the hyperplane that maximizes the distance between the different classes. 
MultinomialNB showed better performance than LinearSVC.

In model tuning, I found `count_vectorizer` to predict better than using `tfidf_vectorizer` and an `ngram_range=(1,2)` to be ideal. Using no bigrams or using n-grams of greater than 2 did not improve model performance. 

##### Summary of Model Performance

| Model Name                        | RMSE    |  
| ----------------------------------| ------- |
| Baseline                          | 1.68    |
| Multinomial NB                    | 1.66    |
| **Multinomial NB (n-grams tuned)**  | **1.57**  |
| Linear SVC                        | 1.60    |

The **Multinomial NB model using n-grams** is the best performing model based on RMSE. However, when looking at the predictions it generates:

##### Multinomial NB (n-grams) Confusion Matrix
![matrix](/images/multinomialNB_ngrams.png)

The model is simply predicting reviews around the average rating! It does not predict any low reviews. This is because the training data is so unbalanced that it can't detect a negative review.

### Predictive Modeling Part II - Dealing with Unbalanced Data

For the model to predict a range of sentiment, we need to give it more negative reviews to train on. I subsample the data and create a set of reviews with 3000 reviews of each rating. The best performing Multinomial NB model is trained on the subsample. The resulting model is then used to predict on the unbalanced dataset.

To address the issue of negative reviews being classified positively, another method to assess the model is to use **weighted RMSE**. If our goal is to be able to distinguish negative reviews from positive reviews, the model can be penalized more for incorrectly classifying a review that deviates from the mean. In our case, it will give more weight to errors when predicting extremely negative ratings.

The resulting confusion matrix shows a far better distribution of predictions. There are both negative predictions and a cluster of predictions around the mean, resembling the distribution of the review data.

##### Multinomial NB (re-trained on balanced data) Confusion Matrix
![matrix2](/images/multinomial_retrained.png)

The weighted RMSE of the new model is a large improvement over the previous models built. While the new Multinomial model may have higher regular RMSE at 2.3, it is doing a better job predicting sentiment instead of predicting only around the mean.  

##### Model Performance based on weighted RMSE

| Model Name                     | Weighted RMSE  |  
| -------------------------------| -------------- |
| Baseline                       | 3.95           |
| Multinomial NB                 | 4.05           |
| Multinomial NB (n-grams tuned) | 3.68           |
| Linear SVC                     | 3.56           |
| **Multinomial NB (re-trained)**  | **2.23**         |





