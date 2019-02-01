#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:02:19 2019

@author: myidispg
"""

import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

a = 'This is a good movie'
sid.polarity_scores(a)

a = 'This was the best, most awesome movie EVER MADE!!!'
sid.polarity_scores(a)

a = 'This was the worst movie that has ever disgraced the scene.'
sid.polarity_scores(a)

import pandas as pd

df = pd.read_csv('../UPDATED_NLP_COURSE/TextFiles/amazonreviews.tsv', sep='\t')
df.head()

df['label'].value_counts()

# Check if no NAN values
df.isnull().sum()
df.dropna(inplace=True)

blanks = []

for i, lb, rv in df.itertuples():
    # (Index, label, review)
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)

df.drop(blanks, inplace=True)


sid.polarity_scores(df.iloc[0]['review'])

df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))

df.head()

df['compound'] = df['scores'].apply(lambda d: d['compound'])

df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')
df.head()

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy_score(df['label'], df['comp_score'])

print(classification_report(df['label'], df['comp_score']))

print(confusion_matrix(df['label'], df['comp_score']))


