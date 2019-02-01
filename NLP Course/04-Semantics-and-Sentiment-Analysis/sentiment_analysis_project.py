#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:16:22 2019

@author: myidispg
"""

import numpy as np
import pandas as pd

df = pd.read_csv('../UPDATED_NLP_COURSE/TextFiles/moviereviews.tsv', sep='\t')
df.head()

# REMOVE NaN VALUES AND EMPTY STRINGS:
df.dropna(inplace=True)

blanks = []  # start with an empty list

for i,lb,rv in df.itertuples():  # iterate over the DataFrame
    if type(rv)==str:            # avoid NaN values
        if rv.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list

df.drop(blanks, inplace=True)

df['label'].value_counts()


# Import SentimentIntensityAnalyzer and create an sid object
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Use sid to append a comp_score to the dataset
df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))

df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])

df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')

df.head()

# Perform a comparison analysis between the original label and comp_score
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy_score(df['label'],df['comp_score'])

print(classification_report(df['label'],df['comp_score']))

print(confusion_matrix(df['label'],df['comp_score']))


