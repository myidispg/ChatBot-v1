#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:42:10 2019

@author: myidispg
"""

import numpy as np
import pandas as pd

df = pd.read_csv('../UPDATED_NLP_COURSE/TextFiles/moviereviews.tsv', sep='\t')
df.head()
len(df)
print(df['review'][0])

# Check if there are any missing values
df.isnull().sum()
#Drop the NAN values
df.dropna(inplace=True)
df.isnull().sum()

# Check if there are any blank review
blanks = []

#(index, label, review text)
for i, lb, rv in df.itertuples():
    if rv.isspace():
        blanks.append(i)
        
# Drop the blank indexes
df.drop(blanks, inplace=True)

from sklearn.model_selection import train_test_split
X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

# Build pipelines to vectorize the data, then train and fit a model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Naive Bayes
text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()),
                       ('clf', MultinomialNB())])
# Linear SVC
text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', LinearSVC())])
    
# ---------Test the first pipeline-----------------------
text_clf_nb.fit(X_train, y_train)

# Form a prediction set
predictions = text_clf_nb.predict(X_test)

# Report the confusion matrix
from sklearn import metrics
print(metrics.confusion_matrix(y_test,predictions))
    
# Print a classification report
print(metrics.classification_report(y_test,predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))

# -----Test the second pipeline----------------------------
text_clf_lsvc.fit(X_train, y_train)

# Form a prediction set
predictions = text_clf_lsvc.predict(X_test)

# Report the confusion matrix
from sklearn import metrics
print(metrics.confusion_matrix(y_test,predictions))

# Print a classification report
print(metrics.classification_report(y_test,predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))