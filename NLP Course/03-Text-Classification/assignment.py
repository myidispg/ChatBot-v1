#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:00:53 2019

@author: myidispg
"""

import numpy as np
import pandas as pd

df = pd.read_csv('../UPDATED_NLP_COURSE/TextFiles/moviereviews2.tsv', sep='\t')
df.head()

# Check for missing values
df.isnull().sum()
df.dropna(inplace=True)

# Check for whitespace strings
blanks = []

#(index, label, review text)
for i, lb, rv in df.itertuples():
    if rv.isspace():
        blanks.append(i)

# Since there are no blanks, no need to drop them.

# Quick look at labels column
df['label'].value_counts()

from sklearn.model_selection import train_test_split
X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)


# Build pipelines to vectorize the data, then train and fit a model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Linear SVM
text_clf_svm = Pipeline([('tfidf', TfidfVectorizer()),
                       ('clf', LinearSVC())])
    
text_clf_svm.fit(X_train, y_train)

# Form a prediction set
predictions = text_clf_svm.predict(X_test)

# Report the confusion matrix
from sklearn import metrics
print(metrics.confusion_matrix(y_test,predictions))
    
# Print a classification report
print(metrics.classification_report(y_test,predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))
