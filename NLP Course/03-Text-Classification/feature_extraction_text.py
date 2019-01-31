#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 20:17:12 2019

@author: myidispg
"""

# Write two text files

%%writefile 1.txt
This is a story about cats
our feline pets
Cats are furry animals

%%writefile 2.txt
This story is about surfing
Catching waves is fun
Surfing is a popular water sport


#------Build a vocabulary------------------
vocab = {}
i = 1
with open('1.txt') as f:
    x = f.read().lower().split()

for word in x:
    if word in vocab:
        continue
    else:
        vocab[word] = i
        i += 1

print(vocab)

with open('2.txt') as f:
    x = f.read().lower().split()

for word in x:
    if word in vocab:
        continue
    else:
        vocab[word] = i
        i += 1

print(vocab)

#----------FEATURE EXTRACTION--------------
# Create an empty vector with space for each word in the vocabulary:
one = ['1.txt']+[0]*len(vocab)
one

# map the frequencies of each word in 1.txt to our vector:
with open('1.txt') as f:
    x = f.read().lower().split()
    
for word in x:
    one[vocab[word]]+=1
    
one

# Do the same for the second document:
two = ['2.txt']+[0]*len(vocab)

with open('2.txt') as f:
    x = f.read().lower().split()
    
for word in x:
    two[vocab[word]]+=1
    
# Compare the two vectors:
print(f'{one}\n{two}')

# It is recommended to restart kernel now.

# -------FEATURE EXTRACTION FROM TEXT------------
# Perform imports and load the dataset:
import numpy as np
import pandas as pd

df = pd.read_csv('../UPDATED_NLP_COURSE/TextFiles/smsspamcollection.tsv', sep='\t')
df.head()

# See if there are any missing values
df.isnull().sum()

# Get the count of each values in the labels column
df['label'].value_counts()

# Split into test and train set
from sklearn.model_selection import train_test_split

X = df['message']  # this time we want to look at the text
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Scikit-learn's CountVectorizer
"""Text preprocessing, tokenizing and the ability to filter out stopwords are 
all included in CountVectorizer, which builds a dictionary of features and 
transforms documents to feature vectors.
"""

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape

# Transform Counts to Frequencies with Tf-idf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

# Combine Steps with TfidVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train) # remember to use the original X_train set
X_train_tfidf.shape

# Train a Classifier
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train_tfidf,y_train)

# Build a Pipeline
from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)  

# Test the classifier and display results
predictions = text_clf.predict(X_test)

# Test on a dummy spam message
print(text_clf.predict(['Congratulations, you have won a luck draw. Share your card details to claim your prize.']))
# ^^ Worked fine.

# Report the confusion matrix
from sklearn import metrics
print(metrics.confusion_matrix(y_test,predictions))

# Print a classification report
print(metrics.classification_report(y_test,predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))