#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 18:49:18 2019

@author: myidispg
"""
import numpy as np
import pandas as pd
df = pd.read_csv('../UPDATED_NLP_COURSE/TextFiles/smsspamcollection.tsv', sep='\t')

df.head()

df.isnull().sum() # Check if any data is missing.

len(df)

df['label'].unique()
df['label'].value_counts()

# Visualize the data
df['length'].describe()

import matplotlib.pyplot as plt
%matplotlib inline

plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df[df['label']=='ham']['length'],bins=bins,alpha=0.8)
plt.hist(df[df['label']=='spam']['length'],bins=bins,alpha=0.8)
plt.legend(('ham','spam'))
plt.show()

df['punct'].describe()

plt.xscale('log')
bins = 1.5**(np.arange(0,15))
plt.hist(df[df['label']=='ham']['punct'],bins=bins,alpha=0.8)
plt.hist(df[df['label']=='spam']['punct'],bins=bins,alpha=0.8)
plt.legend(('ham','spam'))
plt.show()

# --------Split into train and test set----------------

# selecting features

# Create Feature and Label sets
X = df[['length','punct']]  # note the double set of brackets
y = df['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print('Training Data Shape:', X_train.shape)
print('Testing Data Shape: ', X_test.shape)


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(solver='lbfgs')

lr_model.fit(X_train, y_train)

from sklearn import metrics

# Create a prediction set:
predictions = lr_model.predict(X_test)

# Print a confusion matrix
print(metrics.confusion_matrix(y_test,predictions))

df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])
df

# Print a classification report
print(metrics.classification_report(y_test,predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))

#------------Train a naïve Bayes classifier:------------
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)
predictions = nb_model.predict(X_test)
print(metrics.confusion_matrix(y_test,predictions))
print(metrics.classification_report(y_test,predictions))

print(metrics.accuracy_score(y_test,predictions))


#---------Train a Support Vector Machine classifier-----------
from sklearn.svm import SVC
svc_model = SVC(gamma='auto')
svc_model.fit(X_train,y_train)

predictions = svc_model.predict(X_test)
print(metrics.confusion_matrix(y_test,predictions))

print(metrics.classification_report(y_test,predictions))

print(metrics.accuracy_score(y_test,predictions))