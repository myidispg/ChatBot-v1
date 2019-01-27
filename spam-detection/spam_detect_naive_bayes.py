#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 14:35:14 2019

@author: myidispg
"""

from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

data = pd.read_csv('spambase-dataset/spambase.data').values
np.random.shuffle(data)

X = data[:, :58]
Y = data[:, -1]

X_train = X[:-100,]
y_train = Y[:-100,]

Xtest = X[-100:,]
Ytest = Y[-100:,]

model = MultinomialNB()
model.fit(X_train, y_train)
print('Classification rate for NB: {}'.format(model.score(Xtest, Ytest)))

from sklearn.ensemble import AdaBoostClassifier

model =AdaBoostClassifier()
model.fit(X_train, y_train)
print('Classification rate for AdaBoost: {}'.format(model.score(Xtest, Ytest)))
