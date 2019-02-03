#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:46:07 2019

@author: myidispg
"""

import pandas as pd
import numpy as np

npr = pd.read_csv('npr.csv')

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')

dtm = cv.fit_transform(npr['Article'])

dtm

from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=7, random_state=42)

LDA.fit(dtm)

# Grab the vocabulary of the words
len(cv.get_feature_names())
type(cv.get_feature_names())

import random
random_word_id = random.randint(0, 54777)

cv.get_feature_names()[random_word_id]

# Grab the topics
len(LDA.components_)
type(LDA.components_)
LDA.components_.shape
LDA.components_

single_topic = LDA.components_[0]
single_topic.argsort()
# Returns the index positions sorted from least to greatest
single_topic.argsort()[-10:] # grabs the last 10 values
top_ten_words = single_topic.argsort()[-15:]
for index in top_ten_words:
    print(cv.get_feature_names()[index])    

# Grab the highest probablity words per topic
for index, topic in enumerate(LDA.components_):
    print(f"The top 15 words for topic #{index} are- ")
    print([cv.get_feature_names()[i] for i in topic.argsort()[15:]])
    print('\n')
    
# Attaching Discovered Topic Labels to Original Articles
dtm
dtm.shape
len(npr)
topic_results = LDA.transform(npr)
topic_results.shape # Result is 11992x7 => Number of docs x Number of topics
topic_results[0].argmax() # this shows the topic our first article belongs to.

# Combining with original data
topic_results.argmax(axis=1)
npr['Topic'] = topic_results.argmax(axis=1)
npr.head(10)
