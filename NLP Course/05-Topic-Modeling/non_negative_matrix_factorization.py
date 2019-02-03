#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 15:11:09 2019

@author: myidispg
"""

import pandas as pd
npr = pd.read_csv('npr.csv')

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df = 0.95, min_df=2, stop_words='english')

dtm = tfidf.fit_transform(npr['Article'])

from sklearn.decomposition import NMF
nmf_model = NMF(n_components = 7, random_state=42)
nmf_model.fit(dtm)

len(tfidf.get_feature_names())

import random
for i in range(10):
    random_word_id = random.randint(0, 54776)
    print(tfidf.get_feature_names()[random_word_id])
    
len(nmf_model.components_)

single_topic = nmf_model.components_[0]
# Returns the indices that would sort this array.
single_topic.argsort()
# Top 10 words for this topic:
single_topic.argsort()[-10:]

# print the top 10 words for all 7 topics
for index, topic in enumerate(nmf_model.components_):
    print(f'The 10 top words for topic #{index} are-')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')
    
# Attaching Discovered Topic Labels to Original Articles
dtm

topic_results = nmf_model.transform(dtm)
topic_results.shape

npr['Topic'] = topic_results.argmax(axis=1)


npr.head()
