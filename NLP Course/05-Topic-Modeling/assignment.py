#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 15:28:28 2019

@author: myidispg
"""

import pandas as pd
quora = pd.read_csv('quora_questions.csv')
quora.head()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(quora['Question'])

dtm

from sklearn.decomposition import NMF
nmf_model = NMF(n_components=20, random_state=42)
nmf_model.fit(dtm)

len(tfidf.get_feature_names())

# Print the top 15 words for all the topics
for index, topic in enumerate(nmf_model.components_):
    print(f'The top 15 WORDS for topic #{index} are-')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')

# Attach the discovered topics to the questions
topic_results = nmf_model.transform(dtm)
topic_results.shape
quora['topic'] = topic_results.argmax(axis=1)
quora.head()

