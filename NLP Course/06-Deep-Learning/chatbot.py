#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:55:29 2019

@author: myidispg
"""

import pickle
import numpy

with open('train_qa.txt', 'rb') as f:
    train_data = pickle.load(f)
    
with open('test_qa.txt', 'rb') as f:
    test_data = pickle.load(f)

train_data[0]

all_data = test_data + train_data
len(all_data)

vocab = set()

for story, question, answer in all_data:
    # Adds the unadded words to the vocabulary
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))

vocab.add('no')
vocab.add('yes')

vocab_len = len(vocab) + 1

# Longest story in all data
all_story_lengths = [len(data[0]) for data in all_data]
max_story_len = max(all_story_lengths)
max_ques_len = max([len(data[1]) for data in all_data])







