#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 12:06:17 2019

@author: myidispg
"""

import numpy as np
import pandas as pd
import emoji
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.callbacks import ReduceLROnPlateau

%matplotlib inline

import os
print(os.listdir('../Datasets'))


train = pd.read_csv('../Datasets/emojify-data/train_emoji.csv', header=None, usecols=[0,1])
test = pd.read_csv('../Datasets/emojify-data/test_emoji.csv', header=None, usecols=[0,1])

# Split train and test into X and Y
x_train, y_train = train[0], train[1]
x_test, y_test = test[0], test[1]
print(f'Shape of X is: {x_train.shape}')
print(f'Shape of Y is: {x_test.shape}')

# Check a random sentence and the emoji associated with it
emoji_dict = {
        "0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
        "1": ":baseball:",
        "2": ":smile:",
        "3": ":disappointed:",
        "4": ":fork_and_knife:"
        }


def label_to_emoji(label):
    return emoji.emojize(emoji_dict[str(label)], use_aliases=True)

print(x_train[20], label_to_emoji(y_train[20]))

# Finding the max sentence length in the dataset
maxWords = len(max(x_train, key=len).split())
print('Maximum words in sentence are:',maxWords)

# Convert Y to one-hot vectors
y_train_one_hot = pd.get_dummies(y_train)
print(y_train_one_hot.shape)

# Use glove vectors for representing the words in the sentences.
# The downloaded files encodign files are in a seperate directory
def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set() # ensure unique words
        word_to_vec_map = {} # this will map words to vectors
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
        i = 1
        words_to_index = {} # dictionary mapping words to their index in the dictionary
        index_to_words = {}   # dictionary mapping index to the word in the dictionary
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i += 1
        return words_to_index, index_to_words, word_to_vec_map
    
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('../Datasets/glove-global-vectors-for-word-representation/glove.6B.100d.txt')
        




