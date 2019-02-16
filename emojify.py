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
