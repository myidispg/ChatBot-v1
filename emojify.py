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