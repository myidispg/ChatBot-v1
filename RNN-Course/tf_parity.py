#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:53:59 2019

@author: myidispg
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell

from sklearn.utils import shuffle
from util import all_parity_pairs, all_parity_pairs_with_sequence_labels

def x2sequence(x, T, D, batch_size):
    # Permuting batch_size, and n_steps
    x = tf.transpose(x, (1, 0, 2)) # NxTxD => TxNxD
    # reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, (T*batch_size, D))
    # Split to get a list of n_steps tensors of shape(batch_size, n_input)
    x = tf.split(x, T)
    
    return x

X, Y = all_parity_pairs_with_sequence_labels(10)
x = x2sequence(X, 10, 1, 1100)