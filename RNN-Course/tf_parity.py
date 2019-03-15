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

X, Y = all_parity_pairs_with_sequence_labels(12)