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
#from keras.layers.embeddings import Embedding
from keras.callbacks import ReduceLROnPlateau

import tensorflow as tf
from tensorflow.keras.layers import Embedding

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
        
# Initially, keras will be used for implementing LSTM. Later, this will be replaced by Tensorflow model

# The below function creates a Keras embedding layer with the downloaded embedding weights
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1 # +1 for keras
    emb_dim = word_to_vec_map['happy'].shape[0] # dimensions of embeddings
    
    emb_matrix = np.zeros((vocab_len, emb_dim)) # initialize with zeros
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
        
    # Define keras embedding layer with the correct output/input sizes
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    # Build the embedding layer
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

# EMBEDDING LAYER WITH TENSORFLOW
def tensorflow_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map['happy'].shape[0] # dimensions of embeddings
    emb_matrix = np.zeros((vocab_len, emb_dim)) # initialize with zeros
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
        
    # Define the embedding layer with the correct output/input sizes
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    # Build the embedding layer
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer
    
    
# We now need to convert all training sentences into lists of indices, and then zero-pad all these lists so that their length is the length of the longest sentence.
def sentences_to_indices(x, word_to_index, max_len):
    m = x.shape[0] # number of training samples
    x_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = (x[i].lower()).split() # split sentence into words
        j = 0
        for w in sentence_words:
            x_indices[i, j] = word_to_index[w]
            j = j + 1
            
    return x_indices

x_train_indices = sentences_to_indices(x_train, word_to_index, maxWords)

# EMOJIFIER MODEL
def emojify(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(shape=input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)
    
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(5, activation='softmax')(X)
    X = Activation('softmax')(X)
    
    model = Model(inputs=sentence_indices, outputs=X)
    
    return model

# EMOJIFIER MODEL IN TENSORFLOW
def emojify_tensorflow(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = tf.placeholder(shape=input_shape, dtype=tf.int32)
    embedding_layer = tensorflow_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)
    print(embedding_layer)
    
    input_data = tf.placeholder(shape=input_shape, dtype=tf.int32)
    output = tf.placeholder(shape = [5, ], dtype=tf.int32)
    
    
    
    
    
emojify_tensorflow([maxWords,], word_to_vec_map, word_to_index)

emojifier = emojify((maxWords,), word_to_vec_map, word_to_index)
emojifier.summary()
    
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
emojifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
emojifier.fit(x_train_indices, y_train_one_hot, epochs = 100, batch_size = 16, shuffle=True, 
                               callbacks=[reduce_lr])

# Testing the model
x_test_indices = sentences_to_indices(x_test, word_to_index, max_len=maxWords)
y_test_one_hot = pd.get_dummies(y_test)
loss, acc = emojifier.evaluate(x_test_indices, y_test_one_hot)
print()
print("Test accuracy = ", acc)

pred = emojifier.predict(x_test_indices)
for i in range(len(x_test)):
    x = x_test_indices
    num = np.argmax(pred[i])
    if(num != y_test[i]):
        print(f'Expected emoji: {label_to_emoji(y_test[i])}  prediction:  {x_test[i]}  {label_to_emoji(num).strip()}')
