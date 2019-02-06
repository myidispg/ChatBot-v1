#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:55:29 2019

@author: myidispg
"""

import pickle
import numpy as np

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


# --------VECTORIZE THE DATA------------------
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)
tokenizer.word_index # Check the tokenized indexes

train_story_text = []
train_ques_text = []
train_ans_text = []

for story, ques, answer in train_data:
    train_story_text.append(story)
    train_ques_text.append(ques)
    train_ans_text.append(answer)

# Creates a sequence but the words are now replaced by the indexed numbers
train_story_seq = tokenizer.texts_to_sequences(train_story_text)

len(train_story_seq)

def vectorize(data, word_index=tokenizer.word_index, max_story_len=max_story_len, max_ques_len=max_ques_len):
    
    # Stories
    X = []
    # questions
    Xq = []
    # answers
    Y = []
    
    for story, ques, answer in data:
        # for stories
        x = [word_index[word.lower()] for word in story]
        # questions
        xq = [word_index[word.lower()] for word in ques]
        
        y = np.zeros(len(word_index) + 1)
        y[word_index[answer]] = 1
        
        X.append(x)
        Xq.append(xq)
        Y.append(y)
        
    return (pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_ques_len), np.array(Y))

inputs_train, queries_train, answers_train = vectorize(train_data)
inputs_test, queries_test, answers_test = vectorize(test_data)

# ------BUILD THE MODEL-----------------------
    
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM

input_sequence = Input(shape=(max_story_len, ))
question = Input(shape=(max_ques_len, ))

vocab_size = len(vocab) + 1

# INPUT ENCODER M
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64))
input_encoder_m.add(Dropout(0.3))
# OUTPUT
# (samples, story_max_len, embedding_dim)

# INPUT ENCODER C
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=max_ques_len))
input_encoder_c.add(Dropout(0.3))
# OUTPUT
# (samples, story_max_len, embedding_dim)

# QUESTION ENCODER M
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size, output_dim=64, input_length = max_ques_len))
question_encoder.add(Dropout(0.3))
# OUTPUT
# (samples, question_max_len, embedding_dim)

# ENCODED <----- ENCODER(INPUT)
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

match = dot([input_encoded_m, question_encoded], axes=(2,2))
match = Activation('softmax')(match)

response = add([match, input_encoded_c])
response = Permute((2,1))(response)

answer = concatenate([response, question_encoded])

# Reduce with RNN (LSTM)
answer = LSTM(32)(answer)  # (samples, 32)
# Regularization with Dropout
answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
# we output a probability distribution over the vocabulary
answer = Activation('softmax')(answer)

# build the final model
model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# train
history = model.fit([inputs_train, queries_train], answers_train,batch_size=32,epochs=120,validation_data=([inputs_test, queries_test], answers_test))



















