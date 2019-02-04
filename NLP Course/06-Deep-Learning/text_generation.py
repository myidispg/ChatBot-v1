#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:07:09 2019

@author: myidispg
"""

def read_files(filepath):
    with open(filepath) as f:
        str_text = f.read()
        
    return str_text

# Tokenize and Clean Text
import spacy

nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

nlp.max_length = 1198623

def seperate_punctuation(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']
            
d = read_files('moby_dick_four_chapters.txt')

tokens = seperate_punctuation(d)


# Create Sequences of Tokens
# Input 25 words and the network predicts the 26th word.
train_len = 25 + 1

text_sequences = []

for i in range(train_len, len(tokens)):
    seq= tokens[i-train_len: i]
    text_sequences.append(seq)
    
# Keras Tokenization
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)

sequences = tokenizer.texts_to_sequences(text_sequences)

for i in sequences[0]:
    print(f'{i} : {tokenizer.index_word[i]}')
    
tokenizer.word_counts

vocabulary_size = len(tokenizer.word_counts)

import numpy as np
sequences = np.array(sequences)

# Creating an LSTM based model
























