#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:49:01 2019

@author: myidispg
"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2
        
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            

hindi_data_path = '../../Datasets/English-Hindi-IIT/parallel/IITB.en-hi.hi'
english_data_path = '../../Datasets/English-Hindi-IIT/parallel/IITB.en-hi.en'

hindi_lines = open(hindi_data_path).read().strip().split('\n')
english_lines = open(english_data_path).read().strip().split('\n')

hindi_data = [(word for word in sentence.split(' ')) for sentence in hindi_lines]

hindi_data = []

for sentence in hindi_lines:
    word_sequence = ()
    for word in sentence.split(' '):
        word_sequence += tuple(word)
    hindi_data.append(word_sequence)
