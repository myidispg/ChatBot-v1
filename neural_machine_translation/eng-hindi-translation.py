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
        self.max_sent_length = 1
        
    def addSentence(self, sentence):
        sent_length = len(sentence.split(' '))
        self.max_sent_length = sent_length if sent_length > self.max_sent_length else self.max_sent_length        
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

hindi_lang = Lang('hindi')
english_lang = Lang('english')

def addWordsToLang(lang, lines):
    for line in lines:
        lang.addSentence(line)
    
    return lang

def create_pairs(lang1, lang2):
    pairs = []

    for lang1_sent, lang2_sent in zip(lang1, lang2):
        pairs.append([lang1_sent, lang2_sent])
        
    return pairs

def createLanguagesAndPairs(lang1_path, lang2_path, lang1, lang2):
    print('Opening files and reading the sentences')
    lang1_lines = open(hindi_data_path).read().strip().split('\n')
    lang2_lines = open(english_data_path).read().strip().split('\n')
    
    print('Creating pairs...')
    pairs = create_pairs(lang1_lines, lang2_lines)
    
    print('Adding words to languages')
    lang1 = addWordsToLang(lang1, lang1_lines)
    lang2 = addWordsToLang(lang2, lang2_lines)
    
    print('Done creating languages')
    
    return pairs, lang1, lang2

pairs, hindi_lang, english_lang = createLanguagesAndPairs(hindi_data_path, english_data_path, hindi_lang, english_lang)

class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # The dimensions of the embedding is same as hidden size.
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
    def forward(self, input, hidden_state):
        embedded = self.embedded(input).view(1, 1, -1)
        output = embedded
        output, hidden_state = self.gru(output, hidden_state)
        return output, hidden_state
    
    def initHidden(self):
        return torch.randn(1, 1, self.hidden_size, device=device)
    
class DecoderRNN(nn.Module):
    
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden_state):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden_state)
        output = self.softmax(self.out(output[0]))
        return output, hidden_state
    
    def initHidden(self):
        return torch.randn(1, 1, self.hidden_size, device=device)
    

        
        
        
