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
        self.max_ent_length = 1
        
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

pairs = []

for hindi_sent, english_sent in zip(hindi_lines, english_lines):
    pairs.append([hindi_sent, english_sent])
    
hindi_lang = Lang('hindi')
english_lang = Lang('english')

def addWordsToLang(lang, lines):
    for line in lines:
        lang.addSentence(line)
    
    return lang

addWordsToLang(english_lang, english_lines)

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
    lang2 = addWordsToLang(lang1, lang1_lines)
    
    return pairs, lang1, lang2

pairs, hindi_lang, english_lang = createLanguagesAndPairs(hindi_data_path, english_data_path, hindi_lang, english_lang)


