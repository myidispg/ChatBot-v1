#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 14:24:34 2019

@author: myidispg
"""
# Load doc into memory
def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text

# split a loaded document into sentences
def to_sentences(doc):
    return doc.strip().split('\n')

# shortest and longest sentences
def sentence_lengths(sentences):
    lengths = [len(s.split()) for s in sentence]
    return min(lengths), max(lengths)

