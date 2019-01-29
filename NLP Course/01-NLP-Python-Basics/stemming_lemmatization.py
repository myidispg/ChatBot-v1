#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 18:46:27 2019

@author: myidispg
"""

import nltk
from nltk.stem.porter import PorterStemmer

p_stemmer = PorterStemmer()

words = ['run', 'runner', 'ran', 'runs', 'easily', 'fairly', 'fairness']
for word in words:
    print(word + '------>' + p_stemmer.stem(word))
    
from nltk.stem.snowball import SnowballStemmer

s_stemmer = SnowballStemmer(language='english')

for word in words:
    print(word + '------->' + s_stemmer.stem(word))

# ------------LEMMATIZATION----------
    
import spacy
nlp = spacy.load('en_core_web_sm')

doc1 = nlp(u"I am a runner running in a race because I love to run since I ran today.")
for token in doc1:
    print(token.text, '\t', token.pos_, '\t', token.lemma, '\t', token.lemma_)
    
def show_lemmas(text):
    for token in text:
        print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma:<{22}} {token.lemma_}')
        
doc2 = nlp(u'I saw ten mice today!')
show_lemmas(doc2)