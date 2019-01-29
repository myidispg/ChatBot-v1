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