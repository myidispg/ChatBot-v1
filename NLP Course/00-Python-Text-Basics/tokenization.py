#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 18:06:04 2019

@author: myidispg
"""

import spacy

nlp = spacy.load('en_core_web_sm')

mystring = '"We\'re moving to L.A.!"'

doc = nlp(mystring)

for token in doc:
    print(token.text)