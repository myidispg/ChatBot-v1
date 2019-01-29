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
    
doc2 = nlp(u"We're here to help! Send snail-mail, email support@oursite.com or visit us at http://www.oursite.com")
for t in doc2:
    print(t)
    
doc3 = nlp(u"A 5km NYC cab ride costs $10.30")
for t in doc3:
    print(t)

doc4 = nlp(u"Let's visit St. Louis in the U.S. next year.")
for t in doc4:
    print(t)
len(doc4)
doc4[0]
doc4[2:5]
# Check the token vocabulary of the loaded spacy library file.
len(doc4.vocab)

doc5 = nlp(u"Apple built a Hong Kong factory for $6 million")
for token in doc5:
    print(token.text, end=' | ')
for entity in doc5.ents:
    print(entity)
    print(entity.label_)
    print(str(spacy.explain(entity.label_)))
    print('\n')

doc6 = nlp(u"Autonomous cars shift insurance laibility toward manufacturers.")
for chunk in doc6.noun_chunks:
    print(chunk)
    
# ------Visualization-------------
from spacy import displacy

doc7 = nlp("Apple is going to build a U.K. factory for $6 million.")
displacy.serve(doc7, style='dep', options={'distance':100})

doc8 = nlp(u"Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million.")
displacy.serve(doc8, style='ent')
    
    