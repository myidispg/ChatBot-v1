#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:03:53 2019

@author: myidispg
"""

import spacy
nlp = spacy.load('en_core_web_sm')

with open('../UPDATED_NLP_COURSE/TextFiles/peterrabbit.txt') as f:
    doc = nlp(f.read())
    
# Verifying if everything is correct
doc[:36]

# Segment into sentences
doc_sents = [sent for sent in doc.sents]

# For every token in the third sentence, print the token text, the POS tag, 
# the fine-grained TAG tag, and the description of the fine-grained tag.
for token in doc_sents[2]:
    print(f'{token.text:{10}} {token.pos_:{8}} {token.tag_:{6}} {spacy.explain(token.tag_)}')
    
# Provide a frequency list of POS tags from the entire document
POS_counts = doc.count_by(spacy.attrs.POS)
print(POS_counts)
something = doc.vocab[91].text
# What percentage are NOUNS?
count = 0
total = 0
for key, value in POS_counts.items():
    if doc.vocab[key].text == 'NOUN':
        count = value 
        total += value
    else:
        total += value
        
print(f'Percentage of nouns- {(count/total) * 100}')

# Display the Dependency Parse for the third sentence
from spacy import displacy

displacy.serve(list(doc.sents)[2], style='dep', options={'distance': 110})

# Show the first two named entities from Beatrix Potter's The Tale of Peter Rabbit
doc_ents = doc.ents 

for i in range(2):
    print(doc_ents[i].text+' - '+doc_ents[i].label_+' - '+str(spacy.explain(doc_ents[i].label_)))
    
# How many sentences are contained in The Tale of Peter Rabbit?
print(len(doc_sents))

# CHALLENGE: Display the named entity visualization for list_of_sents[0] from the previous problem
displacy.serve(doc_sents[0], style='ent', )