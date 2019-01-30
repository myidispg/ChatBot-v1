#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:45:26 2019

@author: myidispg
"""

import spacy
nlp = spacy.load('en_core_web_sm')

with open('../UPDATED_NLP_COURSE/TextFiles/owlcreek.txt') as f:
    doc = nlp(f.read())
    
# Verifying if everything is correct
doc[:36]

# number of tokens
print(f'Number of tokens- {len(doc)}')

# Number of sentences in the documents
sentences = list(doc.sents)
print(f'Number of sentences in the document- {len(sentences)}')
# Second sentence
print(f'The second sentence is- \n{sentences[1]}\n')

for token in sentences[1]:
    print(f'{token.text:{15}} {token.pos_:{5}} {token.dep_:{7}} {token.lemma_:{10}}')
    
# Import the Matcher library:
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

# Pattern for swimming vigorously
pattern = [{'LOWER': 'swimming'}, {'IS_SPACE': True}, {'LOWER': 'vigorously'}]

# Add pattern to the matcher
matcher.add('swimming_vigorously', None, pattern)

found_matches = matcher(doc)

# Print text surrounding each pattern.
for match_id, start, end in found_matches:
    span = doc[start-8: end+8]
    print(span.text, '\n')

# Print the sentence of each match
for match_id, start, end in found_matches:
    for sent in sentences:
        if end < sent.end:
            print(sent, '\n')
            break