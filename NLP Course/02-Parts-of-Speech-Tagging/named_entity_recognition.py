#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 18:35:22 2019

@author: myidispg
"""

import spacy
nlp = spacy.load('en_core_web_sm')

# Write a function to display basic entity info:
def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    else:
        print('No named entities found.')
        
doc = nlp(u'May I go to Washington, DC next May to see the Washington Monument?')

show_ents(doc)

doc = nlp(u'Can I please borrow 500 dollars from you to buy some Microsoft stock?')

for ent in doc.ents:
    print(ent.text, ent.start, ent.end, ent.start_char, ent.end_char, ent.label_)
    
# Adding a Named Entity to a Span

doc = nlp(u'Tesla to build a U.K. factory for $6 million')

show_ents(doc)

# Here Tesla is not recognized as a Named entity. Adding now.
from spacy.tokens import Span

# Get the hash value of the ORG entity label
ORG = doc.vocab.strings[u'ORG']  

# Create a Span for the new entity
new_ent = Span(doc, 0, 1, label=ORG) # 0, 1 are teh indexes of the entity.

# Add the entity to the existing Doc object
doc.ents = list(doc.ents) + [new_ent]

show_ents(doc)

# Adding multiple entities to a Span
doc = nlp(u'Our company plans to introduce a new vacuum cleaner. '
          u'If successful, the vacuum-cleaner will be our first product.')

show_ents(doc)

# Import PhraseMatcher and create a matcher object:
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)

# Create the desired phrase patterns:
phrase_list = ['vacuum cleaner', 'vacuum-cleaner']
phrase_patterns = [nlp(text) for text in phrase_list]

# Apply the patterns to our matcher object:
matcher.add('newproduct', None, *phrase_patterns)

# Apply the matcher to our Doc object:
matches = matcher(doc)

# See what matches occur:
matches

# Here we create Spans from each match, and create named entities from them:
from spacy.tokens import Span

PROD = doc.vocab.strings[u'PRODUCT']

new_ents = [Span(doc, match[1],match[2],label=PROD) for match in matches]

doc.ents = list(doc.ents) + new_ents
show_ents(doc)

# Counting Entities
doc = nlp(u'Originally priced at $29.50, the sweater was marked down to five dollars.')

show_ents(doc)

len([ent for ent in doc.ents if ent.label_=='MONEY'])

# Noun Chunks
doc = nlp(u"Autonomous cars shift insurance liability toward manufacturers.")

for chunk in doc.noun_chunks:
    print(chunk.text+' - '+chunk.root.text+' - '+chunk.root.dep_+' - '+chunk.root.head.text)
    
len(list(doc.noun_chunks))