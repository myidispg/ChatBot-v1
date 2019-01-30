#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:55:48 2019

@author: myidispg
"""

import spacy

nlp = spacy.load('en_core_web_sm')

doc = nlp(u'The quick brown fox jumped over the lazy dog\'s back.')

print(doc.text)

# Print the fifth word and associated tags:
print(doc[4].text, doc[4].pos_, doc[4].tag_, spacy.explain(doc[4].tag_))

for token in doc:
    print(f'{token.text:{10}} {token.pos_:{8}} {token.tag_:{6}} {spacy.explain(token.tag_)}')

# The below 2 docs show the difference in context based on the minute changes.   
doc = nlp(u'I read books on NLP.')
r = doc[1]

print(f'{r.text:{10}} {r.pos_:{8}} {r.tag_:{6}} {spacy.explain(r.tag_)}')

doc = nlp(u'I read a book on NLP.')
r = doc[1]

print(f'{r.text:{10}} {r.pos_:{8}} {r.tag_:{6}} {spacy.explain(r.tag_)}')

doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")

# Count the frequenies of different coarse-grained POS tags:
POS_counts = doc.count_by(spacy.attrs.POS)
POS_counts

# Check what the keys in the POS_counts dictionary means
doc.vocab[89].text

doc[2].pos
doc[2].pos_

for k,v in sorted(POS_counts.items()):
    print(f'{k}. {doc.vocab[k].text:{5}}: {v}')
    
# Count the different fine-grained tags:
TAG_counts = doc.count_by(spacy.attrs.TAG)

for k,v in sorted(TAG_counts.items()):
    print(f'{k}. {doc.vocab[k].text:{5}}: {v}')
    
# Count the different dependencies:
DEP_counts = doc.count_by(spacy.attrs.DEP)

for k,v in sorted(DEP_counts.items()):
    print(f'{k}. {doc.vocab[k].text:{4}}: {v}')
    
# --------VISUALISATION---------------------

from spacy import displacy
displacy.serve(doc, style='dep')

options = {'distance': 110, 'compact': True, 'color': 'yellow', 'bg': '#09a3d5', 'font': 'Times'}
           
displacy.serve(doc, style='dep', options=options)


doc2 = nlp(u"This is a sentence. This is another, possibly longer sentence.")
# Create spans from Doc.sents:
spans = list(doc2.sents)
displacy.serve(spans, style='dep', options={'distance': 110})
options = {'distance': 110, 'compact': 'True', 'color': 'yellow', 'bg': '#09a3d5', 'font': 'Times'}
displacy.serve(doc, style='dep', options=options)