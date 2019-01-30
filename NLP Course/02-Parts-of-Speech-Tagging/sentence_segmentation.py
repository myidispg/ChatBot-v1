#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 20:27:31 2019

@author: myidispg
"""

# Perform standard imports
import spacy
nlp = spacy.load('en_core_web_sm')
# From Spacy Basics:
doc = nlp(u'This is the first sentence. This is another sentence. This is the last sentence.')

for sent in doc.sents:
    print(sent)
    
doc_sents = [sent for sent in doc.sents]
doc_sents

# Now you can access individual sentences:
print(doc_sents[1])

type(doc_sents[1])

print(doc_sents[1].start, doc_sents[1].end)

"""
Adding Rules
spaCy's built-in sentencizer relies on the dependency parse and end-of-sentence
 punctuation to determine segmentation rules. We can add rules of our own, 
 but they have to be added before the creation of the Doc object, as that is 
 where the parsing of segment start tokens happens:
"""

def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == ';':
            doc[token.i+1].is_sent_start = True # Token.i is index.
    return doc

nlp.add_pipe(set_custom_boundaries, before='parser')
nlp.pipe_names
# Parsing the segmentation start tokens happens during the nlp pipeline
# Re-run the Doc object creation:
doc4 = nlp(u'"Management is doing things right; leadership is doing the right things." -Peter Drucker')

for sent in doc4.sents:
    print(sent)
    
    
"""
Changing the Rules
In some cases we want to replace spaCy's default sentencizer with our own set
 of rules. In this section we'll see how the default sentencizer breaks on 
 periods. We'll then replace this behavior with a sentencizer that breaks on 
 linebreaks.
"""

mystring = u"This is a sentence. This is another.\n\nThis is a \nthird sentence."

# SPACY DEFAULT BEHAVIOR:
doc = nlp(mystring)

for sent in doc.sents:
    print([token.text for token in sent])
    
# CHANGING THE RULES
from spacy.pipeline import SentenceSegmenter

def split_on_newlines(doc):
    start = 0
    seen_newline = False    
    for word in doc:
        if seen_newline:
            yield doc[start:word.i]
            start = word.i
            seen_newline = False
        elif word.text.startswith('\n'): # handles multiple occurrences
            seen_newline = True
    yield doc[start:]      # handles the last group of tokens


sbd = SentenceSegmenter(nlp.vocab, strategy=split_on_newlines)
nlp.add_pipe(sbd)

doc = nlp(mystring)
for sent in doc.sents:
    print([token.text for token in sent])