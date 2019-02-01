#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:22:33 2019

@author: myidispg
"""

# Import spaCy and load the language library
import spacy
nlp = spacy.load('en_core_web_md')  # make sure to use a larger model!

# Check the vector for the word lion
nlp(u'lion').vector

"""Doc and span objects also have vectors derived from the averages of the token vectors."""
doc = nlp(u'The quick brown fox jumped over the lazy dogs.')
doc.vector

# ---------------Identifying similar vectors----------------------------
# Create a three-token Doc object:
tokens = nlp(u'lion cat pet')

# Iterate through token combinations:
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# -----Vector norms------------
"""
It's sometimes helpful to aggregate 300 dimensions into a Euclidian (L2) norm,
 computed as the square root of the sum-of-squared-vectors. This is accessible
 as the .vector_norm token attribute. Other helpful attributes 
 include .has_vector and .is_oov or out of vocabulary.
"""


tokens = nlp(u'dog cat nargle')

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
# oov- Out of Vocabulary
    
    
# ----Vector arithmetic--------
from scipy import spatial

cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

king = nlp.vocab['king'].vector
man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector

# Now we find the closest vector in the vocabulary to the result of "man" - "woman" + "queen"
new_vector = king - man + woman
computed_similarities = []

for word in nlp.vocab:
    # Ignore words without vectors and mixed-case words:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_vector, word.vector)
                computed_similarities.append((word, similarity))

computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

print([w[0].text for w in computed_similarities[:10]])


    
