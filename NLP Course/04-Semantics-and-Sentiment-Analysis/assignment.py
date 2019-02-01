#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:26:56 2019

@author: myidispg
"""

import spacy

nlp = spacy.load('en_core_web_md')

#---Vector arithmatic------------
wolf = nlp.vocab['wolf'].vector
dog = nlp.vocab['dog'].vector
cat = nlp.vocab['cat'].vector


from scipy import spatial
def cosine_similarity(x, y):
    return 1 - spatial.distance.cosine(x,y)

new_vector = wolf - dog + cat
computed_similarities = []

for word in nlp.vocab:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                computed_similarities.append((word, cosine_similarity(new_vector, word.vector)))
                
computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

print([w[0].text for w in computed_similarities[:10]])

# Perform vector arithmetic with a single function

def vector_math(a,b,c):
    a = nlp.vocab[a].vector
    b = nlp.vocab[b].vector
    c = nlp.vocab[c].vector
    
    new_vector = a - b + c
    computed_similarities = []
    
    for word in nlp.vocab:
        if word.has_vector:
            if word.is_lower:
                if word.is_alpha:
                    computed_similarities.append((word, cosine_similarity(new_vector, word.vector)))
    
    computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

    print([w[0].text for w in computed_similarities[:10]])

# Test the function on known words:
vector_math('king','man','woman')


#---------Task #2: Perform VADER Sentiment Analysis on your own review--------
# Import SentimentIntensityAnalyzer and create an sid object
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
review = 'This movie portrayed real people, and was based on actual events.'

sid.polarity_scores(review)

def review_rating(string):
    score = sid.polarity_scores(string)['compound']
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    return 'Neutral'

review_rating(review)    

                    