#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 20:08:00 2019

@author: myidispg
"""

import spacy

nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)

# SolarPower
pattern1 = [{'LOWER': 'solarpower'}]
# Solar-power
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}]
# Solar Power
pattern3 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]

# Add the above patterns to the matcher.
matcher.add('SolarPower', None, pattern1, pattern2, pattern3)

doc= nlp(u'The Solar Power industry continues to grow as solarpower increases. Solar-power is amazing.')

found_matches = matcher(doc)
print(found_matches)

for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(match_id, string_id, start, end, span.text)
    
# Remove the pattern
matcher.remove('SolarPower')

# SolarPower, solarpower
pattern1 = [{'LOWER': 'solarpower'}]
# Solar*power *- punctuation 0 or more times.
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'OP': '*'}, {'LOWER': 'power'}]

matcher.add('Solarpower', None, pattern1, pattern2)
doc2 = nlp(u'Solar--pwer is solarpower yay.')

found_matches = matcher(doc2)

from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab)

with open('../UPDATED_NLP_COURSE/TextFiles/reaganomics.txt') as f:
    doc3 = nlp(f.read())
    
phrase_list = ['voodooeconomics', 'supply-side economics', 'trickle-down economics', 'free-market economics']
phrase_patterns = [nlp(text) for text in phrase_list]

matcher.add('EconMatcher', None, *phrase_patterns)

found_matches = matcher(doc3)

for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(match_id, string_id, start, end, span.text)