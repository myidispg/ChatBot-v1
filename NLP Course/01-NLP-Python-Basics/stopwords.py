#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:09:34 2019

@author: myidispg
"""
import spacy

nlp = spacy.load('en_core_web_sm')

print(nlp.Defaults.stop_words)

len(nlp.Defaults.stop_words)

nlp.vocab['is'].is_stop # Returns true if 'is' is a stop word.