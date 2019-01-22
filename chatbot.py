#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 20:06:46 2019

@author: myidispg
"""

# Building a chatbot with Deep NLP


# Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time


#########- Part 1-DATA PREPROCESSING ########

# Import the dataset
lines = open('cornell movie-dialogs corpus/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversations = open('cornell movie-dialogs corpus/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# Creating a dictionary that maps each ine with its id
id2line = {}

for line in lines:
    _line = line.split('+++$+++')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
        
# Creating a list of all the conversations
conversations_ids = []

for conversation in conversations[:-1]:
    _conversation = conversation.split('+++$+++')[-1][2:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))
    