#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 18:52:52 2019

@author: myidispg
"""

person = 'Prashant'

print(f'my name is {person}')

mylist = {'a': 123, 'b': 456}

print(f"{mylist['a']}")

library = [
        ('Author', 'Topic', 'Pages'),
        ('Twain', 'Rafting in water alone', 601),
        ('Feynman', 'Physics', 95),
        ('Hamilton', 'Mythology', 144)
        ]

for author, topic, pages in library:
    print(f"{author:{10}}{topic:{30}}{pages:.>{10}}")
    
from datetime import datetime


today = datetime(year=2019, month=1, day=28)

print(f"{today:%d %B %Y}")
today