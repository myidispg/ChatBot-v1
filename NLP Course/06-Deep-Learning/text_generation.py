#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:07:09 2019

@author: myidispg
"""

def read_files(filepath):
    with open(filepath) as f:
        str_text = f.read()
        
    return str_text


