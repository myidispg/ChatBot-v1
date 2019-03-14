#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:44:25 2019

@author: myidispg
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

def main(we_file='word_embeddings.npy', w2i_file='wikipedia_word2idx.json', Model=PCA):
    We = np.load(we_file)
    V, D = We.shape
    with open(w2i_file) as f:
        word2idx = json.load(f)
        
    idx2word = {word2idx[k]: k for k in word2idx}
    
    model = Model()
    Z = model.fit_transform(We)
    plt.scatter(Z[:, 0], Z[:, 1])
    for i in range(V):
        plt.annotate(s=idx2word[i], xy=(Z[i,0], Z[i, 1]))
    plt.show()
    
main(we_file='gru_nonorm_part1_word_embeddings.npy',w2i_file='gru_nonorm_part1_wikipedia_word2idx.json', Model=TSNE)

