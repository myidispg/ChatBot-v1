#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:47:37 2019

@author: myidispg
"""

import numpy as np
import theano 
import theano.tensor as T
import matplotlib.pyplot as plt

from util import init_weight, all_parity_pairs
from sklearn.utils import shuffle

class HiddenLayer:
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W = init_weight(M1, M2)
        b = np.zeros(M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]
        
    def forward(self, X):
        return T.nnet.relu(X.dot(self.W) + self.b)
    
class ANN:
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
        
    def fit(self, X, Y, learning_rate = 10e-3, mu=0.99, reg=10e-12, eps=10e-10, epochs=400, batch_size=20, print_period=1, show_fig=False):
        # Make the data type of y as int32
        Y = Y.astype(np.int32)
        # shape of X -> No of samples, dimensions(no. of features)       
        N, D = X.shape
        # take only unique samples from Y and get the number of unique samples
        K = len(set(Y))
        # create a list for storing weights of hidden layers
        self.hidden_layers = []
        # Matrix 1 will have no of rows = D
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:
            # Create a hidden layer matrix using size with number of units in previous as rows.
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            # change M1 to M2 for next layer
            M1 = M2
            count += 1
        
        # Weight layer for final layer(Not sure)
        W = init_weight(M1, K)
        b = np.zeros(K)
        self.W = theano.shared(W, 'W_logreg')
        self.b = theano.shared(b, 'b_logreg')
        
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params
        
        # initialize all the params to zero.
        dparams = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]
        
        
        thX = T.matrix('X')
        thY = T.ivector('Y')
        pY = self.forward(thX)
        
        rcost = reg*T.sum([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY]))
        prediction = self.predict(thX)
        grads = T.grad(cost, self.params)
        
        updates = [
                (p, p + mu*dp - learning_rate * g) for p, dp, g in zip(self.params, dparams, grads)
                ] + [
                        (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
                        ]
        train_op = theano.function(
                inputs=[thX, thY],
                outputs=[cost, prediction],
                updates = updates,
                )
        n_batches = N // batch_size
        
        costs = []
        
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                X_batch = X[j*batch_size: (j*batch_size + batch_size)]
                Y_batch = Y[j*batch_size: (j*batch_size + batch_size)]
                
                c, p = train_op(X_batch, Y_batch)
                
                if j % print_period == 0:
                    costs.append(c)
                    e = np.mean(Y_batch != p)
                    print(f'epoch: {i}; batch: {j} nb: {n_batches}; cost: {c}; error rate: {e}')
                    
        if show_fig:
            plt.plot(costs)
            plt.show()
            
    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.W) + self.b)
    
    def predict(self, X):
        py = self.forward(X)
        return T.argmax(py, axis=1)
    
def wide():
    X, Y = all_parity_pairs(12)
    model = ANN([2048])
    model.fit(X, Y, learning_rate=10e-5, print_period=10, epochs=300, show_fig=True)
    
def deep():
    X, Y = all_parity_pairs(12)
    model = ANN([1024, 1024])
    model.fit(X, Y, learning_rate=10e-4, print_period=10, epochs=100, show_fig=True)
    
        
    
wide()
deep()