#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:54:14 2019

@author: myidispg
"""

import numpy as np
import theano
import theano.tensor as T

x = T.vector('x')

def square(x):
    return x*x

outputs, updates = theano.scan(fn=square, sequences=x, n_steps=x.shape[0])

square_op = theano.function(
        inputs=[x], outputs=[outputs])

o_val = square_op(np.array([1,2,3,4,5]))

print('output: {}'.format(o_val))


import numpy as np
import theano
import theano.tensor as T

N = T.iscalar('N')

def recurrence(n, fn1, fn2):
    return fn1+ fn2, fn1

outputs, updates = theano.scan(fn=recurrence, sequences=T.arange(N), n_steps=N, outputs_info=[1, 1])

fibonacci = theano.function(inputs=[N], outputs=outputs)

o_val = fibonacci(8)

print('output: {}'.format(o_val))

#------SIMULATE A LOW PASS FILTER------------
import numpy as np
import theano 
import theano.tensor as T
import matplotlib.pyplot as plt

# -------GENERATE A RANDOM SIN SIGNAL--------------
X = 2*np.random.randn(300) + np.sin(np.linspace(0, 3*np.pi, 300))

plt.plot(X)
plt.title('original')
plt.show()

decay = T.scalar('decay')
sequence = T.vector('sequence')

def recurrence(x, last, decay):
    return (1-decay)*x + decay*last

outputs, _ = theano.scan(
        fn=recurrence,
        sequences=sequence,
        n_steps=sequence.shape[0],
        outputs_info=[np.float64(0)],
        non_sequences=decay
        )
lpf = theano.function(inputs=[sequence, decay], outputs=outputs)

Y = lpf(X, 0.99)

plt.plot(Y)
plt.title('filtered')
plt.show()