"""Main function for binary classifier"""

import numpy as np

from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = None
max_iters = None

if __name__ == '__main__':


    # Load dataset.
    #  A, T = read_dataset('../data/trainset','indexing.txt')
    A,T = read_dataset('../trainset','indexing.txt')

    # Initialize model.
    ndims = A.shape[1] - 1
    model = LogisticModel(ndims, W_init='ones')
    # Trainprint model via gradient descent.
    # f= model.forward(A)
    # total_grad = model.backward(T,A)
    # prediction = model.classify(A)
    model.fit(T, A, learn_rate = 0.001, max_iters = 100)
    # Save trained model to 'trained_weights.np'
    # model.save_model("trained_weights.np")
    # Load trained model from 'trained_weights.np'
    # model.load_model("trained_weights.np")
    # Try all other methods: forward, backward, classify, compute accuracy
    prediction = model.classify(A)
    print("Accurate rate is ",np.mean(prediction == T),"percent")

    pass
