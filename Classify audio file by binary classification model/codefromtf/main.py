"""Main function for binary classifier"""
import tensorflow as tf
import numpy as np
from io_tools import *
from logistic_model import *


""" Hyperparameter for Training """
learn_rate = None
max_iters = None

def main(_):

    # Load dataset.
    A,T = read_dataset_tf('../trainset','indexing.txt')
    # Initialize model.
    ndims = A.shape[1] - 1
    model = LogisticModel_TF(ndims, W_init='zeros')
    # Build TensorFlow training graph
    model.build_graph(learn_rate = 0.005)
    # Train model via gradient descent.
    prediction = model.fit(T,A,max_iters=100)
    for i in range(len(prediction)):
        if prediction[i] >= 0.5:
            prediction[i] = 1
        else:
            prediction[i] = 0
    accurate = np.mean(prediction == True)

    print(accurate,"rate")

    # Compute classification accuracy based on the return of the "fit" method




if __name__ == '__main__':
    tf.app.run()
