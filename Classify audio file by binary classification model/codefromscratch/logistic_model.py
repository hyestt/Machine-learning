"""logistic model class for binary classification."""

import numpy as np

class LogisticModel(object):

    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of W is the bias term,
            self.W = [Bias, W1, W2, W3, ...]
            where Wi corresponds to each feature dimension

        W_init needs to support:
          'zeros': initialize self.W with all zeros.
          'ones': initialze self.W with all ones.
          'uniform': initialize self.W with uniform random number between [0,1)
          'gaussian': initialize self.W with gaussion distribution (0, 0.1)

        Args:
            ndims(int): feature dimension
            W_init(str): types of initialization.
        """
        self.ndims = ndims
        self.W_init = W_init
        self.W = None
        ###############################################################
        # Fill your code below
        ###############################################################
        if W_init == "zeros":
            self.W = np.zeros((ndims+1,1))
        elif W_init == "ones":
            self.W = np.ones((ndims+1,1))
        elif W_init == "uniform":
            self.W = np.random.sample((ndims+1,1))
        elif W_init == 'gaussian':
            self.W = np.random.normal(0,0.1,(ndims+1,1))
        else:
            print ('Unknown W_init ', W_init)

    def save_model(self, weight_file):
        """ Save well-trained weight into a binary file.
        Args:
            weight_file(str): binary file to save into.
        """
        self.W.astype('float32').tofile(weight_file)
        print('model saved to', weight_file)

    def load_model(self, weight_file):
        """ Load pretrained weghit from a binary file.
        Args:
            weight_file(str): binary file to load from.
        """
        self.W = np.fromfile(weight_file, dtype=np.float32)
        print('model loaded from', weight_file)

    def forward(self, X):
        """ Forward operation for logistic models.
            Performs the forward operation, and return probability score (sigmoid).
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): probability score of (label == +1) for each sample
                             with a dimension of (# of samples,)
        """
        ###############################################################

        f = 1.0 / (1.0 + np.exp(-X.dot(self.W)))
        ###############################################################
        return f

    def backward(self, Y_true, X):
        """ Backward operation for logistic models.
            Compute gradient according to the probability loss on lecture slides
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,)
        Returns:
            (numpy.ndarray): gradients of self.W
        """
        ###############################################################
        #L = loss function
        # L = np.sum(log(1+np.exp(-1*Y_true.dot(X.dot(self.W)))))
        #total_grad = gradient of loss function to self.W
        # print(X.shape)

        exp = np.exp(-np.multiply(Y_true, X.dot(self.W)))

        ###############################################################
        return np.sum(-np.divide((Y_true * exp) * X, (1 + exp)),axis = 0).reshape(self.ndims+1,1)

    def classify(self, X):
        """ Performs binary classification on input dataset.
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): predicted label = +1/-1 for each sample
                             with a dimension of (# of samples,)
        """
        ###############################################################
        classify_list = []
        print(self.forward(X)[0])
        for i in self.forward(X):
            if i > 0.5:
                classify_list.append(1)
            else:
                classify_list.append(-1)

        prediction = np.asarray(classify_list).reshape((len(classify_list),1))
        ###############################################################
        return prediction

    def fit(self, Y_true, X, learn_rate, max_iters):
        """ train model with input dataset using gradient descent.
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,)
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            learn_rate: learning rate for gradient descent
            max_iters: maximal number of iterations
            ......: append as many arguments as you want
        """
        ###############################################################
        i = 0
        while i<max_iters:
            self.W = self.W - learn_rate* self.backward(Y_true, X)
            i+=1

        ###############################################################
        pass

