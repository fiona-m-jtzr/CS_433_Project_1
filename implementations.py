'''
File name: implementations.py
Authors: Mayeul, Sondre, Fiona
Python version: 3.13.9  
'''

import numpy as np

def mean_squared_error_gd(y, x, initial_w, max_iters, gamma):
    """The Gradient Descent algorithm using mean squared loss function.
    
        Args:
            y: numpy array of shape=(N, )
            x: numpy array of shape=(N,D)
            initial_w: numpy array of shape=(D, )
            max_iters: scalar denoting the total number of iterations
            gamma: scalar denoting the stepsize
    
        Returns:
            loss: scalar denoting final loss value
            w: numpy array of shape=(D, ) denoting the final model parameters
        """
    w = initial_w
    N = len(y)
    for n_iter in range(max_iters):
        e = y - x@w
        grad = 1/N * x.T@e
        w = w - gamma * grad

    e = y - x@w
    loss = 1/2 * 1/N * np.sum(e**2)
    
    return loss, w


def mean_squared_error_sgd(y, x, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm with batch_size=1 using mean squared loss function.
    
        Args:
            y: numpy array of shape=(N, )
            x: numpy array of shape=(N,D)
            initial_w: numpy array of shape=(D, )
            max_iters: scalar denoting the total number of iterations
            gamma: scalar denoting the stepsize
    
        Returns:
            loss: scalar denoting final loss value
            w: numpy array of shape=(D, ) denoting the final model parameters
        """
    w = initial_w
    N = len(y)
    for n_iter in range(max_iters):
        idx = np.random.randint(N)
        e = y[idx] - x[idx]@w
        grad = x[idx].T@e
        w = w - gamma * grad

    e = y - x@w
    loss = 1/2 * 1/N * np.sum(e**2)
    
    return loss, w

