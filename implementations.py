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
            w: numpy array of shape=(D, ) denoting the final model parameters
            loss: scalar denoting final loss value
        """
    w = initial_w
    N = len(y)
    for n_iter in range(max_iters):
        e = y - x@w
        grad = 1/N * x.T@e
        w = w - gamma * grad

    e = y - x@w
    loss = 1/2 * 1/N * np.sum(e**2)
    
    return w, loss


def mean_squared_error_sgd(y, x, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm with batch_size=1 using mean squared loss function.
    
        Args:
            y: numpy array of shape=(N, )
            x: numpy array of shape=(N,D)
            initial_w: numpy array of shape=(D, )
            max_iters: scalar denoting the total number of iterations
            gamma: scalar denoting the stepsize
    
        Returns:
            w: numpy array of shape=(D, ) denoting the final model parameters
            loss: scalar denoting final loss value
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
    
    return w, loss


def least_squares(y, tx):
    """Implementation of the least squares solution using normal equations.

    Args:
        y: numpy array of shape (N,), N is the number of samples
        tx: numpy array of shape (N,D), D is the number of features

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features
        mse: mean square error
    """
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    e = y - tx@w
    mse = np.sum(e**2) / (2 * len(y))

    return w, mse


def ridge_regression(y, tx, lambda_):
    """Implementation of ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: mean square error.
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    e = y - tx@w
    mse = np.sum(e**2) / (2 * len(y))

    return w, mse
