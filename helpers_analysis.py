####################### CONTENT #######################

#This file contains useful functions, only for the analysis pipeline



####################### IMPORTS #######################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import sklearn
import time
import random
from datetime import datetime

from helpers import *
from helpers_analysis import * 
from implementations import * 



####################### DISPLAY UTILS #######################

def display_losses(losses_train, losses_test, title) :
    #Loss display: pure logistic
    plt.plot(losses_train, label="Train")
    plt.plot(losses_test, label="Test")
    plt.ylabel("Loss value")
    plt.xlabel("Iteration")
    plt.title(f"Loss value during optimization, {title}")
    plt.legend(loc="upper right")
    plt.savefig(f"image/losses_{title}.png", dpi=300)
    plt.show()


def confusion_matrix_custom(y_te, y_te_pred) :
    """Computes the confusion matrix (cm_custom) between true labels (y_te) and predicted labels (y_te_pred)"""
    cm_custom = np.zeros((2,2))

    for i,elt in enumerate(y_te_pred) :
        #Class 0 which is true (True negative)
        if elt == 0 and elt == y_te[i] :
            cm_custom[1,1] += 1

        #Class 0 which is false (under-estimated ==> death) (false negative)
        elif elt == 0 and elt != y_te[i] :
            cm_custom[0, 1] += 1

        #Class 1 which is false (over-prediction) (false positive)
        elif elt == 1 and elt != y_te[i] :
            cm_custom[1, 0] += 1
            
        #Class 1 which is true (True negative)
        elif elt == 1 and elt == y_te[i] :
            cm_custom[0, 0] += 1

    cm_ratio_custom = cm_custom/len(y_te)

    return(cm_custom, cm_ratio_custom)


def display_confusion_matrix(cm, cm_ratio, title) :
    """Display confusion matrix only with matplotlib. Convention: positive when class 1 (ie heart disease)"""
    
    L_labels = [["True Positive", "False Negative"], ["False Positive", "True Negative"]]

    #Figure creation
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    #Heatmap
    im = ax[0].imshow(cm, cmap='Blues', origin='upper')
    fig.colorbar(im, ax=ax[0])

    #Plot values and type
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax[0].text(
                j, i, f'{cm[i, j]:.0f} \n {L_labels[i][j]}',
                ha='center',
                va='center',
                color='black'
            )

    ax[0].set_title(f"confusion_matrix_{title}")

    #Heatmap ratio
    im = ax[1].imshow(cm_ratio, cmap='Blues', origin='upper')
    fig.colorbar(im, ax=ax[1])

    #Plot values and type
    for i in range(cm_ratio.shape[1]):
        for j in range(cm_ratio.shape[1]):
            ax[1].text(
                j, i, f'{cm_ratio[i, j]:.3f} \n {L_labels[i][j]}',
                ha='center',
                va='center',
                color='black'
            )

    ax[1].set_title(f"confusion_matrix_ratio_{title}")

    #Metrics computations and show in title
    TP = cm[0,0]
    FN = cm[0,1]
    FP = cm[1,0]
    TN = cm[1,1]
    F1_score = (2 * TP)/(2 * TP + FP + FN)
    accuracy_score = (TP + TN)/(TP + FN + FP + TN)

    fig.suptitle(f"{title}: F1 = {F1_score:.2f}, accuracy = {accuracy_score:.2f}")

    fig.savefig(f"image/confusion_matrix_{title}.png", dpi=300)
    plt.show()



####################### LOGISTIC REGRESSION UTILS #######################

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return(1/(1+np.exp(-t)))



####################### LOGISTIC REGRESSION SGD #######################

def logistic_regression_sgd(y_tr, tx_tr, y_te, tx_te, initial_w, max_iters, gamma):
    """Implementation of logistic regression.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w:  shape=(D, 1)
        max_iters: scalar denoting the total number of iterations
        gamma: scalar denoting the stepsize


    Returns:
        w: numpy array of shape=(D, ) denoting the final model parameters
        loss: scalar denoting final loss value
    """

    w = initial_w
    N = len(y_tr)
    losses_train = []
    losses_test = []
    for n_iter in range(max_iters):

        #Index random choice and update
        idx = np.random.randint(N)
        grad = tx_tr[idx].T*(1/(1+np.exp(-tx_tr[idx]*w)) - y_tr[idx])/N
        w = w - gamma * grad

        #Loss computations
        loss_train = np.mean(-y_tr*(tx_tr@w) + np.log(1 + np.exp(tx_tr@w)))
        losses_train.append(loss_train)
        loss_test = np.mean(-y_te*(tx_te@w) + np.log(1 + np.exp(tx_te@w)))
        losses_test.append(loss_test)
    
    return w, losses_train, losses_test


def reg_logistic_regression_sgd(y_tr, tx_tr, y_te, tx_te, lambda_, initial_w, max_iters, gamma):
    """Implementation of logistic regression.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w:  shape=(D, 1)
        max_iters: scalar denoting the total number of iterations
        gamma: scalar denoting the stepsize


    Returns:
        w: numpy array of shape=(D, ) denoting the final model parameters
        loss: scalar denoting final loss value
    """

    w = initial_w
    N = len(y_tr)
    losses_train = []
    losses_test = []
    for n_iter in range(max_iters):
        idx = np.random.randint(N)
        grad = tx_tr[idx].T*(1/(1+np.exp(-tx_tr[idx]*w)) - y_tr[idx])/N + 2 * lambda_ * w 
        w = w - gamma * grad

        #Loss computations
        loss_train = np.mean(-y_tr*(tx_tr@w) + np.log(1 + np.exp(tx_tr@w)))
        losses_train.append(loss_train)
        loss_test = np.mean(-y_te*(tx_te@w) + np.log(1 + np.exp(tx_te@w)))
        losses_test.append(loss_test)

    loss = np.mean(-y_tr*(tx_tr@w) + np.log(1 + np.exp(tx_tr@w)))
    
    return w, losses_train, losses_test



####################### kNN UTILS #######################

def kNN_custom(x_tr, y_tr, x_te, N) :
    n = len(x_te)
    y_te_pred = np.zeros((n, 1))
    for i in range(n) :
        x_te_case = x_te[i, :]

        #Distances
        distances = np.linalg.norm(x_tr - x_te_case, axis=1)
        
        #N indices of min distance
        indices = np.argpartition(distances, N - 1)[:N]

        #Counting of classes corresponding to above indices
        values, counts = np.unique(y_tr[indices], return_counts=True)

        #Class prediction
        class_pred = values[np.argmax(counts)]  
        y_te_pred[i,0] = class_pred
    
    return(y_te_pred)

def kNN_custom_vectorized(x_tr, y_tr, x_te, N, batch_size=100):
    x_tr = np.asarray(x_tr)
    x_te = np.asarray(x_te)
    y_tr = np.asarray(y_tr).ravel()

    n_test = len(x_te)
    y_te_pred = np.empty((n_test, 1), dtype=y_tr.dtype)

    # Process test samples in batches
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        x_batch = x_te[start:end]

        # Compute squared Euclidean distances without a 3D array
        distances = (
            np.sum(x_batch**2, axis=1)[:, None]
            + np.sum(x_tr**2, axis=1)[None, :]
            - 2 * x_batch @ x_tr.T
        )

        # Select N nearest neighbors
        indices = np.argpartition(
            distances, N - 1, axis=1
        )[:, :N]

        # Majority vote
        for i, idx in enumerate(indices):
            values, counts = np.unique(y_tr[idx], return_counts=True)
            y_te_pred[start + i, 0] = values[np.argmax(counts)]

    return y_te_pred



####################### SVM utils #######################

def calculate_primal_objective(y, X, w, lambda_):
    """compute the full cost (the primal objective, equation (1) in the exercise pdf),
        that is loss plus regularizer.

    Args:
        X: the full dataset matrix, shape = (num_examples, num_features)
        y: the corresponding +1 or -1 labels, shape = (num_examples)
        w: shape = (num_features)

    Returns:
        scalar, non-negative

    """
    N = len(y)
    cost = lambda_/2 * np.linalg.norm(w)**2
    for i in range(N) :
        cost += 1/N * np.maximum(0, 1 - y[i] * X[i,:].T @ w)
    return(cost)

def calculate_accuracy(y, X, w):
    """compute the accuracy on the given dataset (X, y) using the model w.

    Args:
        X: the full dataset matrix, shape = (num_examples, num_features)
        y: the corresponding +1 or -1 labels, shape = (num_examples)
        w: shape = (num_features)

    Returns:
        scalar, between 0 and 1
    """
    y_pred = np.sign(X @ w)
    TP_TN = np.sum(y + y_pred)/2
    accuracy = (TP_TN)/len(y)
    return(accuracy)

def calculate_stochastic_gradient(y, X, w, lambda_, n, num_examples):
    """compute the stochastic gradient of loss plus regularizer.

    Args:
        X: the dataset matrix, shape = (num_examples, num_features)
        y: the corresponding +1 or -1 labels, shape = (num_examples)
        w: shape = (num_features)
        lambda_: positive scalar number
        n: the index of the (one) datapoint we have sampled
        num_examples: N

    Returns:
        numpy array, shape = (num_features)
    """

    N = num_examples
    margin = y[n] * (X[n, :].T @ w)
    gradient = lambda_ * w - y[n] * X[n, :] * (margin < 1)
    return(gradient)

def sgd_for_svm_demo(y, X, max_iter, gamma, lambda_):
    xs = np.unique(np.round(np.logspace(0, 5, 201)))[:-1]

    num_examples, num_features = X.shape
    w = np.zeros(num_features)
    costs = []
    start = time.time()

    for it in range(max_iter):
        # n = sample one data point uniformly at random data from x
        n = random.randint(0, num_examples - 1)

        grad = calculate_stochastic_gradient(y, X, w, lambda_, n, num_examples)
        w -= gamma / (it + 1) * grad

        cost = calculate_primal_objective(y, X, w, lambda_)
        costs.append(cost)

    end = time.time()

    print(
        "Final training accuracy = {l} %".format(
            l=np.round(100 * calculate_accuracy(y, X, w), 2)
        )
    )
    print("Training time: " + str(np.round(end - start, 1)) + " seconds ")

    return(w, costs)