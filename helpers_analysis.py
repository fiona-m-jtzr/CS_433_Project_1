####################### CONTENT #######################

#This file contains useful functions, only for the analysis pipeline



####################### IMPORTS #######################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import sklearn

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
    
    L_labels = [["True Positive", "False Negative"], ["False Positive", "False positive"]]

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
    im = ax[1].imshow(cm, cmap='Blues', origin='upper')
    fig.colorbar(im, ax=ax[1])

    #Plot values and type
    for i in range(cm.shape[1]):
        for j in range(cm.shape[1]):
            ax[1].text(
                j, i, f'{cm[i, j]:.0f} \n {L_labels[i][j]}',
                ha='center',
                va='center',
                color='black'
            )

    ax[1].set_title(f"confusion_matrix_ratio_{title}")

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