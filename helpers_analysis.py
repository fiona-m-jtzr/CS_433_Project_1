####################### CONTENT #######################

#This file contains useful functions, only for the analysis pipeline



####################### IMPORTS #######################

import numpy as np
import matplotlib.pyplot as plt
import time

from helpers_preprocessing import *
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
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels([1, -1])
    ax[0].set_xlabel("Prediction")
    ax[0].set_yticks([0, 1])
    ax[0].set_yticklabels([1, -1])
    ax[0].set_ylabel("Truth")

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
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels([1, -1])
    ax[1].set_xlabel("Prediction")
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels([1, -1])
    ax[1].set_ylabel("Truth")

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

def logistic_regression_sgd(y_tr, tx_tr, y_te, tx_te, initial_w, max_iters, gamma, rng, log_every=1000):
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
    losses_train, losses_test = [], []

    for n_iter in range(max_iters):
        idx = rng.integers(N)
        xi = tx_tr[idx]
        grad = xi * (1 / (1 + np.exp(-(xi @ w))) - y_tr[idx])
        w = w - gamma * grad

        if n_iter % log_every == 0:
            z_tr, z_te = tx_tr @ w, tx_te @ w
            losses_train.append(np.mean(np.logaddexp(0, z_tr) - y_tr * z_tr))
            losses_test.append(np.mean(np.logaddexp(0, z_te) - y_te * z_te))

    return w, losses_train, losses_test


def reg_logistic_regression_sgd(y_tr, tx_tr, y_te, tx_te, lambda_, initial_w, max_iters, gamma, rng, log_every=1000):
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
    losses_train, losses_test = [], []

    for n_iter in range(max_iters):
        idx = rng.integers(N)
        xi = tx_tr[idx]
        grad = xi * (1 / (1 + np.exp(-(xi @ w))) - y_tr[idx]) + 2 * lambda_ * w
        w = w - gamma * grad

        if n_iter % log_every == 0:
            z_tr, z_te = tx_tr @ w, tx_te @ w
            losses_train.append(np.mean(np.logaddexp(0, z_tr) - y_tr * z_tr))
            losses_test.append(np.mean(np.logaddexp(0, z_te) - y_te * z_te))

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
    hinge_loss = np.clip(1 - y*(X@w), 0, np.inf)
    loss = np.mean(hinge_loss) + lambda_/2 * np.sum(w**2)
    
    return loss

def calculate_accuracy(y, X, w):
    """compute the accuracy on the given dataset (X, y) using the model w.

    Args:
        X: the full dataset matrix, shape = (num_examples, num_features)
        y: the corresponding +1 or -1 labels, shape = (num_examples)
        w: shape = (num_features)

    Returns:
        scalar, between 0 and 1
    """
    prediction = ((X @ w > 0) * 2) - 1 
    accuracy = np.mean(prediction == y)

    return(accuracy)

def calculate_stochastic_gradient(y, X, w, lambda_, n):
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

    margin = y[n] * (X[n, :].T @ w)
    gradient = lambda_ * w - y[n] * X[n, :] * (margin < 1)
    return(gradient)



def sgd_for_svm_demo(y, X, max_iter, gamma, lambda_, rng):
    xs = np.unique(np.round(np.logspace(0, np.log10(max_iter), 201))).astype(int)
    log_iters = set(xs.tolist())

    num_examples, num_features = X.shape
    w = np.zeros(num_features)
    costs, logged = [], []
    start = time.time()

    for it in range(max_iter):
        n = rng.integers(num_examples)
        grad = calculate_stochastic_gradient(y, X, w, lambda_, n)
        w -= gamma / (it + 1) * grad

        if it in log_iters:
            costs.append(calculate_primal_objective(y, X, w, lambda_))
            logged.append(it)

    print("Final training accuracy = {} %".format(
        np.round(100 * calculate_accuracy(y, X, w), 2)))
    print("Training time: " + str(np.round(time.time() - start, 1)) + " seconds")

    return w, costs, np.array(logged)

def plot_SVM(iters, costs, title="SVM", save=True):
    plt.figure()
    plt.title(f"{title}: SGD on primal objective")
    plt.loglog(iters, costs)
    plt.xlabel("Number of iterations")
    plt.ylabel("Primal objective")
    if save:
        plt.savefig("image/losses_SVM.png", dpi=300)   # before show()
    plt.show()


####################### Threshold logistic regression #######################

def threshold_tuning(y, probs, grid=np.linspace(0.02, 0.6, 117)):
    best_t, best_f1 = 0.5, -1
    for t in grid:
        pred = probs >= t
        tp = np.sum((pred == 1) & (pred == y))
        fp = np.sum((pred == 1) & (pred != y))
        fn = np.sum((pred == 0) & (pred != y))
        f1 = 2*tp / (2*tp + fp + fn) if tp > 0 else 0
        if f1 > best_f1: best_t, best_f1 = t, f1
    return best_t, best_f1


                     