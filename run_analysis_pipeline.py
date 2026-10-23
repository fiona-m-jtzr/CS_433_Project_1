####################### CONTENT #######################

#This file does the full analysis pipeline: 
#a-Process data according to Fiona's and Sondre's codes
#b-Train classifier with Mayeul's code
#c-Display results



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
print("Imports: OK\n")



####################### PARAMS TO CHANGE #######################

#Data directory
DATA_DIR = ""

#Preprocessing
imputation = False

#Optimization params
max_iters = 500
gamma = 0.01
lambda_ = 0.5


####################### ANALYSIS PIPELINE #######################


if __name__ == "__main__" :

    print("BEGIN\n")

    #### LOAD AND PREPROCESS ####

    #Train
    x_train, y_train, cont_params, cat_params, mask = load_data_train(DATA_DIR, imputation=imputation)
    print("Load and preprocess train: OK \n")

    #Test
    x_test = load_data_test(DATA_DIR, cont_params, cat_params, mask, imputation=imputation)
    print("Load and preprocess test: OK \n")



    #### PREPROCESS FOR CLASSIFIERS ####

    #Dtype (to change in case of being unable to allocate matrices with large type. Can be changed to np.float32)
    dtype_array = np.float32

    #Train/validation split
    #Mask definition
    rng = np.random.default_rng()
    n = len(y_train)
    ratio_mask = 33
    mask = rng.random(n) >= ratio_mask / 100

    #Train/validation split
    x_tr = x_train[mask]
    y_tr = y_train[mask]
    x_te = x_train[~mask]
    y_te = y_train[~mask]

    #Add intercept
    tx_tr = np.c_[np.ones((x_tr.shape[0], 1)), x_tr]
    tx_tr = tx_tr.astype(dtype_array)
    tx_te = np.c_[np.ones((x_te.shape[0], 1)), x_te]
    tx_te = tx_te.astype(dtype_array)

    # Initialize weights
    initial_w = np.zeros(tx_tr.shape[1])



    #### LOGISTIC REGRESSION ####

    #Pure logistic regression
    w_log, loss_log = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
    print("Pure logistic regression: OK \n")

    #Penalized logistic regression
    w_log_reg, loss_log_reg = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, max_iters, gamma)
    print("Penalized logistic regression: OK \n")


    #Results
    #Pure logistic
    #Class choice
    logistic_mask_log = sigmoid(tx_te @ w_log) >= 0.5
    y_te_pred_log = np.zeros((np.shape(tx_te)[0],1))
    y_te_pred_log[logistic_mask_log] = 1

    #Display
    plot_heatmap(y_te, y_te_pred_log, "pure_logistic")

    #Reg logistic
    #Class choice
    logistic_mask_log_reg = sigmoid(tx_te @ w_log_reg) >= 0.5
    y_te_pred_log_reg = np.zeros((np.shape(tx_te)[0],1))
    y_te_pred_log_reg[logistic_mask_log_reg] = 1

    #Display
    plot_heatmap(y_te, y_te_pred_log_reg, "reg_logistic")



    #### kNN ####

    #Fit
    neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
    neigh.fit(x_tr, y_tr.ravel())

    #Predict
    y_te_pred_kNN = neigh.predict(x_te)
    print("kNN: OK \n")

    #Results
    plot_heatmap(y_te, y_te_pred_kNN, "kNN")