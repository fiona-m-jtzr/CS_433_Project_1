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

#Pipeline choice
launch_pure_log = True
launch_reg_log = True
launch_pure_log_sgd = True
launch_reg_log_sgd = True
launch_kNN = True


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
    #Optimization params
    max_iters = 500
    gamma = 0.01
    lambda_ = 0.5

    #Pure logistic regression
    if launch_pure_log == True :
        w_log, loss_log = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
        print("Pure logistic regression: OK \n")

        #Class choice
        logistic_mask_log = sigmoid(tx_te @ w_log) >= 0.5
        y_te_pred_log = np.zeros((np.shape(tx_te)[0],1))
        y_te_pred_log[logistic_mask_log] = 1

        #Confusion matrix display
        confusion_matrix_log, confusion_matrix_ratio_log = confusion_matrix_custom(y_te, y_te_pred_log)
        display_confusion_matrix(confusion_matrix_log, confusion_matrix_ratio_log, "pure_logistic")


    #Penalized logistic regression
    if launch_reg_log == True :
        w_log_reg, loss_log_reg = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, max_iters, gamma)
        print("Penalized logistic regression: OK \n")

        #Class choice
        logistic_mask_log_reg = sigmoid(tx_te @ w_log_reg) >= 0.5
        y_te_pred_log_reg = np.zeros((np.shape(tx_te)[0],1))
        y_te_pred_log_reg[logistic_mask_log_reg] = 1

        #Confusion matrix display
        confusion_matrix_log_reg, confusion_matrix_ratio_log_reg = confusion_matrix_custom(y_te, y_te_pred_log_reg)
        display_confusion_matrix(confusion_matrix_log_reg, confusion_matrix_ratio_log, "reg_logistic")



    #### LOGISTIC REGRESSION SGD ####

    #Pure logistic regression sgd
    if launch_pure_log_sgd == True :
        #Optimization params
        max_iters = 500
        gamma = 5000
        w_log_sgd, losses_train_log_sgd, losses_test_log_sgd = logistic_regression_sgd(y_tr, tx_tr, y_te, tx_te, initial_w, max_iters, gamma)
        print("Pure logistic regression SGD: OK \n")

        #Losses display
        display_losses(losses_train_log_sgd, losses_test_log_sgd, "logistic_SGD")

        #Class choice
        logistic_mask_log_sgd = sigmoid(tx_te @ w_log_sgd) >= 0.5
        y_te_pred_log_sgd = np.zeros((np.shape(tx_te)[0],1))
        y_te_pred_log_sgd[logistic_mask_log_sgd] = 1

        #Display
        confusion_matrix_log_sgd, confusion_matrix_ratio_log_sgd = confusion_matrix_custom(y_te, y_te_pred_log_sgd)
        display_confusion_matrix(confusion_matrix_log_sgd, confusion_matrix_ratio_log, "pure_logistic_SGD")

    #Penalized logistic regression sgd
    if launch_reg_log_sgd == True :
        #Optimization params
        max_iters = 500
        gamma = 5000
        lambda_ = 0.0000005
        w_log_reg_sgd, losses_train_log_reg_sgd, losses_test_log_reg_sgd = reg_logistic_regression_sgd(y_tr, tx_tr, y_te, tx_te, lambda_, initial_w, max_iters, gamma)
        print("Penalized logistic regression SGD: OK \n")

        #Losses display
        display_losses(losses_train_log_reg_sgd, losses_test_log_reg_sgd, "logistic_regularized_SGD")

        #Class choice
        logistic_mask_log_reg_sgd = sigmoid(tx_te @ w_log_reg_sgd) >= 0.5
        y_te_pred_log_reg_sgd = np.zeros((np.shape(tx_te)[0],1))
        y_te_pred_log_reg_sgd[logistic_mask_log_reg_sgd] = 1

        #Confusion matrix display
        confusion_matrix_log_reg_sgd, confusion_matrix_ratio_log_reg_sgd = confusion_matrix_custom(y_te, y_te_pred_log_reg_sgd)
        display_confusion_matrix(confusion_matrix_log_reg_sgd, confusion_matrix_ratio_log, "reg_logistic_SGD")


    #### kNN ####
    if launch_kNN == True :
        #Fit
        neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        neigh.fit(x_tr, y_tr.ravel())

        #Predict
        y_te_pred_kNN = neigh.predict(x_te)
        print("kNN: OK \n")

        #Confusion matrix display
        confusion_matrix_kNN, confusion_matrix_ratio_kNN = confusion_matrix_custom(y_te, y_te_pred_kNN)
        display_confusion_matrix(confusion_matrix_kNN, confusion_matrix_ratio_log, "kNN")