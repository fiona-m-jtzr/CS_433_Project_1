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
import time
import random
from datetime import datetime

from helpers import *
from helpers_analysis import * 
from implementations import *  
print("Imports: OK\n")



####################### PARAMS TO CHANGE #######################

#Data directory
DATA_DIR = ""

#Preprocessing
imputation = False
downsample = True

#Pipeline choice
launch_pure_log = True
launch_reg_log = True
launch_pure_log_sgd = True
launch_reg_log_sgd = True
launch_kNN = True
launch_SVM = True


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

    #Downsampling
    if downsample == True :
        x_train, y_train = downsample_classes(x_train, y_train, random_state=0)
        print("Downsample: OK \n")

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
        gamma = 2500
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
        display_confusion_matrix(confusion_matrix_log_sgd, confusion_matrix_ratio_log_sgd, "pure_logistic_SGD")

    #Penalized logistic regression sgd
    if launch_reg_log_sgd == True :
        #Optimization params
        max_iters = 500
        gamma = 5000
        lambda_ = 0.000005
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
        display_confusion_matrix(confusion_matrix_log_reg_sgd, confusion_matrix_ratio_log_reg_sgd, "reg_logistic_SGD")


    #### kNN ####
    if launch_kNN == True :
        #Prediction
        y_te_pred_kNN = kNN_custom_vectorized(x_tr, y_tr, x_te, 1)
        print("kNN: OK \n")

        #Confusion matrix
        confusion_matrix_kNN, confusion_matrix_ratio_kNN = confusion_matrix_custom(y_te, y_te_pred_kNN)
        display_confusion_matrix(confusion_matrix_kNN, confusion_matrix_ratio_kNN, "kNN")


    #### SVM ####
    if launch_SVM == True :
        #SVM params
        max_iter = 100
        gamma = 0.1
        lambda_ = 0.01

        #SVM training
        w_SVM, costs = sgd_for_svm_demo(y_tr, x_tr, max_iter, gamma, lambda_)
        print("SVM: OK \n")

        #Display cost
        plt.figure()
        plt.title("SVM costs")
        plt.plot(costs)
        plt.xlabel("Number of iterations")
        plt.ylabel("Primal objective")
        plt.yscale("log")
        plt.savefig(f"image/losses_SVM.png", dpi=300)
        plt.show()

        #Prediction
        y_te_pred_SVM = np.sign(x_te @ w_SVM)

        #Confusion matrix
        confusion_matrix_SVM, confusion_matrix_ratio_SVM = confusion_matrix_custom(y_te, y_te_pred_SVM)
        display_confusion_matrix(confusion_matrix_SVM, confusion_matrix_ratio_SVM, "SVM")




####################### SUBMISSION PIPELINE #######################

#Reference x_test for indices
x_test_ref = np.genfromtxt(
    "x_test.csv",
    delimiter=",",
    skip_header=1
)
indices_array = x_test_ref[:,0].reshape(-1, 1)
print("Reference: OK\n")

#Intercept
tx_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]
tx_test = tx_test.astype(dtype_array)

#Pure logistic regression
if launch_pure_log == True :

    #Class choice
    logistic_mask_log = sigmoid(tx_test @ w_log) >= 0.5
    y_test_pred_log = - np.ones((np.shape(tx_test)[0],1))
    y_test_pred_log[logistic_mask_log] = 1

    #Prediction with indices
    y_submission_pred_log = np.hstack([indices_array, y_test_pred_log])

    #Save prediction
    np.savetxt(
        "predictions/y_test_pred_log.csv",
        y_submission_pred_log,
        delimiter=",",
        header="Id, Prediction",
        comments="",
        fmt="%d"
    )
    print("Submission pure log: OK\n")

#Penalized logistic regression
if launch_reg_log == True :

    #Class choice
    logistic_mask_log_reg = sigmoid(tx_test @ w_log_reg) >= 0.5
    y_test_pred_log_reg = - np.ones((np.shape(tx_test)[0],1))
    y_test_pred_log_reg[logistic_mask_log_reg] = 1

    #Prediction with indices
    y_submission_pred_log_reg = np.hstack([indices_array, y_test_pred_log_reg])

    #Save prediction
    np.savetxt(
        "predictions/y_test_pred_log_reg.csv",
        y_submission_pred_log_reg,
        delimiter=",",
        header="Id, Prediction",
        comments="",
        fmt="%d"
    )
    print("Submission log reg: OK\n")

if launch_pure_log_sgd == True :

    #Class choice
    logistic_mask_log_sgd = sigmoid(tx_test @ w_log_sgd) >= 0.5
    y_test_pred_log_sgd = - np.ones((np.shape(tx_test)[0],1))
    y_test_pred_log_sgd[logistic_mask_log_sgd] = 1

    #Prediction with indices
    y_submission_pred_log_sgd = np.hstack([indices_array, y_test_pred_log_sgd])

    #Save prediction
    np.savetxt(
        "predictions/y_test_pred_log_sgd.csv",
        y_submission_pred_log_sgd,
        delimiter=",",
        header="Id, Prediction",
        comments="",
        fmt="%d"
    )
    print("Submission pure log sgd: OK\n")

#Penalized logistic regression sgd
if launch_reg_log_sgd == True :

    #Class choice
    logistic_mask_log_reg_sgd = sigmoid(tx_test @ w_log_reg_sgd) >= 0.5
    y_test_pred_log_reg_sgd = - np.ones((np.shape(tx_test)[0],1))
    y_test_pred_log_reg_sgd[logistic_mask_log_reg_sgd] = 1

    #Prediction with indices
    y_submission_pred_log_reg_sgd = np.hstack([indices_array, y_test_pred_log_reg_sgd])

    #Save prediction
    np.savetxt(
        "predictions/y_test_pred_log_reg_sgd.csv",
        y_submission_pred_log_reg_sgd,
        delimiter=",",
        header="Id, Prediction",
        comments="",
        fmt="%d"
    )
    print("Submission log reg sgd: OK\n")

#SVM
if launch_SVM == True :
    #Prediction
    y_test_pred_SVM = np.sign(x_test @ w_SVM).reshape(-1,1)*2 - 1

    #Prediction with indices
    y_submission_pred_SVM = np.hstack([indices_array, y_test_pred_SVM])

    #Save prediction
    np.savetxt(
        "predictions/y_test_pred_SVM.csv",
        y_submission_pred_SVM,
        delimiter=",",
        header="Id, Prediction",
        comments="",
        fmt="%d"
    )
    print("Submission SVM: OK\n")

#kNN
if launch_kNN == True :
    #Prediction
    y_test_pred_kNN = kNN_custom_vectorized(x_tr, y_tr, x_test, 1)*2 - 1

    #Prediction with indices
    y_submission_pred_kNN = np.hstack([indices_array, y_test_pred_kNN])

    #Save prediction
    np.savetxt(
        "predictions/y_test_pred_kNN.csv",
        y_submission_pred_kNN,
        delimiter=",",
        header="Id, Prediction",
        comments="",
        fmt="%d"
    )
    print("Submission kNN: OK\n")