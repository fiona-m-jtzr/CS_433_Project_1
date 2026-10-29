####################### CONTENT #######################

#This file does the full analysis pipeline: 
#a-Process data according to Fiona's and Sondre's codes
#b-Train classifier with Mayeul's code
#c-Display results



####################### IMPORTS #######################

import numpy as np
import matplotlib.pyplot as plt
import os

from helpers_preprocessing import *
from helpers_analysis import * 
from implementations import *  
print("Imports: OK\n")



####################### PARAMS TO CHANGE #######################

#Data directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")

SEED = 0
ss = np.random.SeedSequence(SEED)
rng_split, rng_down, rng_sgd1, rng_sgd2, rng_svm = [np.random.default_rng(s) for s in ss.spawn(5)]

#Preprocessing
imputation_threshold = 0.2
downsample = True

#Pipeline choice
launch_pure_log = False
launch_reg_log = False
launch_pure_log_sgd = True
launch_reg_log_sgd = True
launch_kNN = True
launch_SVM = True


####################### ANALYSIS PIPELINE #######################


if __name__ == "__main__" :

    print("BEGIN\n")

    #### LOAD AND PREPROCESS ####

    #Train
    x_train, y_train, cont_params, cat_params, mask_nan, missing_codes, zero_codes = load_data_train(DATA_DIR, imputation_threshold)
    print("Load and preprocess train: OK \n")

    #Test
    x_test = load_data_test(DATA_DIR, cont_params, cat_params, mask_nan, missing_codes, zero_codes)
    print("Load and preprocess test: OK \n")



    #### PREPROCESS FOR CLASSIFIERS ####

    #Dtype (to change in case of being unable to allocate matrices with large type. Can be changed to np.float32)
    dtype_array = np.float32

    #Train/validation split
    #Mask definition
    n = len(y_train)
    ratio_mask = 20
    mask = rng_split.random(n) >= ratio_mask / 100

    #Train/validation split
    x_tr = x_train[mask]
    y_tr = y_train[mask]
    x_te = x_train[~mask]
    y_te = y_train[~mask]

    #Downsampling only training set
    if downsample == True :
        x_tr, y_tr = downsample_classes(x_tr, y_tr, rng_down)
        print("Downsample: OK \n")

    #Add intercept
    tx_tr = np.c_[np.ones((x_tr.shape[0], 1)), x_tr]
    tx_tr = tx_tr.astype(dtype_array)
    tx_te = np.c_[np.ones((x_te.shape[0], 1)), x_te]
    tx_te = tx_te.astype(dtype_array)

    # Initialize weights
    initial_w = np.zeros(tx_tr.shape[1])



    #### LOGISTIC REGRESSION ####

    #Optimization params
    max_iters = 1000
    gamma = 0.001
    lambda_ = 1e-2
    print("Parameters for logistic regression:\n")
    print(f"max_iters = {max_iters} \n")
    print(f"gamma = {gamma} \n")
    print(f"lambda = {lambda_} \n")

    #Pure logistic regression
    if launch_pure_log == True :
        w_log, loss_log = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
        print("Pure logistic regression: OK \n")

        #Tune threshold
        probs_val_log = sigmoid(tx_te @ w_log).ravel()
        thresh_log, f1_val = threshold_tuning(y_te, probs_val_log)
        print(f"Pure log: best threshold = {thresh_log:.3f}, validation F1 = {f1_val:.3f}")
        logistic_mask_log = sigmoid(tx_te @ w_log) >= thresh_log
        
        y_te_pred_log = np.zeros((np.shape(tx_te)[0],1))
        y_te_pred_log[logistic_mask_log] = 1

        #Confusion matrix display
        confusion_matrix_log, confusion_matrix_ratio_log = confusion_matrix_custom(y_te, y_te_pred_log)
        display_confusion_matrix(confusion_matrix_log, confusion_matrix_ratio_log, "pure_logistic")


    #Penalized logistic regression
    if launch_reg_log == True :
        w_log_reg, loss_log_reg = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, max_iters, gamma)
        print("Penalized logistic regression: OK \n")

        #Tune threshold
        probs_val_log_reg = sigmoid(tx_te @ w_log_reg).ravel()
        thresh_log_reg, f1_val = threshold_tuning(y_te, probs_val_log_reg)
        print(f"Reg log: best threshold = {thresh_log_reg:.3f}, validation F1 = {f1_val:.3f}")
        logistic_mask_log_reg = sigmoid(tx_te @ w_log_reg) >= thresh_log_reg


        y_te_pred_log_reg = np.zeros((np.shape(tx_te)[0],1))
        y_te_pred_log_reg[logistic_mask_log_reg] = 1

        #Confusion matrix display
        confusion_matrix_log_reg, confusion_matrix_ratio_log_reg = confusion_matrix_custom(y_te, y_te_pred_log_reg)
        display_confusion_matrix(confusion_matrix_log_reg, confusion_matrix_ratio_log_reg, "reg_logistic")



    #### LOGISTIC REGRESSION SGD ####

    #Optimization params
    max_iters_sgd = 500_000
    gamma_sgd = 0.001
    lambda_sgd = 1e-2

    print("Parameters for logistic regression SGD:\n")
    print(f"max_iters = {max_iters_sgd} \n")
    print(f"gamma = {gamma_sgd} \n")
    print(f"lambda = {lambda_sgd} \n")

    #Pure logistic regression sgd
    if launch_pure_log_sgd == True :
        w_log_sgd, losses_train_log_sgd, losses_test_log_sgd = logistic_regression_sgd(y_tr, tx_tr, y_te, tx_te, initial_w, max_iters_sgd, gamma_sgd, rng_sgd1)
        print("Pure logistic regression SGD: OK \n")

        #Losses display
        display_losses(losses_train_log_sgd, losses_test_log_sgd, "logistic_SGD")

        #Tune threshold
        probs_val_log_sgd = sigmoid(tx_te @ w_log_sgd).ravel()
        thresh_log_sgd, f1_val = threshold_tuning(y_te, probs_val_log_sgd)
        print(f"SGD log: best threshold = {thresh_log_sgd:.3f}, validation F1 = {f1_val:.3f}")
        logistic_mask_log_sgd = sigmoid(tx_te @ w_log_sgd) >= thresh_log_sgd

        y_te_pred_log_sgd = np.zeros((np.shape(tx_te)[0],1))
        y_te_pred_log_sgd[logistic_mask_log_sgd] = 1

        #Display
        confusion_matrix_log_sgd, confusion_matrix_ratio_log_sgd = confusion_matrix_custom(y_te, y_te_pred_log_sgd)
        display_confusion_matrix(confusion_matrix_log_sgd, confusion_matrix_ratio_log_sgd, "pure_logistic_SGD")

    #Penalized logistic regression sgd
    if launch_reg_log_sgd == True :
        w_log_reg_sgd, losses_train_log_reg_sgd, losses_test_log_reg_sgd = reg_logistic_regression_sgd(y_tr, tx_tr, y_te, tx_te, lambda_sgd, initial_w, max_iters_sgd, gamma_sgd, rng_sgd2)
        print("Penalized logistic regression SGD: OK \n")

        #Losses display
        display_losses(losses_train_log_reg_sgd, losses_test_log_reg_sgd, "logistic_regularized_SGD")

        #Tune threshold
        probs_val_reg_log_sgd = sigmoid(tx_te @ w_log_reg_sgd).ravel()
        thresh_reg_log_sgd, f1_val = threshold_tuning(y_te, probs_val_reg_log_sgd)
        print(f"SGD reg log: best threshold = {thresh_reg_log_sgd:.3f}, validation F1 = {f1_val:.3f}")
        logistic_mask_log_reg_sgd = sigmoid(tx_te @ w_log_reg_sgd) >= thresh_reg_log_sgd

        y_te_pred_log_reg_sgd = np.zeros((np.shape(tx_te)[0],1))
        y_te_pred_log_reg_sgd[logistic_mask_log_reg_sgd] = 1

        #Confusion matrix display
        confusion_matrix_log_reg_sgd, confusion_matrix_ratio_log_reg_sgd = confusion_matrix_custom(y_te, y_te_pred_log_reg_sgd)
        display_confusion_matrix(confusion_matrix_log_reg_sgd, confusion_matrix_ratio_log_reg_sgd, "reg_logistic_SGD")


    #### kNN ####
    if launch_kNN == True :
        #Prediction
        N = 1
        y_te_pred_kNN = kNN_custom_vectorized(x_tr, y_tr, x_te, N)

        print("Parameters kNN:\n")
        print(f"N = {N}")

        #Confusion matrix
        confusion_matrix_kNN, confusion_matrix_ratio_kNN = confusion_matrix_custom(y_te, y_te_pred_kNN)
        display_confusion_matrix(confusion_matrix_kNN, confusion_matrix_ratio_kNN, "kNN")


    #### SVM ####
    if launch_SVM == True :
        #SVM params
        max_iter_SVM = 1_000_000
        gamma_SVM = 100
        lambda_SVM = 0.01

        print("Parameters for SVM:\n")
        print(f"max_iters = {max_iter_SVM} \n")
        print(f"gamma = {gamma_SVM}\n")
        print(f"lambda = {lambda_SVM}\n")

        #SVM training
        # Labels in {-1, +1} for the hinge loss (SVM only)
        y_tr_svm = 2 * y_tr - 1
        w_SVM, costs, iters = sgd_for_svm_demo(y_tr_svm, tx_tr, max_iter_SVM, gamma_SVM, lambda_SVM, rng_svm)

        # Prediction back to {0, 1} for confusion_matrix_custom
        scores_val_SVM = (tx_te @ w_SVM).ravel()

        plot_SVM(iters, costs)

        grid_SVM = np.percentile(scores_val_SVM, np.linspace(50, 99.5, 200))
        thresh_SVM, f1_val = threshold_tuning(y_te, scores_val_SVM, grid=grid_SVM)
        print(f"SVM: best threshold = {thresh_SVM:.3f}, validation F1 = {f1_val:.3f}")

        y_te_pred_SVM = (scores_val_SVM >= thresh_SVM).astype(float).reshape(-1, 1)

        confusion_matrix_SVM, confusion_matrix_ratio_SVM = confusion_matrix_custom(y_te, y_te_pred_SVM)
        display_confusion_matrix(confusion_matrix_SVM, confusion_matrix_ratio_SVM, "SVM")



    ####################### SUBMISSION PIPELINE #######################

    #Reference x_test to get indices for final submission
    path_dataset_x = os.path.join(DATA_DIR, "x_test.csv")
    x_test_ref = np.genfromtxt(
        path_dataset_x,
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
        #Only adjust threshold if no downsampling
        logistic_mask_log = sigmoid(tx_test @ w_log) >= thresh_log
        y_test_pred_log = - np.ones((np.shape(tx_test)[0],1))
        y_test_pred_log[logistic_mask_log] = 1

        #Prediction with indices
        y_submission_pred_log = np.hstack([indices_array, y_test_pred_log])

        #Save prediction
        np.savetxt(
            "predictions/y_test_pred_log.csv",
            y_submission_pred_log,
            delimiter=",",
            header="Id,Prediction",
            comments="",
            fmt="%d"
        )
        print("Submission pure log: OK\n")

    #Penalized logistic regression
    if launch_reg_log == True :

        #Class choice
        logistic_mask_log_reg = sigmoid(tx_test @ w_log_reg) >= thresh_log_reg
        y_test_pred_log_reg = - np.ones((np.shape(tx_test)[0],1))
        y_test_pred_log_reg[logistic_mask_log_reg] = 1

        #Prediction with indices
        y_submission_pred_log_reg = np.hstack([indices_array, y_test_pred_log_reg])

        #Save prediction
        np.savetxt(
            "predictions/y_test_pred_log_reg.csv",
            y_submission_pred_log_reg,
            delimiter=",",
            header="Id,Prediction",
            comments="",
            fmt="%d"
        )
        print("Submission log reg: OK\n")

    if launch_pure_log_sgd == True :

        #Class choice
        logistic_mask_log_sgd = sigmoid(tx_test @ w_log_sgd) >= thresh_log_sgd
        y_test_pred_log_sgd = - np.ones((np.shape(tx_test)[0],1))
        y_test_pred_log_sgd[logistic_mask_log_sgd] = 1

        #Prediction with indices
        y_submission_pred_log_sgd = np.hstack([indices_array, y_test_pred_log_sgd])

        #Save prediction
        np.savetxt(
            "predictions/y_test_pred_log_sgd.csv",
            y_submission_pred_log_sgd,
            delimiter=",",
            header="Id,Prediction",
            comments="",
            fmt="%d"
        )
        print("Submission pure log sgd: OK\n")

    #Penalized logistic regression sgd
    if launch_reg_log_sgd == True :

        #Class choice
        logistic_mask_log_reg_sgd = sigmoid(tx_test @ w_log_reg_sgd) >= thresh_reg_log_sgd
        y_test_pred_log_reg_sgd = - np.ones((np.shape(tx_test)[0],1))
        y_test_pred_log_reg_sgd[logistic_mask_log_reg_sgd] = 1

        #Prediction with indices
        y_submission_pred_log_reg_sgd = np.hstack([indices_array, y_test_pred_log_reg_sgd])

        #Save prediction
        np.savetxt(
            "predictions/y_test_pred_log_reg_sgd.csv",
            y_submission_pred_log_reg_sgd,
            delimiter=",",
            header="Id,Prediction",
            comments="",
            fmt="%d"
        )
        print("Submission log reg sgd: OK\n")

    #SVM
    if launch_SVM == True :
        #Prediction
        y_test_pred_SVM = np.where(tx_test @ w_SVM >= thresh_SVM, 1, -1).reshape(-1, 1)

        #Prediction with indices
        y_submission_pred_SVM = np.hstack([indices_array, y_test_pred_SVM])

        #Save prediction
        np.savetxt(
            "predictions/y_test_pred_SVM.csv",
            y_submission_pred_SVM,
            delimiter=",",
            header="Id,Prediction",
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
            header="Id,Prediction",
            comments="",
            fmt="%d"
        )
        print("Submission kNN: OK\n")