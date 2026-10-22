'''
File name: part_2.py
Authors: Mayeul, Sondre, Fiona
Python version: 3.13.9  
'''

import numpy as np
import os
from helpers import *
from implementations import *


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")

# Load and clean data
x_train, y_train, cont_params, cat_params, mask = load_data_train(DATA_DIR, imputation=False)
x_test = load_data_test(DATA_DIR, cont_params, cat_params, mask, imputation=False)

# Downsample for balanced classes
x_train_downsampled, y_train_downsampled = downsample_classes(x_train, y_train)

# Initialize weights
initial_w = np.zeros(x_train.shape[1])

# Sanity check of dimensions
print(x_train.shape)
print(y_train.shape)
print(initial_w.shape)

w, loss = logistic_regression(y_train, x_train, initial_w, 500, 0.01)

print(loss)

