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

x_train, y_train, cont_params, cat_params, mask_nan, missing_codes, zero_codes = load_data_train(DATA_DIR, imputation_threshold=0.1)
x_test = load_data_test(DATA_DIR, cont_params, cat_params, mask_nan, missing_codes, zero_codes)

assert x_train.shape[1] == x_test.shape[1]
assert not np.isnan(x_train).any() and not np.isnan(x_test).any()
print(x_train.shape, x_test.shape)
