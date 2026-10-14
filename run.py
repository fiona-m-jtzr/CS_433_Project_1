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

x, y = load_data_train(data_dir=DATA_DIR,no_nan=True)