# Description
This repository inventoriates our analyses on the provided health dataset.

# Goal
The goal is to provide good predictions of the risk of cardiovascular diseases.

# Website
The competition website is : https://www.aicrowd.com/challenges/epfl-machine-learning-project-1

# Overleaf
https://www.overleaf.com/project/6ab150541dce52f1539cb0d1

# Files description
- run.py : main file for classification training pipeline and prediction. The user can custom pipeline in section "PARAMS TO CHANGE".
- helpers.py : contains utils functions for preprocessing (loading, imputing depending on percentage of NaNs, downsampling, etc.).
- dictionnaries.py : operations to do on the features. 
- helpers_analysis.py : contains utils functions for training classifier (logistic SGD, SVM, kNN) and plot the results (losses and confusion matrices). 
- implementations.py : contains the 6 functions assigned to be implemented.

# Folders description
- dataset : contains train and test, features and output data.
- image : contains losses and confusion matrix plots during classifiers trainings.
- predictions : contains .csv submission file for each classification technique. 

# Best submissions until now
- 29/10/2026 : F1 = 0.424, accuracy = 0.870

# User guide
- Put the datasets in dataset folder
- Activate python environment
- Launch run.py
