# CS_433_Project_1
Implementation of standard machine learning algorithms to determine the risk of a person in developing CVD based on features of their personal lifestyle factors.

Data processing:
The file helpers.py contains functions used for data cleaning. The only functions which need to be called in run.py to launch the data processing pipeline are:

load_data_train(data_dir: str, imputation_threshold: float = 0.2) -> tuple[np.ndarray, np.ndarray, dict, dict, np.ndarray, dict, dict]

and

load_data_test(data_dir: str, cont_params: dict, cat_params: dict, mask_nan: np.ndarray, missing_codes: dict, zero_codes: dict) -> np.ndarray

Numerical values representing missing data are correctly encoded as Nan. In a second step, Nan values are imputed using the mean for continuous variables and the mode for categorical variables. The argument imputation_threshold of the function load_data_train allows to set a percentage of acceptable Nan values per feature. Features containing more Nan values are discarded and not used for training. Continuous variables are standardised and categorical variables one-hot-encoded. Furthermore, the binary encoding of the value to predict is adjusted from -1, 1 to 0, 1.

The result is a reduced, but clean data matrix containing no Nan values as well as a binary array containing 0 and 1 for training. In addition, the function outputs statistics from the training set in parameter dictionaries and a boolean mask marking discarded features.

The test data is reduced to the same features used for training and imputed and standardised or one-hot-encoded using the statistics from the training data.


