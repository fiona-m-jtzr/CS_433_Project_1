'''
File name: helpers_preprocessing.py
Authors: Sondre, Fiona
Python version: 3.13.9  
'''
import numpy as np
import os
import dictionaries as d

def infer_codes(col: np.ndarray) -> tuple[list[int], int | None]:
    """
    Infer the survey codes of a column from its largest value.
    BRFSS uses 7/77/777/7777 for "Don't know / Not sure" and 9/99/999/9999 for "Refused".
    88/98/888 are used for "None"

    Args:
        col: numpy array of shape (N,), one column of the data.

    Returns:
        missing: list of codes meaning "Don't know / Not sure" and "Refused" in that column
        zero: int or None, the code labelled "None" in the codebook, meaning zero.   
    """

    values = np.unique(col[~np.isnan(col)])
    if len(values) == 0:
        return [], None

    vmax = int(values.max())
    if vmax == 9:
        return [7,9], None
    if vmax == 99:
        return [77, 99], 88
    if vmax == 999:
        return [777, 999], 888
    if vmax == 9999:
        return [7777, 9999], None
    return [], None


def fit_codes(x_train: np.ndarray, columns: np.ndarray, code_exceptions: dict) -> tuple[dict, dict]:
    """
    Args:
        x_train: numpy arrray of shape (N,D)
        columns: list of length D, the column names.
        code_exceptions: dictionary for column names where the infer_codes dont apply
    
        Returns:
            missing_codes: dict mapping column name with code list that should become NaN.
            zero_codes: dict mapping column name with code that should become 0.
    """
    assert x_train.shape[1] == len(columns)
    missing_codes = {}
    zero_codes = {}
        
    for i, var_name in enumerate(columns):
        col = x_train[:, i]
        if var_name in code_exceptions:
            missing, zero = code_exceptions[var_name]
        else:
            missing, zero = infer_codes(col)

        if missing:
            missing_codes[var_name] = missing
        if zero is not None and zero in col:
            zero_codes[var_name] = zero

    return missing_codes, zero_codes


def update_codes(x: np.ndarray, columns: np.ndarray, missing_codes: dict, zero_codes: dict) -> np.ndarray:
    """
    Replace survey answer codes for "Don't know / Not sure" and "Refused" with NaN, and "None" with 0.

    Args:
        x: numpu array of shape (N,D)
        columns: list of length D, the column names. 
        missing_codes: dict mapping column name with code list that should become NaN.
        zero_codes: dict mapping column name with code that should become 0.
    
    Returns:
        x_updated: numpy array of shape (N,D), copy of x with replaced codes
    """
    assert x.shape[1] == len(columns)
    x_updated = x.copy()
   
    for i, var_name in enumerate(columns):
        col = x_updated[:, i]
        if var_name in missing_codes:
            x_updated[np.isin(col, missing_codes[var_name]), i] = np.nan
        if var_name in zero_codes:
            x_updated[col == zero_codes[var_name], i] = 0

    return x_updated



def remove_columns(x: np.ndarray, columns: np.ndarray, remove: list) -> tuple[np.ndarray, list[str]]:
    """
    Remove the columns in the list "remove".

    Args:
        x: numpy array of shape (N,D).
        columns: list of length D, the column names.
        remove: list of column names.

    Returns:
        x_reduced: numpy array of shape (N,D), with removed columns.
        columns_reduced: list of length D, the remaining column names.
    """
    assert x.shape[1] == len(columns)
    remove = set(remove)

    keep = [i for i, var_name in enumerate(columns) if var_name not in remove]
    x_reduced = x[:, keep]
    columns_reduced = [columns[i] for i in keep]

    return x_reduced, columns_reduced


def replace_nan_by_mode(col: np.ndarray) -> tuple[np.ndarray, float]:
    """Replace NaN values with the column's mode."""

    mask_nan = np.isnan(col)
    valid_vals = col[~mask_nan]
        
    if len(valid_vals) > 0:
        vals, counts = np.unique(valid_vals, return_counts=True)
        mode_val = vals[np.argmax(counts)]
        col[mask_nan] = mode_val
    else:
        col[mask_nan] = 0  

    return col, mode_val
    


def replace_nan_by_mean(col: np.ndarray) -> tuple[np.ndarray, float]:
    """Replace NaN values with the column's mean safely."""

    mask_nan = np.isnan(col)
    valid_vals = col[~mask_nan]
    
    if len(valid_vals) > 0:
        mean_val = np.mean(valid_vals)
        col[mask_nan] = mean_val
    else:
        mean_val = 0.0
        col[mask_nan] = mean_val
    
    return col, mean_val


def one_hot_encode(col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-hot encode categorical variables."""
        
    unique_vals = np.unique(col)

    # Discard one value for linear independence
    unique_vals = unique_vals[1:] 
        
    n_samples = len(col)
    encoded_col = np.zeros((n_samples, len(unique_vals)), dtype=int)
    
    for idx, val in enumerate(unique_vals):
        encoded_col[col == val, idx] = 1

    return encoded_col, unique_vals


def standardize(col: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Standardize continuous variables."""

    mean_col = np.mean(col)
    mean_sub_col = col - mean_col
    std_col = np.std(col)

    if std_col == 0:
        std_col = 1.0

    standardized_col = mean_sub_col / std_col

    return standardized_col, mean_col, std_col


def load_data_train(data_dir: str, imputation_threshold: float = 0.2) -> tuple[np.ndarray, np.ndarray, dict, dict, np.ndarray, dict, dict]:
    """Load data for training."""

    print(f'Starting data cleaning with imputation threshold set to {imputation_threshold}.')
    
    path_dataset_x = os.path.join(data_dir, "x_train.csv")
    path_dataset_y = os.path.join(data_dir, "y_train.csv")

    header = np.genfromtxt(path_dataset_x, delimiter=",", names=True, max_rows=1).dtype.names
   
    data_x = np.genfromtxt(path_dataset_x, delimiter=",", skip_header=1)
    data_y = np.genfromtxt(path_dataset_y, delimiter=",", skip_header=1, usecols=1)

    data_x_reduced, header_reduced = remove_columns(data_x, header, d.REMOVE)
    
    missing_codes, zero_codes = fit_codes(data_x_reduced, header_reduced, d.CODE_EXCEPTIONS)
    data_x_nan_val = update_codes(data_x_reduced, header_reduced, missing_codes, zero_codes)

    print(f'First steps of data cleaning done! The data matrix has shape {data_x_nan_val.shape}.')

    # Process each column according to specifications
    processed_cols = []
    mask_nan = np.zeros(data_x_reduced.shape[1], dtype=bool)
    cont_params = {}
    cat_params = {}
        
    for idx, col_name in enumerate(header_reduced):
        col = data_x_nan_val[:, idx]

        nan_frac = np.isnan(col).mean(axis=0)

        if nan_frac < imputation_threshold:
            # Replace NaN with mean for continuous variables and standardize
            if col_name in d.CONT_VARS:
                col_no_nan, mean_val = replace_nan_by_mean(col)
                standardized_col, mean_col, std_col = standardize(col_no_nan)
                processed_cols.append(standardized_col.reshape(-1, 1))
                cont_params[col_name] = {'mean_val': mean_val, 'mean_col': mean_col, 'std_col': std_col}

            # Replace NaN with mode for categorical variables and one-hot-encode
            else:
                col_no_nan, mode_val = replace_nan_by_mode(col)
                encoded_col, unique_vals = one_hot_encode(col_no_nan)
                processed_cols.append(encoded_col)
                cat_params[col_name] = {'mode_val': mode_val, 'unique_vals': unique_vals}

        else:
            mask_nan[idx] = True
        
    # Keep only the headers where the mask is True 
    header_reduced_no_nan = [name for name, keep in zip(header_reduced, ~mask_nan) if keep]

    data_x_nan_val_imputed = np.hstack(processed_cols)
    print(f"Kept {(~mask_nan).sum()} of {len(mask_nan)} input features corresponding to {data_x_nan_val_imputed.shape[1]} features after one-hot-encoding.")
    print(header_reduced_no_nan)

    # Transform labels from -1, 1 to 0, 1
    data_y = np.array((data_y == 1).astype(float))

    return data_x_nan_val_imputed, data_y, cont_params, cat_params, mask_nan, missing_codes, zero_codes



def load_data_test(data_dir: str, cont_params: dict, cat_params: dict, mask_nan: np.ndarray, missing_codes: dict, zero_codes: dict) -> np.ndarray:
    """Load data for testing."""

    path_dataset_x = os.path.join(data_dir, "x_test.csv")
    
    header = np.genfromtxt(path_dataset_x, delimiter=",", names=True, max_rows=1).dtype.names
       
    data_x = np.genfromtxt(path_dataset_x, delimiter=",", skip_header=1)

    data_x_reduced, header_reduced = remove_columns(data_x, header, d.REMOVE)
    data_x_nan_val = update_codes(data_x_reduced, header_reduced, missing_codes, zero_codes)

    data_x_nan_val_mask = data_x_nan_val[:, ~mask_nan]
    header_mask = [name for name, k in zip(header_reduced, ~mask_nan) if k]

    # Process each column according to specifications
    processed_cols = []

    for idx, col_name in enumerate(header_mask):
                
        col = data_x_nan_val_mask[:, idx]

        # Replace NaN with mean for continuous variables and standardize
        if col_name in d.CONT_VARS:
            params = cont_params[col_name]
            mask_nan_row = np.isnan(col)
            col[mask_nan_row] = params['mean_val']
            mean_sub_col = col - params['mean_col']
            standardized_col = mean_sub_col / params['std_col']
            processed_cols.append(standardized_col.reshape(-1, 1))

        # Replace NaN with mode for categorical variables and one-hot-encode
        else:
            params = cat_params[col_name]
            mask_nan_row = np.isnan(col)
            col[mask_nan_row] = params['mode_val']
            
            n_samples = len(col)
            encoded_col = np.zeros((n_samples, len(params['unique_vals'])), dtype=int)
            
            for u_idx, val in enumerate(params['unique_vals']):
                encoded_col[col == val, u_idx] = 1

            processed_cols.append(encoded_col)

    return np.hstack(processed_cols)


def downsample_classes(data_x: np.ndarray, data_y: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Downsamples the majority class to match the size of the minority class."""
    
    # Identify indices for each class
    minority_class_val = 1.0 if np.sum(data_y == 1.0) < np.sum(data_y == 0.0) else 0.0
    majority_class_val = 0.0 if minority_class_val == 1.0 else 1.0
    
    minority_indices = np.where(data_y == minority_class_val)[0]
    majority_indices = np.where(data_y == majority_class_val)[0]
    
    n_minority = len(minority_indices)
    
    downsampled_majority_indices = rng.choice(majority_indices, size=n_minority, replace=False)
    
    balanced_indices = rng.permutation(np.concatenate([minority_indices, downsampled_majority_indices]))
    
    return data_x[balanced_indices], data_y[balanced_indices]
