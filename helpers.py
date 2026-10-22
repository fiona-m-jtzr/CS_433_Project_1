"""some helper functions for data pre-processing."""

import numpy as np
import os

# Derived variables to exclude
DESIGN = {"_STATE", 
          "_PSU", 
          "_STSTR", 
          "_STRWT", 
          "_RAWRAKE", 
          "_WT2RAKE",
          "_CHISPNC", 
          "_CRACE1", 
          "_CPRACE", 
          "_CLLCPWT", 
          "_DUALUSE",
          "_DUALCOR", 
          "_LLCPWT"}

# Derived variables to exclude
FLAGS  = {"_MISFRTN", 
          "_MISVEGN", 
          "_FRTRESP", 
          "_VEGRESP",
          "_FRT16", 
          "_VEG23", 
          "_FRUITEX", 
          "_VEGETEX"}

# Information on how missing values are encoded
MISSING_CODES = { "_RFHLTH": {9},
                    "_HCVU651": set(), 
                    "_RFHYPE5": {9},
                    "_CHOLCHK": {9},
                    "_RFCHOL": {9},
                    "_LTASTH1": {9},
                    "_CASTHM1": {9},
                    "_ASTHMS1": {9},
                    "_DRDXAR1": set(), 
                    "_PRACE1": {77, 99},
                    "_MRACE1": {77, 99},
                    "_HISPANC": {9},
                    "_RACE": {9},
                    "_RACEG21": {9},
                    "_RACEGR3": {9},
                    "_RACE_G1": set(),  
                    "_AGEG5YR": {14},
                    "_AGE65YR": {3},
                    "_AGE80": set(),  
                    "_AGE_G": set(),  
                    "_BMI5": set(), 
                    "_BMI5CAT": set(), 
                    "_RFBMI5": {9},
                    "_CHLDCNT": {9},
                    "_EDUCAG": {9},
                    "_INCOMG": {9},
                    "_SMOKER3": {9},
                    "_RFSMOK3": {9},
                    "_RFBING5": {9},
                    "_DRNKWEK": {99900},
                    "_RFDRHV5": {9},
                    "_FRUTSUM": set(),  
                    "_VEGESUM": set(),  
                    "_FRTLT1": {9},
                    "_VEGLT1": {9},
                    "_TOTINDA": {9},
                    "_MINAC11": set(),  
                    "_MINAC21": set(),  
                    "_PACAT1": {9},
                    "_PAINDX1": {9},
                    "_PA150R2": {9},
                    "_PA300R2": {9},
                    "_PA30021": {9},
                    "_PASTRNG": {9},
                    "_PAREC1": {9},
                    "_PASTAE1": {9},
                    "_LMTACT1": {9},
                    "_LMTWRK1": {9},
                    "_LMTSCL1": {9},
                    "_RFSEAT2": {9},
                    "_RFSEAT3": {9},
                    "_FLSHOT6": {9},
                    "_PNEUMO2": {9},
                    "_AIDTST3": {9}}

# Continuous variables
CONT_VARS = {"_BMI5",    
                "_HTIN4",   
                "_HTM4",     
                "_WTKG3",    
                "_DRNKWEK",  
                "_FRUTSUM",
                "_VEGESUM"}

# Decimal variables
DEC_VARS = {"_BMI5",
            "_DRNKWEK", 
            "_FRUTSUM", 
            "_VEGESUM"}



def replace_missing_codes(data_x, selected_headers):
    """Replace values indicating missing data with NaN."""

    data_x = np.asarray(data_x, dtype=float).copy()

    for j, name in enumerate(selected_headers):
        codes = MISSING_CODES.get(name, set())

        if codes:
            mask = np.isin(data_x[:, j], list(codes))
            data_x[mask, j] = np.nan

    return data_x


def replace_nan_by_mode(col):
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
    


def replace_nan_by_mean(col):
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


def one_hot_encode(col):
    """One-hot encode categorical variables."""
        
    unique_vals = np.unique(col)

    # Discard one value for linear independence
    unique_vals = unique_vals[1:] 
        
    n_samples = len(col)
    encoded_col = np.zeros((n_samples, len(unique_vals)), dtype=int)
    
    for idx, val in enumerate(unique_vals):
        encoded_col[col == val, idx] = 1

    return encoded_col, unique_vals


def standardize(col):
    """Standardize continuous variables."""

    mean_col = np.mean(col)
    mean_sub_col = col - mean_col
    std_col = np.std(col)

    if std_col == 0:
        std_col = 1.0

    standardized_col = mean_sub_col / std_col

    return standardized_col, mean_col, std_col


def load_data_train(data_dir, imputation=True):
    """Load data for training."""

    path_dataset_x = os.path.join(data_dir, "x_train.csv")
    path_dataset_y = os.path.join(data_dir, "y_train.csv")

    header = np.genfromtxt(path_dataset_x, delimiter=",", names=True, max_rows=1).dtype.names

    # Extract only derived variables
    derived_var = [i for i, s in enumerate(header) if s.startswith("_") and s not in DESIGN | FLAGS]

    derived_var_header = [header[i] for i in derived_var]
   
    data_x = np.genfromtxt(path_dataset_x, delimiter=",", skip_header=1, usecols=derived_var)
    data_y = np.genfromtxt(path_dataset_y, delimiter=",", skip_header=1, usecols=1)

    # Mark all missing values as NaN
    data_x = replace_missing_codes(data_x, derived_var_header)

    # Process each column according to specifications
    processed_cols = []
    cont_params = {}
    cat_params = {}

    if imputation:

        mask_nan = np.zeros(data_x.shape[1])
        for idx, col_name in enumerate(derived_var_header):
                
            col = data_x[:, idx]
    
            # Adjust decimal variables
            if col_name in DEC_VARS:
                col = col / 100
    
            # Replace NaN with mean for continuous variables and standardize
            if col_name in CONT_VARS:
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
        mask_nan = np.isnan(data_x).any(axis=0)
        data_x = data_x[:, ~mask_nan]

        # Keep only the headers where the mask is True 
        derived_var_header_filtered = [name for name, keep in zip(derived_var_header, ~mask_nan) if keep]

        print(f"Reduced to {data_x.shape[1]} features.")
        print(derived_var_header_filtered)

        for idx, col_name in enumerate(derived_var_header_filtered):
                        
            col = data_x[:, idx]
    
            # Adjust decimal variables
            if col_name in DEC_VARS:
                col = col / 100
    
            # Replace NaN with mean for continuous variables and standardize
            if col_name in CONT_VARS:
                standardized_col, mean_col, std_col = standardize(col)
                processed_cols.append(standardized_col.reshape(-1, 1))
                cont_params[col_name] = {'mean_col': mean_col, 'std_col': std_col}
    
            # Replace NaN with mode for categorical variables and one-hot-encode
            else:
                col_no_nan, mode_val = replace_nan_by_mode(col)
                encoded_col, unique_vals = one_hot_encode(col)
                processed_cols.append(encoded_col)
                cat_params[col_name] = {'mode_val': mode_val, 'unique_vals': unique_vals}
    
    # Transform labels from -1, 1 to 0, 1
    data_y = (data_y == 1).astype(float)

    return np.hstack(processed_cols), np.array(data_y), cont_params, cat_params, mask_nan


def load_data_test(data_dir, cont_params, cat_params, mask_nan, imputation=True):
    """Load data for testing."""
    path_dataset_x = os.path.join(data_dir, "x_test.csv")
    
    header = np.genfromtxt(path_dataset_x, delimiter=",", names=True, max_rows=1).dtype.names
    
    # Extract only derived variables
    derived_var = [i for i, s in enumerate(header) if s.startswith("_") and s not in DESIGN | FLAGS]

    derived_var_header = [header[i] for i in derived_var]
    
    data_x = np.genfromtxt(path_dataset_x, delimiter=",", skip_header=1, usecols=derived_var)

    # Mark all missing values as NaN
    data_x = replace_missing_codes(data_x, derived_var_header)

    # Process each column according to specifications
    processed_cols = []

    if imputation:

        for idx, col_name in enumerate(derived_var_header):
                    
            col = data_x[:, idx]
    
            # Adjust decimal variables
            if col_name in DEC_VARS:
                col = col / 100
    
            # Replace NaN with mean for continuous variables and standardize
            if col_name in CONT_VARS:
                params = cont_params[col_name]
                mask_nan = np.isnan(col)
                col[mask_nan] = params['mean_val']
                mean_sub_col = col - params['mean_col']
                standardized_col = mean_sub_col / params['std_col']
                processed_cols.append(standardized_col.reshape(-1, 1))
    
            # Replace NaN with mode for categorical variables and one-hot-encode
            else:
                params = cat_params[col_name]
                mask_nan = np.isnan(col)
                col[mask_nan] = params['mode_val']
                
                n_samples = len(col)
                encoded_col = np.zeros((n_samples, len(params['unique_vals'])), dtype=int)
                
                for u_idx, val in enumerate(params['unique_vals']):
                    encoded_col[col == val, u_idx] = 1
    
                processed_cols.append(encoded_col)
        
    else:

        data_x = data_x[:, ~mask_nan]

        derived_var_header_filtered = [name for name, keep in zip(derived_var_header, ~mask_nan) if keep]
    
        for idx, col_name in enumerate(derived_var_header_filtered):
                    
            col = data_x[:, idx]
    
            # Adjust decimal variables
            if col_name in DEC_VARS:
                col = col / 100
    
            # Replace NaN with mean for continuous variables and standardize
            if col_name in CONT_VARS:
                params = cont_params[col_name]
                mask_nan = np.isnan(col)
                col[mask_nan] = params['mean_val']
                mean_sub_col = col - params['mean_col']
                standardized_col = mean_sub_col / params['std_col']
                processed_cols.append(standardized_col.reshape(-1, 1))
    
            # Replace NaN with mode for categorical variables and one-hot-encode
            else:
                params = cat_params[col_name]
                mask_nan = np.isnan(col)
                col[mask_nan] = params['mode_val']
                
                n_samples = len(col)
                encoded_col = np.zeros((n_samples, len(params['unique_vals'])), dtype=int)
                
                for u_idx, val in enumerate(params['unique_vals']):
                    encoded_col[col == val, u_idx] = 1
    
                processed_cols.append(encoded_col)

    return np.hstack(processed_cols)


def downsample_classes(data_x, data_y, random_state=0):
    """Downsamples the majority class to match the size of the minority class."""
    
    np.random.seed(random_state)
    
    # Identify indices for each class
    minority_class_val = 1.0 if np.sum(data_y == 1.0) < np.sum(data_y == 0.0) else 0.0
    majority_class_val = 0.0 if minority_class_val == 1.0 else 1.0
    
    minority_indices = np.where(data_y == minority_class_val)[0]
    majority_indices = np.where(data_y == majority_class_val)[0]
    
    n_minority = len(minority_indices)
    
    downsampled_majority_indices = np.random.choice(majority_indices, size=n_minority, replace=False)
    
    balanced_indices = np.concatenate([minority_indices, downsampled_majority_indices])
    np.random.shuffle(balanced_indices)
    
    return data_x[balanced_indices], data_y[balanced_indices]
    
