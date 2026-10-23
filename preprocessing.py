import numpy as np

def infer_codes(col):
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


def fit_codes(x_train, columns, code_exceptions):
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


def update_codes(x, columns, missing_codes, zero_codes):
    """
    Replace survey answer codes for "Don't know / Not sure" and "Refused" with NaN, and "None" with 0.

    Args:
        x: numpy array of shape (N,D)
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

def remove_columns(x, columns, remove):
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


def clean_data(x_train, x_test, columns, remove, code_exceptions):
    """
    Remove columns and replace survey codes. Codes are fitted on x_train only.
    
    Args:
        x_train: numpy array of shape (N,D), training data
        x_test: numpy array of shape (N,D), test data
        columns: list of length D, the column names.
        remove: list of column names to be removed 
        code_exceptions: dictionary for column names where the infer_codes dont apply
    
    Returns:
        x_train: cleaned x_train
        x_test: cleand x_test
        columns_reduced: list of column names after remove_columns()
    """
    x_train, columns_reduced = remove_columns(x_train, columns, remove)
    x_test, _ = remove_columns(x_test, columns, remove)

    missing_codes, zero_codes = fit_codes(x_train, columns, code_exceptions)
    x_train = update_codes(x_train, columns_reduced, missing_codes, zero_codes)
    x_test = update_codes(x_test, columns_reduced, missing_codes, zero_codes)

    return x_train, x_test, columns_reduced
