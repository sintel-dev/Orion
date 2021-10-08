import numpy as np
import pandas as pd


def slice_array_by_dims(X, target_index, axis=0):
    """Slice an ndarray by given dimensions.

    This function creates a copy of X then slices the array from the specificed
    dimension and returns the sliced array of the specified index.

    Args:
        X (ndarray):
            Array of input sequence.
        target_index (int or list[int]):
            Integer of the index to extract. Can be a list of integer values
            to extract multiple dimensions.
        axis (int or str):
            Optional. Axis along which to extract value. Default is ``0``.

    Returns:
        ndarray:
            Array of sliced values.
    """
    if isinstance(target_index, int):
        target_index = [target_index]

    dims = len(X.shape)
    if axis >= (dims):
        raise ValueError("Axis {} is outside the dimensions of X ({}).".format(axis, dims))

    indices = [slice(None)] * dims
    indices[axis] = target_index

    return X[tuple(indices)].copy()


def fillna(X, value=None, method=None, axis=None, limit=None, downcast=None):
    """Impute missing values.

    This function fills the missing values of the input sequence with the next/
    previous known value. If there are contigous NaN values, they will all be
    filled with the same next/previous known value.

    Args:
        X (ndarray or pandas.DataFrame):
            Array of input sequence.
        value:
            Optional. Value to use to fill holes (e.g. 0), alternately a dict/Series/DataFrame of
            values specifying which value to use for each index (for a Series) or column
            (for a DataFrame). Values not in the dict/Series/DataFrame will not be filled.
            This value cannot be a list. Default is None.
        method (str or list):
            Optional. String or list of strings describing whether to use forward or backward
            fill. pad / ffill: propagate last valid observation forward to next valid.
            backfill / bfill: use next valid observation to fill gap. Otherwise use ``None`` to
            fill with desired value. Possible values include
            ``[‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None]``. Default is None.
        axis (int or str):
            Optional. Axis along which to fill missing value. Possible values include 0 or
            "index", 1 or "columns". Default is None.
        limit (int):
            Optional. If method is specified, this is the maximum number of consecutive NaN values
            to forward/backward fill. In other words, if there is a gap with more than this number
            of consecutive NaNs, it will only be partially filled. If method is not specified,
            this is the maximum number of entries along the entire axis where NaNs will be filled.
            Must be greater than 0 if not None. Default is None.
        downcast (dict):
            Optional. A dict of item->dtype of what to downcast if possible, or the string "infer"
            which will try to downcast to an appropriate equal type (e.g. float64 to int64 if
            possible). Default is None.

    Returns:
        ndarray:
            Array of input sequence with imputed values.
    """
    if isinstance(method, str) or method is None:
        method = [method]

    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X_ = pd.Series(X)
        else:
            X_ = pd.DataFrame(X)

    else:
        X_ = X.copy()

    for fill in method:
        X_ = X_.fillna(value=value, method=fill, axis=axis, limit=limit, downcast=downcast)

    return X_.values
