import pandas as pd


def fillna(X, method=["ffill"], **kwargs):
    """Impute missing values.

    This function fills the missing values of the input sequence with the next/
    previous known value. If there are contigous NaN values, they will all be
    filled with the same next/previous known value.

    Args:
        X (ndarray):
            Array of input sequence.
        method (str or list):
            Optional. String describing whether to use forward or backward fill. pad / ffill:
            propagate last valid observation forward to next valid. backfill / bfill: use next
            valid observation to fill gap.

    Returns:
        ndarray:
            Array of input sequence with imputed values.
    """
    if isinstance(method, str):
        method = [method]

    X_ = pd.Series(X)

    for fill in method:
        X_ = X_.fillna(method=fill, **kwargs)

    return X_.values
