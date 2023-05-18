.. highlight:: shell

SimpleImputer
~~~~~~~~~~~~~

**path**: ``sklearn.impute.SimpleImputer``

**description**: this primitive is an imputation transformer for filling missing values.

see `json <https://github.com/MLBazaar/mlstars/blob/master/mlstars/primitives/sklearn.impute.SimpleImputer.json>`__.

==================== ========================================================= ==========================================
argument              type                                                      description  

**parameters**
-------------------------------------------------------------------------------------------------------------------------
 ``X``                ``numpy.ndarray``                                         n-dimensional sequence of values

**hyperparameters**
-------------------------------------------------------------------------------------------------------------------------

 ``missing_values``   ``int``, ``float``, ``str``, ``numpy.nan``, or ``None``   the placeholder for the missing values. All occurrences of ``missing_values`` will be imputed
  ``strategy``         ``str``                                                  the imputation strategy. If ``mean``, then replace missing values using the mean along each column. Can only be used with numeric data. If ``median``, then replace missing values using the median along each column. Can only be used with numeric data. If ``most_frequent``, then replace missing using the most frequent value along each column. Can be used with strings or numeric data. If ``constant``, then replace missing values with ``fill_value``. Can be used with strings or numeric data.
 ``fill_value``       ``int``, ``float``, or ``str``                            when ``strategy == "constant"``, ``fill_value`` is used to replace all occurrences of ``missing_values``. If left to the default, ``fill_value`` will be 0 when imputing numerical data and "missing_value" for strings or object data types
 ``verbose``          ``bool``                                                  controls the verbosity of the imputer
 ``copy``             ``bool``                                                  if True, a copy of X will be created

**output**
-------------------------------------------------------------------------------------------------------------------------

 ``X``                ``numpy.ndarray``                                         a transformed version of X
==================== ========================================================= ==========================================

.. ipython:: python
    :okwarning:

    import numpy as np
    from mlstars import load_primitive

    X = np.array([1] * 4 + [np.nan]).reshape(-1, 1)
    primitive = load_primitive('sklearn.impute.SimpleImputer', 
        arguments={"X": X})

    primitive.fit()
    primitive.produce(X=X)
