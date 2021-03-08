.. highlight:: shell

regression errors
~~~~~~~~~~~~~~~~~

**path**  ``orion.primitives.timeseries_errors.regression_errors``

**description** this primitive computes an array of absolute errors comparing predictions and expected output. Optionally smooth them using EWMA.

========================== =================== ======================================================================
argument                    type                description  

**parameters**
---------------------------------------------------------------------------------------------------------------------

 ``y``                      ``numpy.ndarray``   ground truth
 ``y_hat``                  ``numpy.ndarray``   predicted values

**hyperparameters**
---------------------------------------------------------------------------------------------------------------------

 ``smooth``                 ``bool``            indicates whether the returned errors should be smoothed with EWMA 
 ``smoothing_window``       ``float``           size of the smoothing window, expressed as a proportion of the total 

**output**
---------------------------------------------------------------------------------------------------------------------

 ``errors``                 ``numpy.ndarray``   array of errors
========================== =================== ======================================================================


.. ipython:: python
    :okwarning:

    import numpy as np
    from mlprimitives import load_primitive

    primitive = load_primitive('orion.primitives.timeseries_errors.regression_errors')
    y = np.array([[1]] * 100)
    y_hat = np.array([[.99]] * 100)

    errors = primitive.produce(y=y, y_hat=y_hat)
    print("average error value: {:.2f}".format(errors.mean()))


