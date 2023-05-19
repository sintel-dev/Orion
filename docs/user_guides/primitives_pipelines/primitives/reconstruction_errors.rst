.. highlight:: shell

reconstuction errors
~~~~~~~~~~~~~~~~~~~~

**path**  ``orion.primitives.timeseries_errors.reconstruction_errors``

**description** this primitive computes an array of errors comparing reconstructed and expected output. There are three main approaches for computing the discrepancies: point-wise, area, and dtw.

see `json <https://github.com/sintel-dev/Orion/blob/master/orion/primitives/jsons/orion.primitives.timeseries_errors.reconstruction_errors.json>`__.

========================== =================== ======================================================================
argument                    type                description  

**parameters**
---------------------------------------------------------------------------------------------------------------------

 ``y``                      ``numpy.ndarray``   ground truth
 ``y_hat``                  ``numpy.ndarray``   predicted values

**hyperparameters**
---------------------------------------------------------------------------------------------------------------------

 ``step_size``              ``int``             indicates the number of steps between windows in the predicted values
 ``score_window``           ``int``             indicates the size of the window over which the scores are calculated
 ``rec_error_type``        ``str``             reconstruction error types, can be one of ``["point", "area", "dtw"]``
 ``smooth``                 ``bool``            indicates whether the returned errors should be smoothed 
 ``smoothing_window``       ``float``           size of the smoothing window, expressed as a proportion of the total 

**output**
---------------------------------------------------------------------------------------------------------------------

 ``errors``                 ``numpy.ndarray``   array of errors
========================== =================== ======================================================================


.. ipython:: python
    :okwarning:

    import numpy as np
    from mlstars import load_primitive

    primitive = load_primitive('orion.primitives.timeseries_errors.reconstruction_errors')
    y = np.array([[1]] * 100)
    y_hat = np.array([[.99]] * 100)

    errors, predictions = primitive.produce(y=y, y_hat=y_hat)
    print("average error value: {:.2f}".format(errors.mean()))


