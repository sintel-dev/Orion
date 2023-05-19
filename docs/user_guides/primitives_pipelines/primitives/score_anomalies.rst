.. highlight:: shell

score anomalies
~~~~~~~~~~~~~~~

**path**: ``orion.primitives.tadgan.score_anomalies``

**description**: this primitive computes an array of anomaly scores based on a combination of reconstruction error and critic output.

see `json <https://github.com/sintel-dev/Orion/tree/master/orion/primitives/jsons/orion.primitives.tadgan.score_anomalies.json>`__.

========================== =================== =================================================================================================
argument                    type                description  

**parameters**
------------------------------------------------------------------------------------------------------------------------------------------------

 ``y``                      ``numpy.ndarray``    ground truth
 ``y_hat``                  ``numpy.ndarray``    predicted values. Each timestamp has multiple predictions
 ``critic``                 ``numpy.ndarray``    critic score. Each timestamp has multiple critic scores
 ``index``                  ``numpy.ndarray``    time index for each y (start position of the window)

**hyperparameters**
------------------------------------------------------------------------------------------------------------------------------------------------

 ``score_window``           ``int``              size of the window over which the scores are calculated. If not given, 10 is used
 ``critic_smooth_window``   ``int``              size of window over which smoothing is applied to critic. If not given, 200 is used
 ``error_smooth_window``    ``int``              size of window over which smoothing is applied to error. If not given, 200 is used.
 ``rec_error_type``         ``str``              the method to compute reconstruction error. Can be one of `["point", "area", "dtw"]`
 ``comb``                   ``str``              how to combine critic and reconstruction error. Can be one of `["mult", "sum", "rec"]`
 ``lambda_rec``             ``float``            used if `comb="sum"` as a lambda weighted sum to combine scores. If not given, 0.5 is used.

**output**
------------------------------------------------------------------------------------------------------------------------------------------------

 ``errors``                 ``numpy.ndarray``    array of anomaly scores
 ``true_index``             ``numpy.ndarray``    time index of errors
 ``true``                   ``numpy.ndarray``    ground truth
 ``predictions``            ``numpy.ndarray``    predicted sequence
========================== =================== =================================================================================================


.. ipython:: python
    :okwarning:

    import numpy as np
    from mlstars import load_primitive

    primitive = load_primitive('orion.primitives.tadgan.score_anomalies', 
        arguments={"error_smooth_window": 10, "critic_smooth_window": 10,
                   "score_window": 10, "comb": "rec"})

    y = np.array([1] * 100).reshape(1, -1, 1)
    y_hat = [0.9] * 40 + [0.5] * 10 + [1.1] * 10 + [0.99] * 40
    y_hat = np.array(y_hat).reshape(1, -1, 1)
    critic = np.array([[0.5]])
    index = np.array([[1]])

    errors, true_index, true, predictions = primitive.produce(
        y=y, y_hat=y_hat, critic=critic, index=index)
    print("average error value: {:.2f}".format(errors.mean()))

