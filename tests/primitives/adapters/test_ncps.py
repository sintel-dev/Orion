from unittest.mock import patch

import ncps
import numpy as np
import tensorflow as tf

from orion.primitives.adapters.ncps import NCPS, build_layer


def test_build_layer_linear():
    # Setup
    layer = {
        'class': 'tensorflow.keras.layers.Activation',
        'parameters': {
            'activation': 'linear'
        }
    }

    # Run
    built = build_layer(layer, {})

    # Assert
    assert isinstance(built, tf.keras.layers.Activation)


def test_build_layer_bidirectional():
    # Setup
    layer = {
        'class': 'tensorflow.keras.layers.Bidirectional',
        'parameters': {
            'layer': {
                'class': "tensorflow.keras.layers.LSTM",
                'parameters': {
                    'units': 1
                }
            }
        }
    }

    # Run
    built = build_layer(layer, {})

    # Assert
    assert isinstance(built, tf.keras.layers.Bidirectional)


def test_build_layer_ltc():
    # Setup
    layer = {
        'class': 'ncps.tf.LTC',
        'parameters': {
            'units': {
                'class': 'ncps.wirings.AutoNCP',
                'parameters': {
                    'units': 10,
                    'output_size': 1
                }
            }
        }
    }

    # Run
    built = build_layer(layer, {})

    # Assert
    assert isinstance(built, ncps.tf.LTC)


def test__build_model():
    # Setup
    layers = [
        {
            'class': 'tensorflow.keras.layers.Activation',
            'parameters': {
                'activation': 'linear'
            }
        }
    ]
    loss = 'tensorflow.keras.losses.mean_squared_error'
    optimizer = 'tensorflow.keras.optimizers.Adam'

    # Run
    ncps = NCPS(layers, loss, optimizer, None)
    model = ncps._build_model()

    # Assert
    assert isinstance(model, tf.keras.models.Sequential)


@patch('orion.primitives.adapters.ncps.build_layer')
def test_ncps_empty(build_mock):
    # Run
    ncps = NCPS(None, None, None, None)

    # Assert
    assert ncps.layers is None
    assert ncps.loss is None
    assert ncps.optimizer is None
    assert ncps.classification is None
    assert not build_mock.called


@patch('orion.primitives.adapters.ncps.NCPS._build_model')
def test_ncps_linear(build_mock):
    # Setup
    layers = [
        {
            'class': 'tensorflow.keras.layers.Activation',
            'parameters': {
                'activation': 'linear'
            }
        }
    ]
    loss = 'tensorflow.keras.losses.mean_squared_error'
    optimizer = 'tensorflow.keras.optimizers.Adam'
    ncps = NCPS(layers, loss, optimizer, False, batch_size=1)

    # Run
    kwargs = dict()
    X = np.array([
        [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]],
        [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]],
    ])
    y = np.array([
        [5],
        [5]
    ])
    ncps.fit(X, y, **kwargs)

    # Assert
    kwargs == {'input_shape': [4, 1], 'target_dim': 1}
    build_mock.assert_called_once()
