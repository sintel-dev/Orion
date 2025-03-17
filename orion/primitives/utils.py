# -*- coding: utf-8 -*-

import tensorflow as tf
from mlstars.utils import import_object


def build_layer(layer: dict, hyperparameters: dict):
    layer_class = import_object(layer['class'])
    layer_kwargs = layer['parameters'].copy()
    # TODO: tf.keras.layers.Wrapper deprecated, Bidirectional inheret from Layer
    if issubclass(layer_class, tf.keras.layers.Layer) and 'layer' in layer_kwargs:
        layer_kwargs['layer'] = build_layer(layer_kwargs['layer'], hyperparameters)
    for key, value in layer_kwargs.items():
        if isinstance(value, str):
            layer_kwargs[key] = hyperparameters.get(value, value)
    return layer_class(**layer_kwargs)