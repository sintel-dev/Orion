# -*- coding: utf-8 -*-

import tensorflow as tf
from mlstars.adapters.keras import Sequential
from mlstars.utils import import_object


def build_layer(layer, hyperparameters):
    layer_class = import_object(layer['class'])
    layer_kwargs = layer['parameters'].copy()
    if issubclass(layer_class, tf.keras.layers.Wrapper):
        layer_kwargs['layer'] = build_layer(layer_kwargs['layer'], hyperparameters)
    elif issubclass(layer_class, tf.keras.layers.RNN) and isinstance(layer_kwargs['units'], dict):
        layer_kwargs['units'] = build_layer(layer_kwargs['units'], hyperparameters)
    for key, value in layer_kwargs.items():
        if isinstance(value, str):
            layer_kwargs[key] = hyperparameters.get(value, value)
    return layer_class(**layer_kwargs)


class NCPS(Sequential):
    def _build_model(self, **kwargs):
        hyperparameters = self.hyperparameters.copy()
        hyperparameters.update(kwargs)

        model = tf.keras.models.Sequential()

        for layer in self.layers:
            built_layer = build_layer(layer, hyperparameters)
            model.add(built_layer)

        model.compile(loss=self.loss, optimizer=self.optimizer(), metrics=self.metrics)
        return model
