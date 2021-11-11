# -*- coding: utf-8 -*-

encoder = [
    {
        "class": "keras.layers.Bidirectional",
        "parameters": {
            "layer": {
                "class": "keras.layers.LSTM",
                "parameters": {
                    "units": 100,
                    "return_sequences": True
                }
            }
        }
    },
    {
        "class": "keras.layers.Flatten",
        "parameters": {}
    },
    {
        "class": "keras.layers.Dense",
        "parameters": {
            "units": 20
        }
    },
    {
        "class": "keras.layers.Reshape",
        "parameters": {
            "target_shape": "encoder_reshape_shape"
        }
    }
]

generator = [
    {
        "class": "keras.layers.Flatten",
        "parameters": {}
    },
    {
        "class": "keras.layers.Dense",
        "parameters": {
            "units": 50
        }
    },
    {
        "class": "keras.layers.Reshape",
        "parameters": {
            "target_shape": "generator_reshape_shape"
        }
    },
    {
        "class": "keras.layers.Bidirectional",
        "parameters": {
            "layer": {
                "class": "keras.layers.LSTM",
                "parameters": {
                    "units": 64,
                    "return_sequences": True,
                    "dropout": 0.2,
                    "recurrent_dropout": 0.2
                }
            },
            "merge_mode": "concat"
        }
    },
    {
        "class": "keras.layers.convolutional.UpSampling1D",
        "parameters": {
            "size": 2
        }
    },
    {
        "class": "keras.layers.Bidirectional",
        "parameters": {
            "layer": {
            "class": "keras.layers.LSTM",
            "parameters": {
                "units": 64,
                "return_sequences": True,
                "dropout": 0.2,
                "recurrent_dropout": 0.2
                }
            },
            "merge_mode": "concat"
        }
    },
    {
        "class": "keras.layers.TimeDistributed",
        "parameters": {
            "layer": {
                "class": "keras.layers.Dense",
                "parameters": {
                    "units": 1
                }
            }
        }
    },
    {
        "class": "keras.layers.Activation",
        "parameters": {
            "activation": "tanh"
        }
    }
]

criticX = [
    {
        "class": "keras.layers.Conv1D",
        "parameters": {
            "filters": 64,
            "kernel_size": 5
        }
    },
    {
        "class": "keras.layers.advanced_activations.LeakyReLU",
        "parameters": {
            "alpha": 0.2
        }
    },
    {
        "class": "keras.layers.Dropout",
        "parameters": {
            "rate": 0.25
        }
    },
    {
        "class": "keras.layers.Conv1D",
        "parameters": {
            "filters": 64,
            "kernel_size": 5
        }
    },
    {
        "class": "keras.layers.advanced_activations.LeakyReLU",
        "parameters": {
            "alpha": 0.2
        }
    },
    {
        "class": "keras.layers.Dropout",
        "parameters": {
            "rate": 0.25
        }
    },
    {
    "class": "keras.layers.Conv1D",
        "parameters": {
            "filters": 64,
            "kernel_size": 5
        }
    },
    {
        "class": "keras.layers.advanced_activations.LeakyReLU",
        "parameters": {
            "alpha": 0.2
        }
    },
    {
        "class": "keras.layers.Dropout",
        "parameters": {
            "rate": 0.25
        }
    },
    {
        "class": "keras.layers.Conv1D",
        "parameters": {
            "filters": 64,
            "kernel_size": 5
        }
    },
    {
        "class": "keras.layers.advanced_activations.LeakyReLU",
        "parameters": {
            "alpha": 0.2
        }
    },
    {
        "class": "keras.layers.Dropout",
        "parameters": {
            "rate": 0.25
    }
    },
    {
        "class": "keras.layers.Flatten",
        "parameters": {}
    },
    {
        "class": "keras.layers.Dense",
        "parameters": {
            "units": 1
        }
    }
]

criticZ = [
    {
        "class": "keras.layers.Flatten",
        "parameters": {}
    },
    {
        "class": "keras.layers.Dense",
        "parameters": {
            "units": 100
        }
    },
    {
        "class": "keras.layers.advanced_activations.LeakyReLU",
        "parameters": {
            "alpha": 0.2
        }
    },
    {
        "class": "keras.layers.Dropout",
        "parameters": {
            "rate": 0.2
        }
    },
    {
        "class": "keras.layers.Dense",
        "parameters": {
            "units": 100
        }
    },
    {
        "class": "keras.layers.advanced_activations.LeakyReLU",
        "parameters": {
            "alpha": 0.2
        }
    },
    {
        "class": "keras.layers.Dropout",
        "parameters": {
            "rate": 0.2
        }
    },
    {
        "class": "keras.layers.Dense",
        "parameters": {
            "units": 1
        }
    }
]

hyperparameters = {
    "epochs": 35,
    "input_shape": (100, 1),
    "target_shape": (100, 1),
    "optimizer": "keras.optimizers.Adam",
    "learning_rate": 0.0005,
    "latent_dim": 20,
    "batch_size": 64,
    "n_critic": 5,
    "encoder_reshape_shape": None,
    "generator_reshape_shape": None,
    "generator_reshape_dim": None,
    "layers_encoder": encoder,
    "layers_generator": generator,
    "layers_critic_x": criticX,
    "layers_critic_z": criticZ
}