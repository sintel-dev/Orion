# -*- coding: utf-8 -*-

encoder = [
    {
        "class": "tensorflow.keras.layers.LSTM",
        "parameters": {
            "units": 100,
            "return_sequences": True,
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "use_bias": True,
            "recurrent_dropout": 0.0,
            "unroll": False
        }
    },
    {
        "class": "tensorflow.keras.layers.Flatten",
        "parameters": {}
    },
    {
        "class": "tensorflow.keras.layers.Dense",
        "parameters": {
            "units": 20
        }
    },
    {
        "class": "tensorflow.keras.layers.Reshape",
        "parameters": {
            "target_shape": "encoder_reshape_shape"
        }
    }
]

generator = [
    {
        "class": "tensorflow.keras.layers.Flatten",
        "parameters": {}
    },
    {
        "class": "tensorflow.keras.layers.Dense",
        "parameters": {
            "units": 50
        }
    },
    {
        "class": "tensorflow.keras.layers.Reshape",
        "parameters": {
            "target_shape": "generator_reshape_shape"
        }
    },
    {
        "class": "tensorflow.keras.layers.Bidirectional",
        "parameters": {
            "layer": {
                "class": "tensorflow.keras.layers.LSTM",
                "parameters": {
                    "units": 64,
                    "return_sequences": True,
                    "dropout": 0.2,
                    "activation": "tanh",
                    "recurrent_activation": "sigmoid",
                    "use_bias": True,
                    "recurrent_dropout": 0.0,
                    "unroll": False
                }
            },
            "merge_mode": "concat"
        }
    },
    {
        "class": "tensorflow.keras.layers.UpSampling1D",
        "parameters": {
            "size": 2
        }
    },
    {
        "class": "tensorflow.keras.layers.Bidirectional",
        "parameters": {
            "layer": {
                "class": "tensorflow.keras.layers.LSTM",
                "parameters": {
                    "units": 64,
                    "return_sequences": True,
                    "dropout": 0.2,
                    "activation": "tanh",
                    "recurrent_activation": "sigmoid",
                    "use_bias": True,
                    "recurrent_dropout": 0.0,
                    "unroll": True
                }
            },
            "merge_mode": "concat"
        }
    },
    {
        "class": "tensorflow.keras.layers.TimeDistributed",
        "parameters": {
            "layer": {
                "class": "tensorflow.keras.layers.Dense",
                "parameters": {
                    "units": 1
                }
            }
        }
    },
    {
        "class": "tensorflow.keras.layers.Activation",
        "parameters": {
            "activation": "tanh"
        }
    }
]

criticX = [
    {
        "class": "tensorflow.keras.layers.Conv1D",
        "parameters": {
            "filters": 64,
            "kernel_size": 5
        }
    },
    {
        "class": "tensorflow.keras.layers.LeakyReLU",
        "parameters": {
            "alpha": 0.2
        }
    },
    {
        "class": "tensorflow.keras.layers.Dropout",
        "parameters": {
            "rate": 0.25
        }
    },
    {
        "class": "tensorflow.keras.layers.Conv1D",
        "parameters": {
            "filters": 64,
            "kernel_size": 5
        }
    },
    {
        "class": "tensorflow.keras.layers.LeakyReLU",
        "parameters": {
            "alpha": 0.2
        }
    },
    {
        "class": "tensorflow.keras.layers.Dropout",
        "parameters": {
            "rate": 0.25
        }
    },
    {
    "class": "tensorflow.keras.layers.Conv1D",
        "parameters": {
            "filters": 64,
            "kernel_size": 5
        }
    },
    {
        "class": "tensorflow.keras.layers.LeakyReLU",
        "parameters": {
            "alpha": 0.2
        }
    },
    {
        "class": "tensorflow.keras.layers.Dropout",
        "parameters": {
            "rate": 0.25
        }
    },
    {
        "class": "tensorflow.keras.layers.Conv1D",
        "parameters": {
            "filters": 64,
            "kernel_size": 5
        }
    },
    {
        "class": "tensorflow.keras.layers.LeakyReLU",
        "parameters": {
            "alpha": 0.2
        }
    },
    {
        "class": "tensorflow.keras.layers.Dropout",
        "parameters": {
            "rate": 0.25
    }
    },
    {
        "class": "tensorflow.keras.layers.Flatten",
        "parameters": {}
    },
    {
        "class": "tensorflow.keras.layers.Dense",
        "parameters": {
            "units": 1
        }
    }
]

criticZ = [
    {
        "class": "tensorflow.keras.layers.Flatten",
        "parameters": {}
    },
    {
        "class": "tensorflow.keras.layers.Dense",
        "parameters": {
            "units": 100
        }
    },
    {
        "class": "tensorflow.keras.layers.LeakyReLU",
        "parameters": {
            "alpha": 0.2
        }
    },
    {
        "class": "tensorflow.keras.layers.Dropout",
        "parameters": {
            "rate": 0.2
        }
    },
    {
        "class": "tensorflow.keras.layers.Dense",
        "parameters": {
            "units": 100
        }
    },
    {
        "class": "tensorflow.keras.layers.LeakyReLU",
        "parameters": {
            "alpha": 0.2
        }
    },
    {
        "class": "tensorflow.keras.layers.Dropout",
        "parameters": {
            "rate": 0.2
        }
    },
    {
        "class": "tensorflow.keras.layers.Dense",
        "parameters": {
            "units": 1
        }
    }
]

hyperparameters = {
    "epochs": 35,
    "input_shape": (100, 1),
    "target_shape": (100, 1),
    "optimizer": "tensorflow.keras.optimizers.Adam",
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
