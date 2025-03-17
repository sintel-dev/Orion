import numpy as np
import pandas as pd
import pytest
from mlblocks import MLPipeline

from orion import analysis


@pytest.fixture
def tadgan_hyperparameters():
    return {
        "mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1": {
            "interval": 1,
            "time_column": "timestamp",
        },
        "mlstars.custom.timeseries_preprocessing.rolling_window_sequences#1": {
            "target_column": 0,
            "window_size": 100,
            "target_size": 1
        },
        'orion.primitives.tadgan.TadGAN#1': {
            'epochs': 2,
            'verbose': False
        }
    }


@pytest.fixture
def tadgan_pipline(tadgan_hyperparameters):
    pipeline_path = 'tadgan'
    pipline = analysis._load_pipeline(pipeline_path, tadgan_hyperparameters)
    return pipline


def test__build_events_df_empty():
    events = np.array([], dtype=np.float64)

    returned = analysis._build_events_df(events)

    assert returned.empty
    assert list(returned.columns) == ['start', 'end', 'score']


def test__build_events_df_events():
    events = np.array([
        [1.22297040e+09, 1.22299200e+09, 5.72643599e-01],
        [1.22301360e+09, 1.22303520e+09, 5.72643599e-01],
        [1.22351040e+09, 1.22353200e+09, 5.72643599e-01],
    ])

    returned = analysis._build_events_df(events)

    expected = pd.DataFrame({
        'start': [1222970400, 1223013600, 1223510400],
        'end': [1222992000, 1223035200, 1223532000],
        'score': [0.572644, 0.572644, 0.572644]
    })
    pd.testing.assert_frame_equal(returned, expected)


def test__load_pipeline_tadgan(tadgan_pipline, tadgan_hyperparameters):
    hyperset = tadgan_pipline.get_hyperparameters()
    returned = list()
    for k, v in tadgan_hyperparameters.items():
        returned.extend([item in hyperset[k].items() for item in v.items()])

    assert isinstance(tadgan_pipline, MLPipeline)
    assert all(returned)
