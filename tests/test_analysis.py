import numpy as np
import pandas as pd

from orion import analysis


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
