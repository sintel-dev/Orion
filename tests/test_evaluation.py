from unittest.mock import ANY, MagicMock, call, patch

import pandas as pd
from mlblocks import MLPipeline

from orion import evaluation
from orion.metrics import f1_score


@patch('orion.evaluation.load_anomalies')
@patch('orion.evaluation.analyze')
@patch('orion.evaluation.load_signal')
def test__evaluate_on_signal_holdout(load_signal_mock, analize_mock, load_anomalies_mock):
    pipeline = MagicMock(autospec=MLPipeline)
    signal = 'signal-name'
    metric_mock = MagicMock(autospec=f1_score, return_value=1)
    metrics = {
        'metric-name': metric_mock
    }

    train = MagicMock(autospec=pd.DataFrame)
    test = MagicMock(autospec=pd.DataFrame)
    load_signal_mock.side_effect = [train, test]

    returned = evaluation._evaluate_on_signal(pipeline, signal, metrics, holdout=True)

    expected_return = {
        'metric-name': 1,
        'elapsed': ANY
    }
    assert returned == expected_return

    expected_calls = [
        call('signal-name-train'),
        call('signal-name-test'),
    ]
    assert load_signal_mock.call_args_list == expected_calls

    analize_mock.assert_called_once_with(pipeline, train, test)


@patch('orion.evaluation.load_anomalies')
@patch('orion.evaluation.analyze')
@patch('orion.evaluation.load_signal')
def test__evaluate_on_signal_no_holdout(load_signal_mock, analize_mock, load_anomalies_mock):
    pipeline = MagicMock(autospec=MLPipeline)
    signal = 'signal-name'
    metric_mock = MagicMock(autospec=f1_score, return_value=1)
    metrics = {
        'metric-name': metric_mock
    }

    train = MagicMock(autospec=pd.DataFrame)
    test = MagicMock(autospec=pd.DataFrame)
    load_signal_mock.side_effect = [train, test]

    returned = evaluation._evaluate_on_signal(pipeline, signal, metrics, holdout=False)

    expected_return = {
        'metric-name': 1,
        'elapsed': ANY
    }
    assert returned == expected_return

    expected_calls = [
        call('signal-name'),
        call('signal-name-test'),
    ]
    assert load_signal_mock.call_args_list == expected_calls

    analize_mock.assert_called_once_with(pipeline, train, test)
