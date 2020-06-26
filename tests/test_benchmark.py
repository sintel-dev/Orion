import os
from unittest.mock import ANY, MagicMock, call, patch

import pandas as pd
from mlblocks import MLPipeline

from orion import benchmark
from orion.evaluation import CONTEXTUAL_METRICS as METRICS


def test__get_pipelines_gpu():
    expected_return = {
        "dummy": 'orion/pipelines/dummy.json',
        "tadgan": 'orion/pipelines/tadgan_gpu.json',
        "arima": 'orion/pipelines/arima.json',
        "lstm_dynamic_threshold": 'orion/pipelines/lstm_dynamic_threshold_gpu.json'
    }
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    expected_return = {k: os.path.join(path, v) for k, v in expected_return.items()}

    returned = benchmark._get_pipelines(with_gpu=True)
    assert returned == expected_return


def test__get_pipelines():
    expected_return = {
        "dummy": 'orion/pipelines/dummy.json',
        "tadgan": 'orion/pipelines/tadgan.json',
        "arima": 'orion/pipelines/arima.json',
        "lstm_dynamic_threshold": 'orion/pipelines/lstm_dynamic_threshold.json'
    }
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    expected_return = {k: os.path.join(path, v) for k, v in expected_return.items()}

    returned = benchmark._get_pipelines(with_gpu=False)
    assert returned == expected_return


def test__sort_leaderboard_rank():
    rank = 'f1'
    metrics = METRICS
    score = pd.DataFrame({
        'pipeline': range(5),
        'f1': range(5),
    })

    expected_return = pd.DataFrame({
        'pipeline': range(5)[::-1],
        'rank': range(1, 6),
        'f1': range(5)[::-1],
    })

    returned = benchmark._sort_leaderboard(score, rank, metrics)
    pd.testing.assert_frame_equal(returned, expected_return)


def test__sort_leaderboard_rank_does_not_exist():
    rank = 'does not exist'
    metrics = {'f1': METRICS['f1']}
    score = pd.DataFrame({
        'pipeline': range(5),
        'f1': range(5),
    })

    expected_return = pd.DataFrame({
        'pipeline': range(5)[::-1],
        'rank': range(1, 6),
        'f1': range(5)[::-1],
    })

    returned = benchmark._sort_leaderboard(score, rank, metrics)
    pd.testing.assert_frame_equal(returned, expected_return)


def test__sort_leaderboard_no_rank():
    rank = None
    metrics = METRICS
    score = {k: range(5) for k in metrics.keys()}
    score['pipeline'] = range(5)
    score = pd.DataFrame(score)

    expected_return = score.iloc[::-1].reset_index(drop=True)
    expected_return['rank'] = range(1, 6)

    returned = benchmark._sort_leaderboard(score, rank, metrics)

    assert len(returned.columns) == len(expected_return.columns)
    assert sorted(returned.columns) == sorted(expected_return.columns)
    pd.testing.assert_frame_equal(returned, expected_return[returned.columns])


def test__detrend_signal_trend():
    df = pd.DataFrame({
        'timestamp': range(5),
        'value': range(5)
    })

    expected_return = pd.DataFrame({
        'timestamp': range(5),
        'value': [0.0] * 5,
    })

    returned = benchmark._detrend_signal(df, 'value')
    pd.testing.assert_frame_equal(returned, expected_return)


def test__detrend_signal_no_trend():
    df = pd.DataFrame({
        'timestamp': range(5),
        'value': [0.0] * 5
    })

    expected_return = df.copy()

    returned = benchmark._detrend_signal(df, 'value')
    pd.testing.assert_frame_equal(returned, expected_return)


def test__get_hyperparameter_pipeline():
    hyperparameters = {
        "pipeline1": "pipeline1.json",
        "pipeline2": "pipeline2.json",
    }
    pipeline = "pipeline1"

    expected_return = "pipeline1.json"
    returned = benchmark._get_hyperparameter(hyperparameters, pipeline)
    assert returned == expected_return


def test__get_hyperparameter_dataset():
    hyperparameters = {
        "dataset1": {
            "pipeline1": "pipeline1.json",
            "pipeline2": "pipeline2.json",
        }
    }
    dataset = "dataset1"

    expected_return = {
        "pipeline1": "pipeline1.json",
        "pipeline2": "pipeline2.json",
    }
    returned = benchmark._get_hyperparameter(hyperparameters, dataset)
    assert returned == expected_return


def test__get_hyperparameter_does_not_exist():
    hyperparameters = None
    pipeline = "pipeline1"

    expected_return = None
    returned = benchmark._get_hyperparameter(hyperparameters, pipeline)
    assert returned == expected_return


def test__get_data_none():
    expected_return = benchmark.BENCHMARK_DATA
    returned = benchmark._get_data()

    assert returned == expected_return


def test__get_data_subset():
    subset = ['MSL', 'YAHOOA2']
    expected_return = {k: benchmark.BENCHMARK_DATA[k] for k in subset}
    returned = benchmark._get_data(subset)

    assert returned == expected_return


def test__get_data_list():
    signals = ['own1', 'own2']
    expected_return = signals
    returned = benchmark._get_data(signals)

    assert returned == expected_return


@patch('orion.benchmark.load_anomalies')
@patch('orion.benchmark.analyze')
@patch('orion.benchmark.load_signal')
def test__evaluate_on_signal_holdout(load_signal_mock, analyze_mock, load_anomalies_mock):
    pipeline = MagicMock(autospec=MLPipeline)
    signal = 'signal-name'
    hyper = None
    metric_mock = MagicMock(autospec=METRICS['f1'], return_value=1)
    metrics = {
        'metric-name': metric_mock
    }

    train = MagicMock(autospec=pd.DataFrame)
    test = MagicMock(autospec=pd.DataFrame)
    load_signal_mock.side_effect = [train, test]

    returned = benchmark._evaluate_on_signal(pipeline, signal, hyper, metrics, holdout=True)

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

    analyze_mock.assert_called_once_with(pipeline, train, test, hyper)


@patch('orion.benchmark.load_anomalies')
@patch('orion.benchmark.analyze')
@patch('orion.benchmark.load_signal')
def test__evaluate_on_signal_no_holdout(load_signal_mock, analyze_mock, load_anomalies_mock):
    pipeline = MagicMock(autospec=MLPipeline)
    signal = 'signal-name'
    hyper = None
    metric_mock = MagicMock(autospec=METRICS['f1'], return_value=1)
    metrics = {
        'metric-name': metric_mock
    }

    train = test = MagicMock(autospec=pd.DataFrame)
    load_signal_mock.side_effect = [train, test]

    returned = benchmark._evaluate_on_signal(pipeline, signal, hyper, metrics, holdout=False)

    expected_return = {
        'metric-name': 1,
        'elapsed': ANY
    }
    assert returned == expected_return

    expected_calls = [
        call('signal-name')
    ]
    assert load_signal_mock.call_args_list == expected_calls

    analyze_mock.assert_called_once_with(pipeline, train, test, hyper)


@patch('orion.benchmark.load_anomalies')
@patch('orion.benchmark.analyze')
@patch('orion.benchmark.load_signal')
def test__evaluate_on_signal_no_detrend(load_signal_mock, analyze_mock, load_anomalies_mock):
    pipeline = MagicMock(autospec=MLPipeline)
    signal = 'signal-name'
    hyper = None
    metric_mock = MagicMock(autospec=METRICS['f1'], return_value=1)
    metrics = {
        'metric-name': metric_mock
    }

    train = MagicMock(autospec=pd.DataFrame)
    test = MagicMock(autospec=pd.DataFrame)
    load_signal_mock.side_effect = [train, test]

    returned = benchmark._evaluate_on_signal(pipeline, signal, hyper, metrics, detrend=False)

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

    analyze_mock.assert_called_once_with(pipeline, train, test, hyper)


@patch('orion.benchmark.load_anomalies')
@patch('orion.benchmark.analyze')
@patch('orion.benchmark.load_signal')
@patch('orion.benchmark._detrend_signal')
def test__evaluate_on_signal_detrend(
        detrend_signal_mock, load_signal_mock, analyze_mock, load_anomalies_mock):
    pipeline = MagicMock(autospec=MLPipeline)
    signal = 'signal-name'
    hyper = None
    metric_mock = MagicMock(autospec=METRICS['f1'], return_value=1)
    metrics = {
        'metric-name': metric_mock
    }

    train = MagicMock(autospec=pd.DataFrame)
    test = MagicMock(autospec=pd.DataFrame)
    load_signal_mock.side_effect = [train, test]
    detrend_signal_mock.side_effect = [train, test]

    returned = benchmark._evaluate_on_signal(pipeline, signal, hyper, metrics, detrend=True)

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

    expected_calls = [
        call(train, 'value'),
        call(test, 'value'),
    ]
    assert detrend_signal_mock.call_args_list == expected_calls

    analyze_mock.assert_called_once_with(pipeline, train, test, hyper)


@patch('orion.benchmark._evaluate_on_signal')
def test_evaluate_pipeline(evaluate_on_signal_mock):
    pipeline = MagicMock(autospec=MLPipeline)
    signal = 'signal-name'
    hyper = MagicMock(autospec=dict)
    metric_mock = MagicMock(autospec=METRICS['f1'], return_value=1)
    metrics = {
        'metric-name': metric_mock
    }

    score = {
        'metric-name': 1,
        'elapsed': 1.0
    }
    evaluate_on_signal_mock.return_value = score

    benchmark.evaluate_pipeline(pipeline, [signal], hyper, metrics)

    expected_calls = [
        call(pipeline, signal, hyper, metrics, False, True),
        call(pipeline, signal, hyper, metrics, False, False),
    ]
    assert evaluate_on_signal_mock.call_args_list == expected_calls


@patch('orion.benchmark._evaluate_on_signal')
def test_evaluate_pipeline_holdout(evaluate_on_signal_mock):
    pipeline = MagicMock(autospec=MLPipeline)
    signal = 'signal-name'
    hyper = MagicMock(autospec=dict)
    metric_mock = MagicMock(autospec=METRICS['f1'], return_value=1)
    metrics = {
        'metric-name': metric_mock
    }

    score = {
        'metric-name': 1,
        'elapsed': 1.0
    }
    evaluate_on_signal_mock.return_value = score

    expected_return = pd.DataFrame({
        'holdout': [True],
        'metric-name': [1],
        'elapsed': [1.0]

    })
    returned = benchmark.evaluate_pipeline(pipeline, [signal], hyper, metrics, holdout=True)
    pd.testing.assert_frame_equal(returned, expected_return)

    evaluate_on_signal_mock.assert_called_once_with(pipeline, signal, hyper, metrics, False, True)


@patch('orion.benchmark._evaluate_on_signal')
def test_evaluate_pipeline_no_holdout(evaluate_on_signal_mock):
    pipeline = MagicMock(autospec=MLPipeline)
    signal = 'signal-name'
    hyper = MagicMock(autospec=dict)
    metric_mock = MagicMock(autospec=METRICS['f1'], return_value=1)
    metrics = {
        'metric-name': metric_mock
    }

    score = {
        'metric-name': 1,
        'elapsed': 1.0
    }
    evaluate_on_signal_mock.return_value = score

    expected_return = pd.DataFrame({
        'holdout': [False],
        'metric-name': [1],
        'elapsed': [1.0]
    })
    returned = benchmark.evaluate_pipeline(pipeline, [signal], hyper, metrics, holdout=False)
    pd.testing.assert_frame_equal(returned, expected_return)

    evaluate_on_signal_mock.assert_called_once_with(pipeline, signal, hyper, metrics, False, False)


@patch('orion.benchmark.evaluate_pipeline')
def test_evaluate_pipelines(evaluate_pipeline_mock):
    pipeline = 'pipeline'
    signal = 'signal-name'
    hyper = None
    metric_mock = MagicMock(autospec=METRICS['f1'], return_value=1)
    metrics = {
        'metric-name': metric_mock
    }

    score = pd.DataFrame({
        'holdout': [True, False],
        'metric-name': [1, 1],
        'elapsed': [1.0, 1.0]
    })
    evaluate_pipeline_mock.return_value = score

    expected_return = pd.DataFrame({
        'pipeline': [pipeline] * 2,
        'rank': range(1, 3),
        'holdout': [True, False],
        'metric-name': [1] * 2,
        'elapsed': [1.0] * 2,
        'hyperparameter': [hyper] * 2
    })
    returned = benchmark.evaluate_pipelines([pipeline], [signal], hyper, metrics)
    pd.testing.assert_frame_equal(returned, expected_return)

    evaluate_pipeline_mock.assert_called_once_with(
        pipeline, [signal], hyper, metrics, detrend=False, holdout=(True, False))


@patch('orion.benchmark.evaluate_pipelines')
def test_run_benchmark(evaluate_pipelines_mock):
    pipeline = ['pipeline']
    signal = ['signal-name']
    datasets = {'dataset-name': signal}
    hyper = None
    metric_mock = MagicMock(autospec=METRICS['f1'], return_value=1)
    metrics = {
        'metric-name': metric_mock
    }

    score = pd.DataFrame({
        'pipeline': pipeline * 2,
        'rank': range(1, 3),
        'holdout': [True, False],
        'metric-name': [1] * 2,
        'elapsed': [1.0] * 2,
        'hyperparameter': [hyper] * 2
    })
    evaluate_pipelines_mock.return_value = score

    expected_return = pd.DataFrame({
        'pipeline': pipeline,
        'rank': 1,
        'metric-name': [1],
        'elapsed': [1.0],
    })
    returned = benchmark.run_benchmark(pipeline, datasets, hyper, metrics)
    pd.testing.assert_frame_equal(returned, expected_return)

    evaluate_pipelines_mock.assert_called_once_with(
        pipeline, signal, hyper, metrics, 'f1', detrend=False, holdout=(True, False))
