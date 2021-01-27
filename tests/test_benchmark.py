from unittest import TestCase
from unittest.mock import ANY, Mock, call, patch

import pandas as pd
from mlblocks import MLPipeline

from orion import benchmark
from orion.evaluation import CONTEXTUAL_METRICS as METRICS
from orion.evaluation import contextual_confusion_matrix


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


def test__get_parameter_pipeline():
    hyperparameters = {
        "pipeline1": "pipeline1.json",
        "pipeline2": "pipeline2.json",
    }
    pipeline = "pipeline1"

    expected_return = "pipeline1.json"
    returned = benchmark._get_parameter(hyperparameters, pipeline)
    assert returned == expected_return


def test__get_parameter_dataset():
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
    returned = benchmark._get_parameter(hyperparameters, dataset)
    assert returned == expected_return


def test__get_parameter_does_not_exist():
    hyperparameters = None
    pipeline = "pipeline1"

    expected_return = None
    returned = benchmark._get_parameter(hyperparameters, pipeline)
    assert returned == expected_return


@patch('orion.benchmark.load_signal')
def test__load_signal_test_split_true(load_signal_mock):
    train = Mock(autospec=pd.DataFrame)
    test = Mock(autospec=pd.DataFrame)

    load_signal_mock.return_value = (train, test)

    test_split = True
    returned = benchmark._load_signal('signal-name', test_split)

    assert isinstance(returned, tuple)
    assert len(returned) == 2

    expected_calls = [
        call('signal-name-train'),
        call('signal-name-test')
    ]
    assert load_signal_mock.call_args_list == expected_calls


@patch('orion.benchmark.load_signal')
def test__load_signal_test_split_false(load_signal_mock):
    df = pd.DataFrame({
        'timestamp': list(range(10)),
        'value': list(range(10, 20))
    })
    load_signal_mock.return_value = df

    test_split = False
    returned = benchmark._load_signal('signal-name', test_split)

    assert isinstance(returned, tuple)
    assert len(returned) == 2

    train, test = returned
    pd.testing.assert_frame_equal(train, test)

    expected_calls = [
        call('signal-name'),
    ]
    assert load_signal_mock.call_args_list == expected_calls


@patch('orion.benchmark.load_signal')
def test__load_signal_test_split_float(load_signal_mock):
    train = Mock(autospec=pd.DataFrame)
    test = Mock(autospec=pd.DataFrame)

    load_signal_mock.return_value = (train, test)

    test_split = 0.2
    returned = benchmark._load_signal('signal-name', test_split)

    assert isinstance(returned, tuple)
    assert len(returned) == 2

    expected_calls = [
        call('signal-name', test_size=test_split),
    ]
    assert load_signal_mock.call_args_list == expected_calls


class TestBenchmark(TestCase):

    @classmethod
    def setup_class(cls):
        cls.pipeline = Mock(autospec=MLPipeline)
        cls.name = 'pipeline-name'
        cls.dataset = 'dataset-name'
        cls.signal = 'signal-name'
        cls.hyper = None
        cls.distributed = False
        cls.rank = 'metric-name'
        cls.metrics = {
            'metric-name': Mock(autospec=METRICS['f1'], return_value=1)
        }

    def set_score(self, metric, elapsed, test_split):
        return {
            'metric-name': metric,
            'elapsed': elapsed,
            'pipeline': self.name,
            'split': test_split,
            'dataset': self.dataset,
            'signal': self.signal,
            'status': 'OK'
        }

    @patch('orion.benchmark.load_anomalies')
    @patch('orion.benchmark.analyze')
    @patch('orion.benchmark._load_pipeline')
    @patch('orion.benchmark.load_signal')
    def test__evaluate_signal(
            self, load_signal_mock, load_pipeline_mock, analyze_mock, load_anomalies_mock):
        train = Mock(autospec=pd.DataFrame)
        test = Mock(autospec=pd.DataFrame)
        load_signal_mock.side_effect = [train, test]
        load_pipeline_mock.return_value = self.pipeline

        anomalies = Mock(autospec=pd.DataFrame)
        analyze_mock.return_value = anomalies

        returned = benchmark._evaluate_signal(
            self.pipeline, self.name, self.dataset, self.signal, self.hyper, self.metrics, True)

        expected_return = self.set_score(1, ANY, ANY)
        assert returned == expected_return

        expected_calls = [
            call('signal-name-train'),
            call('signal-name-test')
        ]
        assert load_signal_mock.call_args_list == expected_calls

        load_pipeline_mock.assert_called_once_with(self.pipeline, self.hyper)
        analyze_mock.assert_called_once_with(self.pipeline, train, test)
        load_anomalies_mock.assert_called_once_with(self.signal)

    @patch('orion.benchmark.load_anomalies')
    @patch('orion.benchmark.analyze')
    @patch('orion.benchmark._load_pipeline')
    @patch('orion.benchmark.load_signal')
    def test__evaluate_signal_exception(
            self, load_signal_mock, load_pipeline_mock, analyze_mock, load_anomalies_mock):
        train = Mock(autospec=pd.DataFrame)
        test = Mock(autospec=pd.DataFrame)
        load_signal_mock.side_effect = [train, test]
        load_pipeline_mock.return_value = self.pipeline

        analyze_mock.side_effect = Exception("failed analyze.")

        returned = benchmark._evaluate_signal(
            self.pipeline, self.name, self.dataset, self.signal, self.hyper, self.metrics, True)

        expected_return = self.set_score(0, ANY, ANY)
        expected_return['status'] = 'ERROR'
        assert returned == expected_return

        expected_calls = [
            call('signal-name-train'),
            call('signal-name-test')
        ]
        assert load_signal_mock.call_args_list == expected_calls

        load_pipeline_mock.assert_called_once_with(self.pipeline, self.hyper)
        analyze_mock.assert_called_once_with(self.pipeline, train, test)

        assert load_anomalies_mock.called

    @patch('orion.benchmark.load_anomalies')
    @patch('orion.benchmark.analyze')
    @patch('orion.benchmark._load_pipeline')
    @patch('orion.benchmark.load_signal')
    def test__evaluate_signal_exception_confusion_matrix(
            self, load_signal_mock, load_pipeline_mock, analyze_mock, load_anomalies_mock):
        anomalies = pd.DataFrame({
            'start': [10, 35],
            'end': [20, 40]
        })

        train = Mock(autospec=pd.DataFrame)
        test = Mock(autospec=pd.DataFrame)
        load_signal_mock.side_effect = [train, test]
        load_pipeline_mock.return_value = self.pipeline
        load_anomalies_mock.return_value = anomalies
        analyze_mock.side_effect = Exception("failed analyze.")

        metrics = {'confusion_matrix': Mock(autospec=contextual_confusion_matrix)}
        metrics = {**metrics, **self.metrics}
        returned = benchmark._evaluate_signal(
            self.pipeline, self.name, self.dataset, self.signal, self.hyper, metrics, True)

        expected_return = self.set_score(0, ANY, ANY)
        expected_return['status'] = 'ERROR'
        expected_return['confusion_matrix'] = (None, 0, 2, 0)

        assert returned == expected_return

    @patch('orion.benchmark.load_anomalies')
    @patch('orion.benchmark.analyze')
    @patch('orion.benchmark._load_pipeline')
    @patch('orion.benchmark.load_signal')
    def test__evaluate_signal_test_split(
            self, load_signal_mock, load_pipeline_mock, analyze_mock, load_anomalies_mock):
        train = Mock(autospec=pd.DataFrame)
        test = Mock(autospec=pd.DataFrame)
        load_signal_mock.side_effect = [train, test]
        load_pipeline_mock.return_value = self.pipeline

        test_split = True

        returned = benchmark._evaluate_signal(
            self.pipeline, self.name, self.dataset, self.signal, self.hyper, self.metrics,
            test_split=test_split)

        expected_return = self.set_score(1, ANY, test_split)
        assert returned == expected_return

        expected_calls = [
            call('signal-name-train'),
            call('signal-name-test')
        ]
        assert load_signal_mock.call_args_list == expected_calls

        load_pipeline_mock.assert_called_once_with(self.pipeline, self.hyper)
        analyze_mock.assert_called_once_with(self.pipeline, train, test)
        load_anomalies_mock.assert_called_once_with(self.signal)

    @patch('orion.benchmark.load_anomalies')
    @patch('orion.benchmark.analyze')
    @patch('orion.benchmark._load_pipeline')
    @patch('orion.benchmark.load_signal')
    def test__evaluate_signal_no_test_split(
            self, load_signal_mock, load_pipeline_mock, analyze_mock, load_anomalies_mock):
        train = test = Mock(autospec=pd.DataFrame)
        load_signal_mock.side_effect = [train, test]
        load_pipeline_mock.return_value = self.pipeline

        test_split = False

        returned = benchmark._evaluate_signal(
            self.pipeline, self.name, self.dataset, self.signal, self.hyper, self.metrics,
            test_split=test_split)

        expected_return = self.set_score(1, ANY, test_split)
        assert returned == expected_return

        expected_calls = [
            call('signal-name')
        ]
        assert load_signal_mock.call_args_list == expected_calls

        load_pipeline_mock.assert_called_once_with(self.pipeline, self.hyper)
        analyze_mock.assert_called_once_with(self.pipeline, train, test)
        load_anomalies_mock.assert_called_once_with(self.signal)

    @patch('orion.benchmark.load_anomalies')
    @patch('orion.benchmark.analyze')
    @patch('orion.benchmark._load_pipeline')
    @patch('orion.benchmark.load_signal')
    def test__evaluate_signal_no_detrend(
            self, load_signal_mock, load_pipeline_mock, analyze_mock, load_anomalies_mock):
        train = Mock(autospec=pd.DataFrame)
        test = Mock(autospec=pd.DataFrame)
        load_signal_mock.side_effect = [train, test]
        load_pipeline_mock.return_value = self.pipeline

        detrend = False

        returned = benchmark._evaluate_signal(
            self.pipeline, self.name, self.dataset, self.signal, self.hyper, self.metrics,
            test_split=True, detrend=detrend)

        expected_return = self.set_score(1, ANY, ANY)
        assert returned == expected_return

        expected_calls = [
            call('signal-name-train'),
            call('signal-name-test')
        ]
        assert load_signal_mock.call_args_list == expected_calls

        load_pipeline_mock.assert_called_once_with(self.pipeline, self.hyper)
        analyze_mock.assert_called_once_with(self.pipeline, train, test)
        load_anomalies_mock.assert_called_once_with(self.signal)

    @patch('orion.benchmark.load_anomalies')
    @patch('orion.benchmark.analyze')
    @patch('orion.benchmark._load_pipeline')
    @patch('orion.benchmark.load_signal')
    @patch('orion.benchmark._detrend_signal')
    def test__evaluate_signal_detrend(self, detrend_signal_mock, load_signal_mock,
                                      load_pipeline_mock, analyze_mock, load_anomalies_mock):

        train = Mock(autospec=pd.DataFrame)
        test = Mock(autospec=pd.DataFrame)
        detrend_signal_mock.side_effect = [train, test]
        load_signal_mock.side_effect = [train, test]
        load_pipeline_mock.return_value = self.pipeline

        detrend = True

        returned = benchmark._evaluate_signal(
            self.pipeline, self.name, self.dataset, self.signal, self.hyper, self.metrics,
            test_split=True, detrend=detrend)

        expected_return = self.set_score(1, ANY, ANY)
        assert returned == expected_return

        expected_calls = [
            call('signal-name-train'),
            call('signal-name-test')
        ]
        assert load_signal_mock.call_args_list == expected_calls

        expected_calls = [
            call(train, 'value'),
            call(test, 'value')
        ]
        assert detrend_signal_mock.call_args_list == expected_calls

        load_pipeline_mock.assert_called_once_with(self.pipeline, self.hyper)
        analyze_mock.assert_called_once_with(self.pipeline, train, test)
        load_anomalies_mock.assert_called_once_with(self.signal)

    @patch('orion.benchmark._evaluate_signal')
    def test__evaluate_pipeline(self, evaluate_signal_mock):
        test_split = (True, False)
        detrend = False

        signals = [self.signal]

        score = self.set_score(1, ANY, ANY)
        evaluate_signal_mock.return_value = score

        benchmark._evaluate_pipeline(
            self.pipeline, self.name, self.dataset, signals, self.hyper, self.metrics,
            self.distributed, test_split, detrend)

        expected_calls = [
            call(self.pipeline, self.name, self.dataset, self.signal,
                 self.hyper, self.metrics, True, detrend),

            call(self.pipeline, self.name, self.dataset, self.signal,
                 self.hyper, self.metrics, False, detrend)
        ]
        assert evaluate_signal_mock.call_args_list == expected_calls

    @patch('orion.benchmark._evaluate_signal')
    def test__evaluate_pipeline_test_split_none(self, evaluate_signal_mock):
        test_split = None
        detrend = False

        signals = [self.signal]

        score = self.set_score(1, ANY, ANY)
        evaluate_signal_mock.return_value = score

        returned = benchmark._evaluate_pipeline(
            self.pipeline, self.name, self.dataset, signals, self.hyper, self.metrics,
            self.distributed, test_split, detrend)

        expected_return = [
            self.set_score(1, ANY, True),
            self.set_score(1, ANY, False)
        ]
        assert returned == expected_return

        expected_calls = [
            call(self.pipeline, self.name, self.dataset, self.signal,
                 self.hyper, self.metrics, True, detrend),

            call(self.pipeline, self.name, self.dataset, self.signal,
                 self.hyper, self.metrics, False, detrend)
        ]
        assert evaluate_signal_mock.call_args_list == expected_calls

    @patch('orion.benchmark._evaluate_signal')
    def test__evaluate_pipeline_test_split(self, evaluate_signal_mock):
        test_split = True
        detrend = False

        signals = [self.signal]

        score = self.set_score(1, ANY, test_split)
        evaluate_signal_mock.return_value = score

        expected_return = [score]
        returned = benchmark._evaluate_pipeline(
            self.pipeline, self.name, self.dataset, signals, self.hyper, self.metrics,
            self.distributed, test_split, detrend)

        assert returned == expected_return

        evaluate_signal_mock.assert_called_once_with(
            self.pipeline, self.name, self.dataset, self.signal, self.hyper, self.metrics,
            test_split, detrend)

    @patch('orion.benchmark._evaluate_signal')
    def test__evaluate_pipeline_no_test_split(self, evaluate_signal_mock):
        test_split = False
        detrend = False

        signals = [self.signal]

        score = self.set_score(1, ANY, test_split)
        evaluate_signal_mock.return_value = score

        expected_return = [score]
        returned = benchmark._evaluate_pipeline(
            self.pipeline, self.name, self.dataset, signals, self.hyper, self.metrics,
            self.distributed, test_split, detrend)

        assert returned == expected_return

        evaluate_signal_mock.assert_called_once_with(
            self.pipeline, self.name, self.dataset, self.signal, self.hyper, self.metrics,
            test_split, detrend)

    @patch('orion.benchmark._evaluate_pipeline')
    def test__evaluate_pipelines(self, evaluate_pipeline_mock):
        test_split = False
        detrend = False

        signals = [self.signal]
        pipelines = {self.name: self.pipeline}

        score = self.set_score(1, ANY, test_split)
        evaluate_pipeline_mock.return_value = [score]

        expected_return = [score]
        returned = benchmark._evaluate_pipelines(pipelines, self.dataset, signals, self.hyper,
                                                 self.metrics, self.distributed, test_split,
                                                 detrend)

        assert returned == expected_return

        evaluate_pipeline_mock.assert_called_once_with(
            self.pipeline, self.name, self.dataset, signals, self.hyper, self.metrics,
            self.distributed, test_split, detrend)

    @patch('orion.benchmark._evaluate_pipeline')
    def test__evaluate_pipelines_hyperparameter(self, evaluate_pipeline_mock):
        test_split = False
        detrend = False

        signals = [self.signal]
        pipelines = {self.name: self.pipeline}

        hyperparameter = Mock(autospec=dict)
        hyperparameters = {self.name: hyperparameter}

        score = self.set_score(1, ANY, test_split)
        evaluate_pipeline_mock.return_value = [score]

        expected_return = [score]
        returned = benchmark._evaluate_pipelines(pipelines, self.dataset, signals, hyperparameters,
                                                 self.metrics, self.distributed, test_split,
                                                 detrend)

        assert returned == expected_return

        evaluate_pipeline_mock.assert_called_once_with(
            self.pipeline, self.name, self.dataset, signals, hyperparameter, self.metrics,
            self.distributed, test_split, detrend)

    @patch('orion.benchmark._evaluate_pipelines')
    def test__evaluate_datasets(self, evaluate_pipelines_mock):
        test_split = False
        detrend = False

        signals = [self.signal]
        datasets = {self.dataset: signals}
        pipelines = {self.name, self.pipeline}

        score = self.set_score(1, ANY, test_split)
        evaluate_pipelines_mock.return_value = [score]

        order = ['dataset', 'elapsed', 'metric-name', 'pipeline', 'signal', 'split', 'status']
        expected_return = pd.DataFrame.from_records([{
            'metric-name': 1,
            'elapsed': ANY,
            'split': test_split,
            'pipeline': self.name,
            'dataset': self.dataset,
            'signal': self.signal,
            'status': 'OK'
        }])[order]

        returned = benchmark._evaluate_datasets(
            pipelines, datasets, self.hyper, self.metrics, self.distributed, test_split, detrend)

        pd.testing.assert_frame_equal(returned[order], expected_return)

        evaluate_pipelines_mock.assert_called_once_with(
            pipelines, self.dataset, signals, self.hyper, self.metrics,
            self.distributed, test_split, detrend)

    @patch('orion.benchmark._evaluate_datasets')
    def test_benchmark(self, evaluate_datasets_mock):
        signals = [self.signal]
        datasets = {self.dataset: signals}
        pipelines = {self.name, self.pipeline}

        score = self.set_score(1, ANY, ANY)
        evaluate_datasets_mock.return_value = pd.DataFrame.from_records([score])

        order = [
            'pipeline',
            'rank',
            'dataset',
            'elapsed',
            'metric-name',
            'signal',
            'split',
            'status']

        expected_return = pd.DataFrame.from_records([{
            'rank': 1,
            'metric-name': 1,
            'elapsed': ANY,
            'split': ANY,
            'pipeline': self.name,
            'dataset': self.dataset,
            'signal': self.signal,
            'status': 'OK'
        }])[order]

        returned = benchmark.benchmark(
            pipelines, datasets, self.hyper, self.metrics, self.rank, self.distributed)

        pd.testing.assert_frame_equal(returned[order], expected_return)

        evaluate_datasets_mock.assert_called_once_with(
            pipelines, datasets, self.hyper, self.metrics, self.distributed, False, False)

    @patch('orion.benchmark._evaluate_datasets')
    def test_benchmark_metrics_list(self, evaluate_datasets_mock):
        test_split = False
        detrend = False

        signals = [self.signal]
        datasets = {self.dataset: signals}
        pipelines = {self.name: self.pipeline}

        metric = Mock(autospec=METRICS['f1'], return_value=1)
        metric.__name__ = 'metric-name'
        metrics = [metric]
        metrics_ = {metric.__name__: metric}

        score = self.set_score(1, ANY, test_split)
        score[metric.__name__] = metric
        evaluate_datasets_mock.return_value = pd.DataFrame.from_records([score])

        order = [
            'pipeline',
            'rank',
            'dataset',
            'elapsed',
            'metric-name',
            'signal',
            'split',
            'status']

        expected_return = pd.DataFrame.from_records([{
            'rank': 1,
            'metric-name': metric,
            'elapsed': ANY,
            'split': test_split,
            'pipeline': self.name,
            'dataset': self.dataset,
            'signal': self.signal,
            'status': 'OK'
        }])[order]

        returned = benchmark.benchmark(pipelines, datasets, self.hyper, metrics, self.rank,
                                       self.distributed, test_split, detrend)

        pd.testing.assert_frame_equal(returned[order], expected_return)

        evaluate_datasets_mock.assert_called_once_with(
            pipelines, datasets, self.hyper, metrics_, self.distributed, test_split, detrend)

    @patch('orion.benchmark._evaluate_datasets')
    def test_benchmark_metrics_exception(self, evaluate_datasets_mock):
        test_split = False
        detrend = False

        signals = [self.signal]
        datasets = {self.dataset: signals}
        pipelines = {self.name: self.pipeline}

        metric = 'does-not-exist'
        metrics = [metric]

        score = self.set_score(1, ANY, test_split)
        evaluate_datasets_mock.return_value = pd.DataFrame.from_records([score])

        with self.assertRaises(ValueError) as ex:
            benchmark.benchmark(pipelines, datasets, self.hyper, metrics, self.rank,
                                self.distributed, test_split, detrend)

            self.assertTrue(metric in ex.exception)

    @patch('orion.benchmark._evaluate_datasets')
    def test_benchmark_pipelines_list(self, evaluate_datasets_mock):
        test_split = False
        detrend = False

        signals = [self.signal]
        datasets = {self.dataset: signals}
        pipelines = [self.pipeline]
        pipelines_ = {self.pipeline: self.pipeline}

        score = self.set_score(1, ANY, test_split)
        score['pipeline'] = self.pipeline
        evaluate_datasets_mock.return_value = pd.DataFrame.from_records([score])

        order = [
            'pipeline',
            'rank',
            'dataset',
            'elapsed',
            'metric-name',
            'signal',
            'split',
            'status']

        expected_return = pd.DataFrame.from_records([{
            'rank': 1,
            'metric-name': 1,
            'elapsed': ANY,
            'split': test_split,
            'pipeline': self.pipeline,
            'dataset': self.dataset,
            'signal': self.signal,
            'status': 'OK'
        }])[order]

        returned = benchmark.benchmark(pipelines, datasets, self.hyper, self.metrics, self.rank,
                                       self.distributed, test_split, detrend)

        pd.testing.assert_frame_equal(returned[order], expected_return)

        evaluate_datasets_mock.assert_called_once_with(
            pipelines_, datasets, self.hyper, self.metrics, self.distributed, test_split, detrend)
