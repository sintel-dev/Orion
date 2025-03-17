import os
import shutil
from pathlib import Path
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
    pd.testing.assert_frame_equal(returned, expected_return[returned.columns], check_dtype=False)


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


def test__get_pipeline_hyperparameter():
    hyperparameters = {
        "pipeline1": "pipeline1.json",
        "pipeline2": "pipeline2.json",
    }
    pipeline = "pipeline1"

    expected_return = "pipeline1.json"
    returned = benchmark._get_pipeline_hyperparameter(hyperparameters, None, pipeline)
    assert returned == expected_return


def test__get_pipeline_hyperparameter_dataset():
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
    returned = benchmark._get_pipeline_hyperparameter(hyperparameters, dataset, None)
    assert returned == expected_return


def test__get_pipeline_hyperparameter_does_not_exist():
    hyperparameters = None
    pipeline = "pipeline1"

    expected_return = None
    returned = benchmark._get_pipeline_hyperparameter(hyperparameters, None, pipeline)
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
        cls.test_split = False
        cls.detrend = False
        cls.pipeline_path = None
        cls.anomaly_path = None
        cls.cache_dir = None
        cls.iteration = 0
        cls.run_id = ANY
        cls.metrics = {
            'metric-name': Mock(autospec=METRICS['f1'], return_value=1)
        }

    def args(self):
        return (
            self.pipeline,
            self.name,
            self.dataset,
            self.signal,
            self.hyper,
            self.metrics,
            self.test_split,
            self.detrend,
            self.iteration,
            self.cache_dir,
            self.pipeline_path,
            self.anomaly_path,
            self.run_id
        )

    def set_score(self, metric, elapsed, test_split):
        return {
            'metric-name': metric,
            'elapsed': elapsed,
            'split': test_split,
            'status': 'OK'
        }

    def set_output(self, metric, elapsed, test_split):
        return {
            'dataset': self.dataset,
            'pipeline': self.name,
            'signal': self.signal,
            'iteration': self.iteration,
            'metric-name': metric,
            'elapsed': elapsed,
            'split': test_split,
            'status': 'OK',
            'run_id': self.run_id
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
            self.pipeline, self.signal, self.hyper, self.metrics, True)

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
            self.pipeline, self.signal, self.hyper, self.metrics, True)

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
            self.pipeline, self.signal, self.hyper, metrics, True)

        expected_return = self.set_score(0, ANY, ANY)
        expected_return['status'] = 'ERROR'
        expected_return['tn'] = None
        expected_return['fp'] = 0
        expected_return['fn'] = 2
        expected_return['tp'] = 0

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
            self.pipeline, self.signal, self.hyper, self.metrics, test_split=test_split)

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
            self.pipeline, self.signal, self.hyper, self.metrics, test_split=test_split)

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
            self.pipeline, self.signal, self.hyper, self.metrics, test_split=True, detrend=detrend)

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
            self.pipeline, self.signal, self.hyper, self.metrics, test_split=True, detrend=detrend)

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
    def test__run_job(self, evaluate_signal_mock):
        pass

        score = self.set_score(1, ANY, ANY)
        evaluate_signal_mock.return_value = score

        benchmark._run_job(self.args())

        expected_calls = [
            self.pipeline, self.signal, self.hyper, self.metrics,
            self.test_split, self.detrend, self.pipeline_path, self.anomaly_path
        ]

        evaluate_signal_mock.assert_called_once_with(*expected_calls)

    @patch('orion.benchmark._run_job')
    def test_benchmark(self, run_job_mock):
        signals = [self.signal]
        datasets = {self.dataset: signals}
        pipelines = {self.name: self.pipeline}

        output = self.set_output(1, ANY, ANY)
        run_job_mock.return_value = pd.DataFrame.from_records([output])

        order = [
            'pipeline',
            'rank',
            'dataset',
            'signal',
            'iteration',
            'metric-name',
            'status',
            'elapsed',
            'split',
            'run_id']

        expected_return = pd.DataFrame.from_records([{
            'metric-name': 1,
            'rank': 1,
            'elapsed': ANY,
            'split': ANY,
            'status': 'OK',
            'iteration': self.iteration,
            'pipeline': self.name,
            'dataset': self.dataset,
            'signal': self.signal,
            'run_id': self.run_id
        }])[order]

        returned = benchmark.benchmark(
            pipelines, datasets, self.hyper, self.metrics, self.rank)[order]

        pd.testing.assert_frame_equal(returned, expected_return)

        run_job_mock.assert_called_once_with(self.args())

    @patch('orion.benchmark._run_job')
    def test_benchmark_dataset_list(self, run_job_mock):
        signals = [self.signal]
        pipelines = {self.name: self.pipeline}

        output = self.set_output(1, ANY, ANY)
        output['dataset'] = 'dataset'
        run_job_mock.return_value = pd.DataFrame.from_records([output])

        order = [
            'pipeline',
            'rank',
            'dataset',
            'signal',
            'iteration',
            'metric-name',
            'status',
            'elapsed',
            'split',
            'run_id']

        expected_return = pd.DataFrame.from_records([{
            'metric-name': 1,
            'rank': 1,
            'elapsed': ANY,
            'split': ANY,
            'status': 'OK',
            'iteration': self.iteration,
            'pipeline': self.name,
            'dataset': 'dataset',
            'signal': self.signal,
            'run_id': self.run_id
        }])[order]

        returned = benchmark.benchmark(
            pipelines, signals, self.hyper, self.metrics, self.rank)[order]

        pd.testing.assert_frame_equal(returned, expected_return)

        args = list(self.args())
        args[2] = 'dataset'
        run_job_mock.assert_called_once_with(tuple(args))

    @patch('orion.benchmark._run_job')
    def test_benchmark_pipelines_list(self, run_job_mock):
        signals = [self.signal]
        datasets = {self.dataset: signals}
        pipelines = [self.pipeline]

        hyper = {}

        output = self.set_output(1, ANY, ANY)
        output['pipeline'] = self.pipeline
        run_job_mock.return_value = pd.DataFrame.from_records([output])

        order = [
            'pipeline',
            'rank',
            'dataset',
            'signal',
            'iteration',
            'metric-name',
            'status',
            'elapsed',
            'split',
            'run_id']

        expected_return = pd.DataFrame.from_records([{
            'metric-name': 1,
            'rank': 1,
            'elapsed': ANY,
            'split': ANY,
            'status': 'OK',
            'iteration': self.iteration,
            'pipeline': self.pipeline,
            'dataset': self.dataset,
            'signal': self.signal,
            'run_id': self.run_id
        }])[order]

        returned = benchmark.benchmark(
            pipelines, datasets, hyper, self.metrics, self.rank)[order]

        pd.testing.assert_frame_equal(returned, expected_return)

        args = list(self.args())
        args[1] = self.pipeline
        args[4] = hyper

        run_job_mock.assert_called_once_with(tuple(args))

    @patch('orion.benchmark._run_job')
    def test_benchmark_metrics_list(self, run_job_mock):
        signals = [self.signal]
        datasets = {self.dataset: signals}
        pipelines = {self.name: self.pipeline}

        metric = Mock(autospec=METRICS['f1'], return_value=1)
        metric.__name__ = 'metric-name'
        metrics = [metric]
        metrics_ = {metric.__name__: metric}

        output = self.set_output(1, ANY, ANY)
        output[metric.__name__] = metric
        run_job_mock.return_value = pd.DataFrame.from_records([output])

        order = [
            'pipeline',
            'rank',
            'dataset',
            'signal',
            'iteration',
            'metric-name',
            'status',
            'elapsed',
            'split',
            'run_id']

        expected_return = pd.DataFrame.from_records([{
            'metric-name': metric,
            'rank': 1,
            'elapsed': ANY,
            'split': ANY,
            'status': 'OK',
            'iteration': self.iteration,
            'pipeline': self.name,
            'dataset': self.dataset,
            'signal': self.signal,
            'run_id': self.run_id
        }])[order]

        returned = benchmark.benchmark(
            pipelines, datasets, self.hyper, metrics, self.rank)[order]

        pd.testing.assert_frame_equal(returned, expected_return)

        args = list(self.args())
        args[5] = metrics_

        run_job_mock.assert_called_once_with(tuple(args))

    def test_benchmark_metrics_exception(self):
        signals = [self.signal]
        datasets = {self.dataset: signals}
        pipelines = {self.name: self.pipeline}

        metric = 'does-not-exist'
        metrics = [metric]

        with self.assertRaises(ValueError) as ex:
            benchmark.benchmark(pipelines, datasets, self.hyper, metrics, self.rank)

            self.assertTrue(metric in ex.exception)

    def test_benchmark_defaults(self):
        pipelines = ['dummy']
        datasets = ['S-1']

        scores = benchmark.benchmark(pipelines=pipelines, datasets=datasets)

        expected = pd.DataFrame.from_records([{
            'pipeline': 'dummy',
            'rank': 1,
            'dataset': 'dataset',
            'signal': 'S-1',
            'iteration': 0,
            'accuracy': ANY,
            'f1': ANY,
            'recall': ANY,
            'precision': ANY,
            'status': 'OK',
            'elapsed': ANY,
            'split': False,
            'run_id': ANY
        }])
        pd.testing.assert_frame_equal(expected, scores, check_dtype=False)

    def test_benchmark_resume(self):
        cache_dir = Path('cache_test')
        pipelines = ['dummy']
        datasets = ['S-1']

        # create a run in cache_test
        os.makedirs(cache_dir, exist_ok=True)
        file_name = str(
            cache_dir / f'{pipelines[0]}_{datasets[0]}_dataset_0'
        )
        with open(file_name + "_run_id.csv", "w") as f:
            f.write("this is a test file.")

        scores = benchmark.benchmark(
            pipelines=pipelines,
            datasets=datasets,
            cache_dir=cache_dir,
            resume=True
        )

        expected = pd.DataFrame()
        pd.testing.assert_frame_equal(expected, scores)

        shutil.rmtree(cache_dir)
