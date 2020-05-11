import json
import os
from unittest.mock import call, patch

import pandas as pd
import pytest
from mlblocks import MLPipeline
from mlblocks.discovery import load_pipeline

from orion import functional
from orion.core import Orion


class TestLoadData:

    @classmethod
    def setup_class(cls):
        cls.data = pd.DataFrame({
            'timestamp': list(range(100)),
            'value': [1] * 100,
        })

    def test_load_data_df(self):
        data = functional._load_data(self.data)

        assert data is self.data

    def test_load_data_path(self, tmpdir):
        path = os.path.join(tmpdir, 'data.csv')
        self.data.to_csv(path, index=False)

        data = functional._load_data(path)

        pd.testing.assert_frame_equal(data, self.data)

    def test_load_data_none(self):
        data = functional._load_data(None)

        assert data is None


class TestLoadDict:

    def test_load_dict_dict(self):
        a_dict = {'a': 'dict'}
        returned = functional._load_dict(a_dict)

        assert returned is a_dict

    def test_load_dict_json(self, tmpdir):
        a_dict = {'a': 'dict'}
        path = os.path.join(tmpdir, 'a.json')
        with open(path, 'w') as json_file:
            json.dump(a_dict, json_file)

        returned = functional._load_dict(path)

        assert returned == a_dict

    def test_load_dict_none(self):
        returned = functional._load_dict(None)

        assert returned is None


class TestLoadOrion:

    @classmethod
    def setup_class(cls):
        data = pd.DataFrame({
            'timestamp': list(range(100)),
            'value': [1] * 100,
        })
        cls.orion = Orion('dummy')
        cls.orion.fit(data)

    def test_load_orion_orion(self):
        orion = functional._load_orion(self.orion)

        assert orion is self.orion

    def test_load_orion_pickle(self, tmpdir):
        path = os.path.join(tmpdir, 'orion.pkl')
        self.orion.save(path)

        orion = functional._load_orion(path)

        assert orion is not self.orion
        assert orion == self.orion

    def test_load_orion_name(self):
        orion = functional._load_orion('dummy')

        assert isinstance(orion, Orion)
        assert orion._pipeline == 'dummy'
        assert not orion._fitted
        assert orion._hyperparameters is None

    def test_load_orion_json_path(self, tmpdir):
        pipeline = load_pipeline('dummy')
        path = os.path.join(tmpdir, 'pipeline.json')
        with open(path, 'w') as json_file:
            json.dump(pipeline, json_file)

        orion = functional._load_orion(path)

        assert isinstance(orion, Orion)
        assert orion._pipeline == path
        assert not orion._fitted
        assert orion._hyperparameters is None

    def test_load_orion_dict(self):
        pipeline = load_pipeline('dummy')
        orion = functional._load_orion(pipeline)

        assert isinstance(orion, Orion)
        assert orion._pipeline == pipeline
        assert not orion._fitted
        assert orion._hyperparameters is None

    def test_load_orion_mlpipeline(self, tmpdir):
        pipeline = MLPipeline('dummy')

        orion = functional._load_orion(pipeline)

        assert isinstance(orion, Orion)
        assert orion._pipeline == pipeline
        assert not orion._fitted
        assert orion._hyperparameters is None

    def test_load_orion_hyperparams(self):
        hyperparams = {
            "orion.primitives.detectors.ThresholdDetector#1": {
                "ratio": 0.9
            }
        }
        orion = functional._load_orion('dummy', hyperparams)

        assert isinstance(orion, Orion)
        assert orion._pipeline == 'dummy'
        assert not orion._fitted
        assert orion._hyperparameters == hyperparams

    def test_load_orion_invalid(self):
        with pytest.raises(ValueError):
            functional._load_orion('invalid')


class TestFitPipeline:

    @classmethod
    def setup_class(cls):
        cls.data = pd.DataFrame({
            'timestamp': list(range(100)),
            'value': [1] * 100,
        })

    @patch('orion.core.Orion.DEFAULT_PIPELINE', new='dummy')
    def test_fit_pipeline_default(self):
        orion = functional.fit_pipeline(self.data)

        assert isinstance(orion, Orion)
        assert orion._pipeline == 'dummy'
        assert orion._fitted
        assert orion._hyperparameters is None

    def test_fit_pipeline_dict(self):
        pipeline = load_pipeline('dummy')

        orion = functional.fit_pipeline(self.data, pipeline)

        assert isinstance(orion, Orion)
        assert orion._pipeline == pipeline
        assert orion._fitted
        assert orion._hyperparameters is None

    def test_fit_pipeline_name(self):
        orion = functional.fit_pipeline(self.data, 'dummy')

        assert isinstance(orion, Orion)
        assert orion._pipeline == 'dummy'
        assert orion._fitted
        assert orion._hyperparameters is None

    def test_fit_pipeline_csv(self, tmpdir):
        data_path = os.path.join(tmpdir, 'data.csv')
        self.data.to_csv(data_path, index=False)

        orion = functional.fit_pipeline(data_path, 'dummy')

        assert isinstance(orion, Orion)
        assert orion._pipeline == 'dummy'
        assert orion._fitted
        assert orion._hyperparameters is None

    def test_fit_pipeline_hyperparams_dict(self):
        hyperparams = {
            "orion.primitives.detectors.ThresholdDetector#1": {
                "ratio": 0.9
            }
        }

        orion = functional.fit_pipeline(self.data, 'dummy', hyperparams)

        assert isinstance(orion, Orion)
        assert orion._pipeline == 'dummy'
        assert orion._fitted
        assert orion._hyperparameters == hyperparams

    def test_fit_pipeline_hyperparams_json(self, tmpdir):
        hyperparams = {
            "orion.primitives.detectors.ThresholdDetector#1": {
                "ratio": 0.9
            }
        }
        hyperparams_path = os.path.join(tmpdir, 'hyperparams.json')
        with open(hyperparams_path, 'w') as json_file:
            json.dump(hyperparams, json_file)

        orion = functional.fit_pipeline(self.data, 'dummy', hyperparams_path)

        assert isinstance(orion, Orion)
        assert orion._pipeline == 'dummy'
        assert orion._fitted
        assert orion._hyperparameters == hyperparams

    def test_fit_pipeline_save_path(self, tmpdir):
        path = os.path.join(tmpdir, 'some/path.pkl')

        returned = functional.fit_pipeline(self.data, 'dummy', save_path=path)

        assert returned is None
        assert os.path.isfile(path)


class TestDetectAnomalies:

    @classmethod
    def setup_class(cls):
        cls.clean = pd.DataFrame({
            'timestamp': list(range(100)),
            'value': [1] * 100,
        })
        cls.anomalous = pd.DataFrame({
            'timestamp': list(range(100, 200)),
            'value': [1] * 45 + [10] * 10 + [1] * 45
        })
        cls.events = pd.DataFrame([
            {'start': 145, 'end': 155, 'severity': 9.0}
        ], columns=['start', 'end', 'severity'])

        cls.all_data = pd.concat((cls.clean, cls.anomalous))
        cls.all_events = pd.DataFrame([
            {'start': 145, 'end': 155, 'severity': 4.275}
        ], columns=['start', 'end', 'severity'])

    @patch('orion.core.Orion.DEFAULT_PIPELINE', new='dummy')
    def test_detect_anomalies_fit_default(self):
        anomalies = functional.detect_anomalies(
            data=self.anomalous,
            train_data=self.clean
        )

        pd.testing.assert_frame_equal(self.events, anomalies)

    def test_detect_anomalies_fit_pipeline(self):
        anomalies = functional.detect_anomalies(
            data=self.anomalous,
            pipeline='dummy',
            train_data=self.clean
        )

        pd.testing.assert_frame_equal(self.events, anomalies)

    @patch('orion.core.Orion.DEFAULT_PIPELINE', new='dummy')
    def test_detect_anomalies_fit_hyperparams(self):
        hyperparams = {
            "orion.primitives.detectors.ThresholdDetector#1": {
                "ratio": 0.9
            }
        }
        anomalies = functional.detect_anomalies(
            data=self.anomalous,
            hyperparameters=hyperparams,
            train_data=self.clean
        )

        pd.testing.assert_frame_equal(self.events, anomalies)

    def test_detect_anomalies_fit_pipeine_dict(self):
        pipeline = load_pipeline('dummy')
        anomalies = functional.detect_anomalies(
            data=self.anomalous,
            pipeline=pipeline,
            train_data=self.clean
        )

        pd.testing.assert_frame_equal(self.events, anomalies)

    def test_detect_anomalies_fitted_orion(self):
        orion = functional.fit_pipeline(self.clean, 'dummy')

        anomalies = functional.detect_anomalies(
            data=self.anomalous,
            pipeline=orion,
        )

        pd.testing.assert_frame_equal(self.events, anomalies)

    def test_detect_anomalies_saved_orion(self, tmpdir):
        orion_path = os.path.join(tmpdir, 'orion.pkl')
        functional.fit_pipeline(self.clean, 'dummy', save_path=orion_path)

        anomalies = functional.detect_anomalies(
            data=self.anomalous,
            pipeline=orion_path,
        )

        pd.testing.assert_frame_equal(self.events, anomalies)


class TestEvaluatePipeline:

    @patch('orion.functional._load_orion')
    @patch('orion.functional._load_data')
    def test_evaluate_pipeline_no_fit(self, load_data_mock, load_orion_mock):
        load_data_mock.side_effect = lambda x: x

        ret = functional.evaluate_pipeline('data', 'truth', 'pipeline', 'hyperparams', 'metrics')

        load_data_calls = [
            call('data'),
            call('truth'),
        ]
        assert load_data_calls == load_data_mock.call_args_list

        load_orion_mock.assert_called_once_with('pipeline', 'hyperparams')

        orion = load_orion_mock.return_value
        orion.detect.assert_called_once_with('data', 'truth', False, None, 'metrics')
        assert ret == orion.detect.return_value

    @patch('orion.functional._load_orion')
    @patch('orion.functional._load_data')
    def test_evaluate_pipeline_fit(self, load_data_mock, load_orion_mock):
        load_data_mock.side_effect = lambda x: x

        ret = functional.evaluate_pipeline(
            'data', 'truth', 'pipeline', 'hyperparams', 'metrics', 'train_data')

        load_data_calls = [
            call('data'),
            call('truth'),
            call('train_data'),
        ]
        assert load_data_calls == load_data_mock.call_args_list

        load_orion_mock.assert_called_once_with('pipeline', 'hyperparams')

        orion = load_orion_mock.return_value
        orion.detect.assert_called_once_with('data', 'truth', True, 'train_data', 'metrics')
        assert ret == orion.detect.return_value
