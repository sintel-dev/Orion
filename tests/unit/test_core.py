import os

import numpy as np
import pandas as pd
from mlblocks import MLPipeline
from mlblocks.discovery import load_pipeline

from orion.core import Orion

REPR = 'Orion:\n{}\nhyperparameters:\n{}\n'


def test_repr():
    orion = Orion('dummy')
    assert repr(orion) == REPR.format('    dummy', 'None')


def test_repr_hyperparameters():
    orion = Orion('dummy', {"orion.primitives.detectors.ThresholdDetector#1": {"ratio": 0.5}})
    hyper = "    orion.primitives.detectors.ThresholdDetector#1: {'ratio': 0.5}"
    assert repr(orion) == REPR.format('    dummy', hyper)


def test_repr_mlpipeline():
    mlpipeline = MLPipeline('dummy')
    orion = Orion(mlpipeline)
    primitives = "    orion.primitives.estimators.MeanEstimator\n" \
                 "    orion.primitives.detectors.ThresholdDetector\n" \
                 "    orion.primitives.intervals.build_anomaly_intervals"

    print(repr(orion))
    assert repr(orion) == REPR.format(primitives, 'None')


def test_repr_dict():
    pipeline = load_pipeline('dummy')
    orion = Orion(pipeline)
    primitives = "    orion.primitives.estimators.MeanEstimator\n" \
                 "    orion.primitives.detectors.ThresholdDetector\n" \
                 "    orion.primitives.intervals.build_anomaly_intervals"

    assert repr(orion) == REPR.format(primitives, 'None')


class TestOrion:

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

        cls.orion = Orion('dummy')

    def test_fit(self):
        self.orion.fit(self.clean)

    def test_detect(self):
        self.orion.fit(self.clean)

        events = self.orion.detect(self.anomalous)

        pd.testing.assert_frame_equal(self.events, events)

    def test_detect_no_visualization(self):
        self.orion.fit(self.clean)

        events, visualization = self.orion.detect(self.anomalous, visualization=True)

        pd.testing.assert_frame_equal(self.events, events)

        assert visualization == {}

    def test_detect_visualization(self):
        pipeline = load_pipeline('dummy')
        pipeline['outputs'] = {
            'visualization': [
                {
                    'name': 'y_hat',
                    'variable': 'orion.primitives.estimators.MeanEstimator#1.y'
                }
            ]
        }
        orion = Orion(pipeline)
        orion.fit(self.clean)

        events, visualization = orion.detect(self.anomalous, visualization=True)

        pd.testing.assert_frame_equal(self.events, events)

        assert isinstance(visualization, dict)
        assert 'y_hat' in visualization
        y_hat = visualization['y_hat']
        np.testing.assert_array_equal(y_hat, np.ones(len(self.anomalous)))

    def test_fit_detect(self):
        events = self.orion.fit_detect(self.all_data)

        pd.testing.assert_frame_equal(self.all_events, events)

    def test_save_load(self, tmpdir):
        path = os.path.join(tmpdir, 'some/path.pkl')
        self.orion.save(path)

        new_orion = Orion.load(path)
        assert new_orion == self.orion

    def test_evaluate(self):
        self.orion.fit(self.clean)
        scores = self.orion.evaluate(data=self.anomalous, ground_truth=self.events)

        expected = pd.Series({
            'accuracy': 1.0,
            'f1': 1.0,
            'recall': 1.0,
            'precision': 1.0,
        })
        pd.testing.assert_series_equal(expected, scores)

    def test_evaluate_fit(self):
        scores = self.orion.evaluate(
            data=self.all_data,
            ground_truth=self.all_events,
            fit=True,
        )

        expected = pd.Series({
            'accuracy': 1.0,
            'f1': 1.0,
            'recall': 1.0,
            'precision': 1.0,
        })
        pd.testing.assert_series_equal(expected, scores)

    def test_evaluate_train_data(self):
        scores = self.orion.evaluate(
            data=self.anomalous,
            ground_truth=self.events,
            fit=True,
            train_data=self.clean
        )

        expected = pd.Series({
            'accuracy': 1.0,
            'f1': 1.0,
            'recall': 1.0,
            'precision': 1.0,
        })
        pd.testing.assert_series_equal(expected, scores)
