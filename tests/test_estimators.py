#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `estimators` module."""
from datetime import datetime
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from orion.estimators import TimeSeriesAnomalyDetector


class TestTimeSeriesAnomalyDetector(TestCase):
    """Tests for `TimeSeriesAnomalyDetector`."""

    def setUp(self):
        time_index = 'timeseries_id'
        index = 'demand_id'
        dataset_name = 'test_dataset'
        target_column = 'label'

        timeseries = pd.DataFrame({
            'timeseries_id': [0, 1, 2, 3, 4],
        })

        demand = pd.DataFrame(
            {
                'label': [1, 1, 2, 2, 1],
                'demand_id': [0, 1, 2, 3, 4],
                'timeseries_id': [0, 1, 2, 3, 4],
                'cutoff_time': [
                    datetime(2010, 1, 25),
                    datetime(2010, 1, 25),
                    datetime(2010, 1, 25),
                    datetime(2010, 1, 25),
                    datetime(2010, 1, 25)
                ]
            }
        )
        self.y = demand.pop(target_column)
        self.X = demand

        data = pd.DataFrame(
            {
                'data_id': [0, 1, 2, 3, 4],
                'timeseries_id': [0, 0, 1, 1, 2],
                'timestamp': [
                    datetime(2010, 1, 1),
                    datetime(2010, 1, 2),
                    datetime(2010, 1, 3),
                    datetime(2010, 1, 4),
                    datetime(2010, 1, 5)
                ],
                'value': [
                    -0.7105199999999999,
                    -1.1833,
                    -1.3724,
                    -1.5931,
                    -1.4669999999999999
                ]
            }
        )

        relationships = [
            ('timeseries', time_index, 'demand', index),
            ('timeseries', time_index, 'data', 'data_id')
        ]

        entities = {
            'timeseries': (timeseries, time_index, ),
            'demand': (demand, index, 'cutoff_time'),
            'data': (data, 'data_id', 'timestamp')
        }

        self.data = {
            'entities': entities,
            'relationships': relationships,
            'target_entity': 'demand',
            'dataset_name': dataset_name,
            'target_column': target_column
        }

    def test___init___cv_splits(self):
        """If cv_splits is passed a new cv object is created with the specified param."""

        # Run
        instance = TimeSeriesAnomalyDetector(cv_splits=5)

        # Check
        assert instance._cv.n_splits == 5
        assert instance._cost is False
        assert instance._db is None
        assert instance._tuner is None
        assert instance.fitted is False

    def test___init___cv_object(self):
        """If a cv object is passed as cv argument, it will be used for cross-validation."""
        # Setup
        scorer = f1_score
        cv = StratifiedKFold(n_splits=5, shuffle=True)

        # Run
        instance = TimeSeriesAnomalyDetector(cv=cv, scorer=scorer)

        # Check
        assert instance._cv == cv
        assert instance._score == scorer
        assert instance._cost is False
        assert instance._db is None
        assert instance._tuner is None
        assert instance.fitted is False

    def test___init___default_args(self):
        """__init__ use defaults if no args are passed."""

        # Run
        instance = TimeSeriesAnomalyDetector()

        # Check
        assert instance._cv.__class__ == TimeSeriesAnomalyDetector._cv_class
        assert instance._cv.n_splits == 5
        assert instance._cv.shuffle is True
        assert instance._cv.random_state == 0
        assert instance._cost is False
        assert instance._db is None
        assert instance._tuner is None
        assert instance.fitted is False

    def test__is_better(self):
        """_is_better only return true if the argument is greater than the _best_score."""
        # Setup
        scorer = f1_score
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        instance = TimeSeriesAnomalyDetector(cv=cv, scorer=scorer)
        instance._best_score = 1

        # Run / Check
        assert not instance._is_better(0)
        assert not instance._is_better(1)
        assert instance._is_better(2)

    @patch('orion.estimators.TimeSeriesAnomalyDetector._score_pipeline')
    @patch('orion.estimators.GP')
    def test_tune(self, gp_mock, score_mock):
        """tune select the best hyperparameters for the given data."""

        # Setup - Classifier
        iterations = 2
        instance = TimeSeriesAnomalyDetector()
        tunables, tunable_keys = instance._get_tunables()

        # Setup - Mock
        score_mock.return_value = 0.0
        gp_mock_instance = MagicMock()
        gp_mock.return_value = gp_mock_instance

        expected_propose_calls = [((1, ), ), ((1, ), )]
        expected_best_score = 0.0
        param_tuples = instance._to_tuples(instance._pipeline.get_hyperparameters(), tunable_keys)
        expected_add_calls = [
            ((param_tuples, expected_best_score), ),
            ((gp_mock_instance.propose.return_value, expected_best_score), ),
            ((gp_mock_instance.propose.return_value, expected_best_score), ),
        ]

        # Run
        instance.tune(self.X, self.y, self.data, iterations)

        # Check
        gp_mock.assert_called_once_with(tunables)
        assert instance._tuner == gp_mock_instance

        assert gp_mock_instance.propose.call_count == iterations
        assert gp_mock_instance.propose.call_args_list == expected_propose_calls

        assert gp_mock_instance.add.call_count == iterations + 1
        assert gp_mock_instance.add.call_args_list == expected_add_calls

        assert instance.fitted is False

    @patch('orion.estimators.MLPipeline')
    def test_fit(self, pipeline_mock):
        """fit prepare the pipeline to make predictions based on the given data."""

        # Setup - Mock
        pipeline_mock_instance = MagicMock()
        pipeline_mock.from_dict.return_value = pipeline_mock_instance

        # Setup - Classifier
        instance = TimeSeriesAnomalyDetector()

        # Run
        instance.fit(self.X, self.y, self.data)

        # Check
        pipeline_mock.from_dict.assert_called_once_with(instance.template)
        assert instance._pipeline == pipeline_mock_instance

        pipeline_mock_instance.fit.assert_called_once_with(self.X, self.y, **self.data)
        assert instance.fitted

    @patch('orion.estimators.MLPipeline')
    def test_predict(self, pipeline_mock):
        """predict produces results using the pipeline."""
        # Setup - Mock
        pipeline_mock_instance = MagicMock()
        pipeline_mock.from_dict.return_value = pipeline_mock_instance

        # Setup - Classifier
        scorer = f1_score
        cv = StratifiedKFold(n_splits=2, shuffle=True)
        instance = TimeSeriesAnomalyDetector(cv=cv, scorer=scorer)
        instance.fit(self.X, self.y, self.data)

        # Run
        instance.predict(self.X, self.data)

        # Check
        pipeline_mock.from_dict.assert_called_once_with(instance.template)
        assert instance._pipeline == pipeline_mock_instance

        pipeline_mock_instance.fit.assert_called_once_with(self.X, self.y, **self.data)
        assert instance.fitted

        pipeline_mock_instance.predict.assert_called_once_with(self.X, **self.data)
