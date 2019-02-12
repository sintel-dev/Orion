#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pipeline` module."""
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from orion.pipeline import OrionPipeline


class TestOrionPipeline(TestCase):
    """Tests for `OrionPipeline`."""

    def setUp(self):
        self.X = pd.DataFrame([
            {'timestamp': 1, 'value': 0.1},
            {'timestamp': 2, 'value': 0.2},
            {'timestamp': 3, 'value': 0.3},
            {'timestamp': 4, 'value': 0.4},
            {'timestamp': 5, 'value': 0.5},
        ])
        self.y = pd.Series([0, 0, 0, 0, 0])

    def test___init___cv_splits(self):
        """If cv_splits is passed a new cv object is created with the specified param."""

        # Run
        instance = OrionPipeline(template={'primitives': []}, cv_splits=5)

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
        instance = OrionPipeline(template={'primitives': []}, cv=cv, scorer=scorer)

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
        instance = OrionPipeline(template={'primitives': []}, )

        # Check
        assert instance._cv.__class__ == OrionPipeline._cv_class
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
        instance = OrionPipeline(template={'primitives': []}, cv=cv, scorer=scorer)
        instance._best_score = 1

        # Run / Check
        assert not instance._is_better(0)
        assert not instance._is_better(1)
        assert instance._is_better(2)

    @patch('orion.pipeline.OrionPipeline._score_pipeline')
    @patch('orion.pipeline.GP')
    def test_tune(self, gp_mock, score_mock):
        """tune select the best hyperparameters for the given data."""

        # Setup - Classifier
        iterations = 2
        instance = OrionPipeline(template={'primitives': []}, )
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
        instance.tune(self.X, self.y, iterations)

        # Check
        gp_mock.assert_called_once_with(tunables)
        assert instance._tuner == gp_mock_instance

        assert gp_mock_instance.propose.call_count == iterations
        assert gp_mock_instance.propose.call_args_list == expected_propose_calls

        assert gp_mock_instance.add.call_count == iterations + 1
        assert gp_mock_instance.add.call_args_list == expected_add_calls

        assert instance.fitted is False

    @patch('orion.pipeline.MLPipeline')
    def test_fit(self, pipeline_mock):
        """fit prepare the pipeline to make predictions based on the given data."""

        # Setup - Mock
        pipeline_mock_instance = MagicMock()
        pipeline_mock.from_dict.return_value = pipeline_mock_instance

        # Setup - Classifier
        instance = OrionPipeline(template={'primitives': []}, )

        # Run
        instance.fit(self.X, self.y)

        # Check
        pipeline_mock.from_dict.assert_called_once_with({'primitives': []})
        assert instance._pipeline == pipeline_mock_instance

        pipeline_mock_instance.fit.assert_called_once_with(self.X, self.y)
        assert instance.fitted

    @patch('orion.pipeline.MLPipeline')
    def test_predict(self, pipeline_mock):
        """predict produces results using the pipeline."""
        # Setup - Mock
        pipeline_mock_instance = MagicMock()
        pipeline_mock.from_dict.return_value = pipeline_mock_instance

        # Setup - Classifier
        scorer = f1_score
        cv = StratifiedKFold(n_splits=2, shuffle=True)
        instance = OrionPipeline(template={'primitives': []}, cv=cv, scorer=scorer)
        instance.fit(self.X, self.y)

        # Run
        instance.predict(self.X)

        # Check
        pipeline_mock.from_dict.assert_called_once_with({'primitives': []})
        assert instance._pipeline == pipeline_mock_instance

        pipeline_mock_instance.fit.assert_called_once_with(self.X, self.y)
        assert instance.fitted

        pipeline_mock_instance.predict.assert_called_once_with(self.X)
