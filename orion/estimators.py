# -*- coding: utf-8 -*-

import json
import logging
import os
from collections import defaultdict

import numpy as np
from btb import HyperParameter
from btb.tuning import GP
from mlblocks import MLPipeline
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold

LOGGER = logging.getLogger(__name__)


class TimeSeriesAnomalyDetector(object):

    template = {
        'primitives': []
    }
    fitted = False

    _cv_class = KFold
    _cost = None
    _tuner = None
    _pipeline = None

    def _get_cv(self, cv_splits, random_state):
        return self._cv_class(n_splits=cv_splits, shuffle=True, random_state=random_state)

    def _load_mlpipeline(self, template):
        if not isinstance(template, dict):
            template_name = template
            if os.path.isfile(template_name):
                with open(template_name, 'r') as template_file:
                    template = json.load(template_file)

            elif self._db:
                template = self._db.load_template(template_name)

            if not template:
                raise ValueError('Unknown template {}'.format(template_name))

            self.template = template

        return MLPipeline.from_dict(template)

    def __init__(self, db=None, template=None, hyperparameters=None,
                 scorer=None, cost=False, cv=None, cv_splits=5, random_state=0):

        self._cv = cv or self._get_cv(cv_splits, random_state)

        if scorer:
            self._score = scorer

        self._cost = cost

        self._db = db

        self._pipeline = self._load_mlpipeline(template or self.template)

        if hyperparameters:
            self._pipeline.set_hyperparameters(hyperparameters)

    def get_hyperparameters(self):
        return self._pipeline.get_hyperparameters()

    def set_hyperparameters(self, hyperparameters):
        self._pipeline.set_hyperparameters(hyperparameters)
        self.fitted = False

    @staticmethod
    def _clone_pipeline(pipeline):
        return MLPipeline.from_dict(pipeline.to_dict())

    def _is_better(self, score):
        if self._cost:
            return score < self._best_score

        return score > self._best_score

    def _get_tunables(self):
        tunables = []
        tunable_keys = []
        for block_name, params in self._pipeline.get_tunable_hyperparameters().items():
            for param_name, param_details in params.items():
                key = (block_name, param_name)
                param_type = param_details['type']
                param_type = 'string' if param_type == 'str' else param_type

                if param_type == 'bool':
                    param_range = [True, False]
                else:
                    param_range = param_details.get('range') or param_details.get('values')

                value = HyperParameter(param_type, param_range)
                tunables.append((key, value))
                tunable_keys.append(key)

        return tunables, tunable_keys

    def _score_pipeline(self, pipeline, X, y, data):
        scores = []

        for fold, (train_index, test_index) in enumerate(self._cv.split(X, y)):
            LOGGER.debug('Scoring fold %s', fold)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            pipeline = self._clone_pipeline(pipeline)
            pipeline.fit(X_train, y_train, **data)

            predictions = pipeline.predict(X_test, **data)
            score = self._score(y_test, predictions)

            LOGGER.debug('Fold fold %s score: %s', fold, score)
            scores.append(score)

        return np.mean(scores)

    def _to_dicts(self, hyperparameters):

        params_tree = defaultdict(dict)
        for (block, hyperparameter), value in hyperparameters.items():
            if isinstance(value, np.integer):
                value = int(value)

            elif isinstance(value, np.floating):
                value = float(value)

            elif isinstance(value, np.ndarray):
                value = value.tolist()

            elif value == 'None':
                value = None

            params_tree[block][hyperparameter] = value

        return params_tree

    def _to_tuples(self, params_tree, tunable_keys):
        param_tuples = defaultdict(dict)
        for block_name, params in params_tree.items():
            for param, value in params.items():
                key = (block_name, param)
                if key in tunable_keys:
                    param_tuples[key] = 'None' if value is None else value

        return param_tuples

    def _get_tuner(self):
        tunables, tunable_keys = self._get_tunables()
        tuner = GP(tunables)

        # Inform the tuner about the score that the default hyperparmeters obtained
        param_tuples = self._to_tuples(self._pipeline.get_hyperparameters(), tunable_keys)
        tuner.add(param_tuples, self._best_score)

        return tuner

    def tune(self, X, y, data, iterations=10):
        if not self._tuner:
            LOGGER.info('Scoring the default pipeline')
            self._best_score = self._score_pipeline(self._pipeline, X, y, data)
            self._tuner = self._get_tuner()

        dataset = data['dataset_name']
        table = data['target_entity']
        column = data['target_column']

        for i in range(iterations):
            LOGGER.info('Scoring pipeline %s', i + 1)

            params = self._tuner.propose(1)
            param_dicts = self._to_dicts(params)

            candidate = self._clone_pipeline(self._pipeline)
            candidate.set_hyperparameters(param_dicts)

            score = self._score_pipeline(candidate, X, y, data)
            LOGGER.info('Pipeline %s score: %s', i + 1, score)

            self._tuner.add(params, score)

            if self._db:
                self._db.insert_pipeline(candidate, score, dataset, table, column)

            if self._is_better(score):
                self._best_score = score
                self.set_hyperparameters(param_dicts)

    def fit(self, X, y, data):
        self._pipeline.fit(X, y, **data)
        self.fitted = True

    def predict(self, X, data):
        if not self.fitted:
            raise NotFittedError()

        return self._pipeline.predict(X, **data)
