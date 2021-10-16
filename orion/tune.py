"""Extension to Orion class"""
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
from btb.tuning import GPTuner, Tunable
from mlblocks import MLPipeline
from sklearn.model_selection import KFold

from orion.core import Orion
from orion.evaluation import CONTEXTUAL_METRICS as METRICS

LOGGER = logging.getLogger(__name__)


class OrionTuner(Orion):
    """Extension of Orion Class.

    The OrionExtended Class provides additional features of
    tunning the pipeline.
    """
    _mlpipeline_base = None
    _mlpipeline_post = None
    _time_column = 'timestamp'
    _label_column = 'label'
    _columns = None
    _scorer = None
    tuned = None

    def _extract_pipeline(self):
        pipeline_base = deepcopy(self._mlpipeline.to_dict())
        pipeline_post = deepcopy(self._mlpipeline.to_dict())

        output_primitive = None
        postprocessing = list()
        for primitive, blocks in self._mlpipeline.blocks.items():
            if blocks.metadata.get('classifiers').get('type') == 'postprocessor':
                postprocessing.append(primitive.split('#')[0])
            else:
                output_primitive = primitive

        for attr, value in pipeline_base.items():
            if isinstance(value, dict):
                primitives = list(value.keys())
                for primitive in primitives:
                    name = primitive.split('#')[0]
                    if name in postprocessing:
                        del pipeline_base[attr][primitive]
                    else:
                        del pipeline_post[attr][primitive]

            else:
                pipeline_base[attr] = [p for p in value if p not in postprocessing]
                pipeline_post[attr] = postprocessing

        # change output
        pipeline_post['outputs'] = deepcopy(pipeline_base['outputs'])
        pipeline_base['outputs'] = {
            'default': [{
                'name': 'output',
                'type': 'dict',
                'variable': output_primitive
            }]
        }

        return MLPipeline(pipeline_base), MLPipeline(pipeline_post)

    def _expand(self, data, anomalies):
        data = data.set_index(self._time_column)
        data[self._label_column] = [0] * len(data)
        for i, anom in anomalies.iterrows():
            data.loc[anom[0]: anom[1], self._label_column] = 1

        return data.reset_index()

    def _compress(self, data):
        labels = np.split(data, np.flatnonzero(np.diff(data[self._label_column]) != 0) + 1)

        anomalies = list()
        for segment in labels:
            if sum(segment[self._label_column]) == 0:
                continue

            interval = (segment.iloc[0][self._time_column], segment.iloc[-1][self._time_column])
            anomalies.append(interval)

        return pd.DataFrame(anomalies, columns=['start', 'end'])

    def _get_split(self, data, index):
        X = data.iloc[index]
        y = self._compress(X)
        return X[self.columns], y

    def _cv_split(self, data, anomalies, n_splits=3, random_state=None):
        self.columns = data.columns
        data = self._expand(data, anomalies)

        splits = list()
        cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=False)
        for train_index, test_index in cv.split(data):
            X_train, y_train = self._get_split(data, train_index)
            X_test, y_test = self._get_split(data, test_index)
            splits.append((X_train, y_train, X_test, y_test))

        return splits

    def scoring_function(self, data, anomalies, hyperparameters=None):
        if hyperparameters:
            self._mlpipeline_post.set_hyperparameters(hyperparameters)

        scores = []
        for X_train, y_train, X_test, y_test in self._cv_split(data, anomalies, 3):
            # base step
            output_train = self._mlpipeline_base.predict(X_train)
            output_test = self._mlpipeline_base.predict(X_test)

            # post step
            self._mlpipeline_post.fit(**output_train)
            outputs = self._mlpipeline_post.predict(**output_test)
            LOGGER.debug('Actual %s - Found %s', y_test.to_dict(), outputs)
            detected = self._build_events_df(outputs)
            scores.append(self._scorer(y_test, detected, X_test))

        return np.nanmean(scores)

    def tune(self, data: pd.DataFrame, anomalies: pd.DataFrame,
             train: pd.DataFrame = None, scorer: str = 'f1',
             max_evals: int = 10, post: bool = False):
        """Fit and tune the pipeline to the given data.

        Args:
            data (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
            anomalies (DataFrame):
                Ground truth anomalies, passed as `pandas.DataFrame``
                containing the start and end timestamps.
        """
        if train is None:
            train = data

        self._scorer = METRICS[scorer]
        self._mlpipeline = self._get_mlpipeline()

        if post:
            self._mlpipeline_base, self._mlpipeline_post = self._extract_pipeline()
            # train the base once
            LOGGER.debug('Training the base pipeline %s', self._mlpipeline_base.primitives)
            self._mlpipeline_base.fit(train)
        else:
            self._mlpipeline_base = self._mlpipeline
            self._mlpipeline_post = self._mlpipeline

        tunables = self._mlpipeline_post.get_tunable_hyperparameters(flat=True)
        tunables = Tunable.from_dict(tunables)

        default_score = self.scoring_function(data, anomalies)
        defaults = tunables.get_defaults()

        tuner = GPTuner(tunables)
        tuner.record(defaults, default_score)

        best_score = default_score
        best_proposal = defaults

        for iteration in range(max_evals):
            proposal = tuner.propose()
            LOGGER.debug('Scoring proposal %s: %s', iteration, proposal)
            try:
                score = self.scoring_function(data, anomalies, proposal)
                LOGGER.debug('Proposal %s scored %f', proposal, score)
                if pd.isnull(score):
                    score = np.random.rand() * 0.01
                LOGGER.debug('Recorded score %f', score)
                tuner.record(proposal, score)

            except Exception as ex:
                LOGGER.exception("Exception tuning pipeline %s",
                                 iteration, ex)

            if score > best_score:
                LOGGER.debug("New best found: {}".format(score))
                best_score = score
                best_proposal = proposal

        self._mlpipeline.set_hyperparameters(best_proposal)
        self.tuned = best_proposal
