"""Extension to Orion class"""
import logging

import numpy as np
import pandas as pd
from btb.tuning import GPTuner, Tunable
from sklearn.model_selection import KFold

from orion.core import Orion
from orion.evaluation import CONTEXTUAL_METRICS as METRICS

LOGGER = logging.getLogger(__name__)


class OrionTuner(Orion):
    """Extension of Orion Class.

    The OrionExtended Class provides additional features of
    tunning the pipeline.
    """
    _time_column = 'timestamp'
    _label_column = 'label'
    _columns = None
    _scorer = None
    tuned = None

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

    def scoring_function(self, data, anomalies, hyperparameters=None, verbose=False):
        if hyperparameters:
            self._mlpipeline.set_hyperparameters(hyperparameters)

        scores = []
        for X_train, y_train, X_test, y_test in self._cv_split(data, anomalies, 3):
            self.fit(X_train, verbose=verbose)
            detected = self.detect(X_test, verbose=verbose)
            scores.append(self._scorer(y_test, detected, X_test))

        return np.mean(scores)

    def tune(self, data: pd.DataFrame, anomalies: pd.DataFrame,
             scorer: str = 'f1', max_evals: int = 10, verbose: bool = False):
        """Fit and tune the pipeline to the given data.

        Args:
            data (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
            anomalies (DataFrame):
                Ground truth anomalies, passed as `pandas.DataFrame``
                containing the start and end timestamps.
        """

        self._scorer = METRICS[scorer]
        self._mlpipeline = self._get_mlpipeline()

        tunables = self._mlpipeline.get_tunable_hyperparameters(flat=True)
        tunables = Tunable.from_dict(tunables)

        default_score = self.scoring_function(data, anomalies, verbose=verbose)
        defaults = tunables.get_defaults()

        tuner = GPTuner(tunables)
        tuner.record(defaults, default_score)

        best_score = default_score
        best_proposal = defaults

        for iteration in range(max_evals):
            proposal = tuner.propose()
            LOGGER.debug('Scoring proposal %s: %s', iteration, proposal)
            try:
                score = self.scoring_function(data, anomalies, proposal, verbose=verbose)
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
