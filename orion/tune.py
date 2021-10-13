"""Extension to Orion class"""

import pandas as pd
from btb import BTBSession

from orion.core import Orion


class OrionExtended(Orion):
    """Extension of Orion Class.

    The OrionExtended Class provides additional features of
    tunning the pipeline.
    """

    def _scorer(self, data, anomalies):
        pass

    def tune(self, data: pd.DataFrame, anomalies: pd.DataFrame,
             scorer: str, max_evals: int):
        """Fit and tune the pipeline to the given data.

        Args:
            data (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
            anomalies (DataFrame):
                Ground truth anomalies, passed as `pandas.DataFrame``
                containing the start and end timestamps.
        """
        self._mlpipeline = self._get_mlpipeline()

        scoring_function = self._scorer(data, anomalies)
        tunables = self._mlpipeline.get_tunable_hyperparameters(flat=True)

        return BTBSession(tunables, scoring_function, maximize=not self._cost)
