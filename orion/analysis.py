import logging
import os

import pandas as pd
from mlblocks import MLPipeline

LOGGER = logging.getLogger(__name__)


def _load_pipeline(pipeline, hyperparams=None):
    if isinstance(pipeline, str) and os.path.isfile(pipeline):
        pipeline = MLPipeline.load(pipeline)
    else:
        pipeline = MLPipeline(pipeline)

    if hyperparams is not None:
        pipeline.set_hyperparams(hyperparams)

    return pipeline


def _run_pipeline(pipeline, train, test):
    LOGGER.debug("Fitting the pipeline")
    pipeline.fit(train)

    LOGGER.debug("Finding events")
    events = pipeline.predict(test)

    LOGGER.debug("%s events found", len(events))
    return events


def _build_events_df(events):
    events = pd.DataFrame(list(events), columns=['start', 'end', 'score'])
    events['start'] = events['start'].astype(int)
    events['end'] = events['end'].astype(int)

    return events


def analyze(pipeline, train, test=None, hyperparams=None):
    if test is None:
        test = train

    pipeline = _load_pipeline(pipeline, hyperparams=None)
    events = _run_pipeline(pipeline, train, test)

    return _build_events_df(events)
