import logging
import os

import pandas as pd
from mlblocks import MLPipeline

from orion import MLBLOCKS_PIPELINES

LOGGER = logging.getLogger(__name__)


def get_available_templates(category=None):
    if isinstance(category, str):
        category = [category]

    category = category or ['verified', 'sandbox']
    templates = list()

    for filename in MLBLOCKS_PIPELINES:
        if os.path.basename(os.path.dirname(filename)) in category:
            templates.append(os.path.basename(filename))

    return templates


def _load_pipeline(pipeline, hyperparams=None):
    if isinstance(pipeline, str) and os.path.isfile(pipeline):
        pipeline = MLPipeline.load(pipeline)
    else:
        pipeline = MLPipeline(pipeline)

    if hyperparams is not None:
        pipeline.set_hyperparameters(hyperparams)

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
    events['start'] = events['start'].astype('int64')
    events['end'] = events['end'].astype('int64')

    return events


def analyze(pipeline, train, test=None, hyperparams=None):
    if test is None:
        test = train

    if not isinstance(pipeline, MLPipeline):
        pipeline = _load_pipeline(pipeline, hyperparams)

    events = _run_pipeline(pipeline, train, test)

    return _build_events_df(events)
