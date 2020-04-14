import logging
import os

import pandas as pd
from mlblocks import MLPipeline
import numpy as np

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

def analyze2(pipeline_input, X, test, truth=None):
    events = list()

    pipeline_path = pipeline_input
    pipeline = _load_pipeline(pipeline_path)

    training_data = X
    testing_data = test

    truth_array = np.asarray(truth)

    # fit pipeline to training data and find anomalies in testing data
    pipeline.fit(training_data, intervals=truth_array)
    anomalies = pipeline.predict(testing_data, intervals=np.array([]))

    for anomaly in anomalies:
        events.append(anomaly)

    LOGGER.debug("%s events found", len(events))

    if len(events) > 0:
        events = pd.DataFrame(np.vstack(events), columns=['start', 'end', 'score'])
        events['start'] = events['start'].astype(int)
        events['end'] = events['end'].astype(int)

    return events

def analyze_variants(pipeline_input, X, test, truth=None):
    events_set = list()

    pipeline_path = pipeline_input
    pipeline = _load_pipeline(pipeline_path)

    training_data = X
    testing_data = test

    truth_array = np.asarray(truth)

    # fit pipeline to training data and find anomalies in testing data
    pipeline.fit(training_data, intervals=truth_array)
    anomalies_set = pipeline.predict(testing_data, intervals=np.array([]))

    for anomalies in anomalies_set:
        events = list()
    
        for anomaly in anomalies:
            events.append(anomaly)

        LOGGER.debug("%s events found", len(events))

        if len(events) > 0:
            events = pd.DataFrame(np.vstack(events), columns=['start', 'end', 'score'])
            events['start'] = events['start'].astype(int)
            events['end'] = events['end'].astype(int)

        events_set.append(events)
    return events_set
