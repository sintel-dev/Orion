import logging

import pandas as pd

from mlblocks import MLPipeline

LOGGER = logging.getLogger(__name__)


def _load_pipeline(pipeline):

    if isinstance(pipeline, MLPipeline):
        return pipeline

    if isinstance(pipeline, str):
        return MLPipeline.load(pipeline)

    if isinstance(pipeline, dict):
        return MLPipeline.from_dict(pipeline)

    raise ValueError('Invalid pipeline %s', pipeline)


def analyze(pipeline, train, test=None):

    if test is None:
        test = train

    pipeline = _load_pipeline(pipeline)

    LOGGER.debug("Fitting the pipeline")
    pipeline.fit(train)

    LOGGER.debug("Finding events")
    events = pipeline.predict(test)

    LOGGER.debug("%s events found", len(events))

    events = pd.DataFrame(events, columns=['start', 'end', 'score'])
    events['start'] = events['start'].astype(int)
    events['end'] = events['end'].astype(int)

    return events
