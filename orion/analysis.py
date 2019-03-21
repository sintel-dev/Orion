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

    LOGGER.info("Fitting the pipeline")
    pipeline.fit(train)

    LOGGER.info("Finding events")
    events = pipeline.predict(test)

    LOGGER.info("%s events found", len(events))

    events = pd.DataFrame(events, columns=['start', 'end', 'score'])
    events['start'] = events['start'].astype(int)
    events['end'] = events['end'].astype(int)

    return events


def analyze_old(explorer, dataset_name, pipeline_name):
    dataset = explorer.get_dataset(dataset_name)
    data = explorer.load_dataset(dataset)

    pipeline = explorer.get_pipeline(pipeline_name)
    mlpipeline = explorer.load_pipeline(pipeline)

    datarun = explorer.start_datarun(dataset, pipeline)

    try:
        LOGGER.info("Fitting the pipeline")
        mlpipeline.fit(data)

        LOGGER.info("Finding events")
        events = mlpipeline.predict(data)

        status = 'done'
    except Exception:
        LOGGER.exception('Error running datarun %s', datarun.id)
        events = list()
        status = 'error'

    explorer.end_datarun(datarun, events, status)

    LOGGER.info("%s events found in %s", len(events), datarun.end_time - datarun.start_time)

    return datarun
