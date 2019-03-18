import logging

LOGGER = logging.getLogger(__name__)


def analyze(explorer, dataset_name, pipeline_name):
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
