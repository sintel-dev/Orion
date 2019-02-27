import logging
from datetime import datetime

LOGGER = logging.getLogger(__name__)


def analyze(explorer, dataset_name, pipeline_name):
    dataset = explorer.get_dataset(dataset_name)
    data = explorer.load_dataset(dataset)

    pipeline = explorer.get_pipeline(pipeline_name)
    mlpipeline = explorer.load_pipeline(pipeline)

    start_time = datetime.utcnow()
    LOGGER.info("Fitting the pipeline")
    mlpipeline.fit(data)

    LOGGER.info("Finding events")
    events = mlpipeline.predict(data)
    end_time = datetime.utcnow()

    LOGGER.info("%s events found in %s", len(events), end_time - start_time)

    LOGGER.info("Storing results")
    return explorer.add_datarun(dataset, pipeline, start_time, end_time, events)
