import logging
from datetime import datetime

from orion.model import Datarun, Event

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

    datarun = Datarun.insert(
        dataset=dataset,
        pipeline=pipeline,
        start_time=start_time,
        end_time=end_time,
        events=len(events)
    )

    for start, stop, score in events:
        Event.insert(
            datarun=datarun,
            start_time=int(start),
            stop_time=int(stop),
            score=score
        )

    return datarun
