import json
import logging

import pandas as pd
from mlblocks import MLPipeline
from mongoengine import connect

from orion import model
from orion.data import load_signal

LOGGER = logging.getLogger(__name__)


class OrionExplorer:

    def __init__(self, database, **kwargs):
        connect(database, **kwargs)

    def _list(self, model, exclude_=None, **kwargs):
        query = {
            key: value
            for key, value in kwargs.items()
            if value is not None
        }
        documents = model.find(exclude_=exclude_, **query)

        return pd.DataFrame([
            document.to_mongo()
            for document in documents
        ])

    def add_dataset(self, name, signal, satellite, location=''):
        return model.Dataset.find_or_insert(
            name=name,
            signal=signal,
            satellite=satellite,
            location=location
        )

    def get_datasets(self, name=None, signal=None, satellite=None):
        return self._list(
            model.Dataset,
            name=name,
            signal=signal,
            satellite=satellite
        )

    def get_dataset(self, name):
        return model.Dataset.last(name=name)

    def load_dataset(self, dataset):
        path_or_name = dataset.location or dataset.name
        LOGGER.info("Loading dataset %s", path_or_name)
        return load_signal(path_or_name)

    def add_pipeline(self, name, path):
        with open(path, 'r') as pipeline_file:
            pipeline_json = json.load(pipeline_file)

        return model.Pipeline.find_or_insert(
            name=name,
            mlpipeline=pipeline_json,
        )

    def get_pipelines(self, name=None):
        return self._list(
            model.Pipeline,
            dataset__name=name,
        )

    def get_pipeline(self, name):
        return model.Pipeline.last(name=name)

    def load_pipeline(self, pipeline):
        LOGGER.info("Loading pipeline %s", pipeline.name)
        return MLPipeline.from_dict(pipeline.mlpipeline)

    def get_dataruns(self, dataset=None):
        return self._list(
            model.Datarun,
            dataset=dataset
        )

    def get_events(self, datarun=None):
        return self._list(
            model.Event,
            datarun=datarun
        )
