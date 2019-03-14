import json
import logging
from datetime import datetime

import pandas as pd
from bson import ObjectId
from bson.errors import InvalidId
from mlblocks import MLPipeline
from mongoengine import connect
from pip._internal.operations import freeze

from orion import model
from orion.data import load_signal

LOGGER = logging.getLogger(__name__)


class OrionExplorer:

    def __init__(self, user, database, **kwargs):
        self._user = user
        self.database = database
        self._db = connect(database, **kwargs)
        self._software_versions = list(freeze.freeze())

    def drop_database(self):
        self._db.drop_database(self.database)

    def _list(self, model, exclude_=None, **kwargs):
        query = {
            key: value
            for key, value in kwargs.items()
            if value is not None
        }
        documents = model.find(exclude_=exclude_, **query)

        data = pd.DataFrame([
            document.to_mongo()
            for document in documents
        ]).rename(columns={'_id': model.__name__.lower() + '_id'})

        for column in exclude_ or []:
            if column in data:
                del data[column]

        return data

    def add_dataset(self, name, signal_set, satellite_id, start_time, stop_time,
                    location=None, timestamp_column=0, value_column=1):
        return model.Dataset.find_or_insert(
            name=name,
            signal_set=signal_set,
            satellite_id=satellite_id,
            start_time=start_time,
            stop_time=stop_time,
            data_location=location,
            timestamp_column=timestamp_column,
            value_column=value_column,
            created_by=self._user
        )

    def get_datasets(self, name=None, signal=None, satellite=None):
        return self._list(
            model.Dataset,
            name=name,
            signal=signal,
            satellite=satellite
        )

    def get_dataset(self, dataset):
        try:
            _id = ObjectId(dataset)
            return model.Dataset.find(_id=_id)
        except InvalidId:
            return model.Dataset.last(name=dataset)

    def load_dataset(self, dataset):
        path_or_name = dataset.data_location or dataset.name
        LOGGER.info("Loading dataset %s", path_or_name)
        data = load_signal(path_or_name, None, dataset.timestamp_column, dataset.value_column)
        if dataset.start_time:
            data = data[data['timestamp'] >= dataset.start_time].copy()

        if dataset.stop_time:
            data = data[data['timestamp'] <= dataset.stop_time].copy()

        return data

    def add_pipeline(self, name, path):
        with open(path, 'r') as pipeline_file:
            pipeline_json = json.load(pipeline_file)

        return model.Pipeline.find_or_insert(
            name=name,
            mlpipeline=pipeline_json,
            created_by=self._user
        )

    def get_pipelines(self, name=None):
        return self._list(
            model.Pipeline,
            dataset__name=name,
        )

    def get_pipeline(self, pipeline):
        try:
            _id = ObjectId(pipeline)
            return model.Pipeline.last(_id=_id)
        except InvalidId:
            return model.Pipeline.last(name=pipeline)

    def load_pipeline(self, pipeline):
        LOGGER.info("Loading pipeline %s", pipeline.name)
        return MLPipeline.from_dict(pipeline.mlpipeline)

    def get_dataruns(self, dataset=None, pipeline=None):
        return self._list(
            model.Datarun,
            exclude_=['software_versions'],
            dataset=dataset,
            pipeline=pipeline,
        )

    def start_datarun(self, dataset, pipeline):
        return model.Datarun.insert(
            dataset=dataset,
            pipeline=pipeline,
            start_time=datetime.utcnow(),
            software_versions=self._software_versions,
            status='running',
            created_by=self._user
        )

    def end_datarun(self, datarun, events, status='done'):
        try:
            for start, stop, score in events:
                model.Event.insert(
                    datarun=datarun,
                    start_time=int(start),
                    stop_time=int(stop),
                    score=score
                )
        except Exception:
            LOGGER.exception('Error storing datarun %s events', datarun.id)
            status = 'error'

        datarun.end_time = datetime.utcnow()
        datarun.status = status
        datarun.events = len(events)
        datarun.save()

    def add_comment(self, event, text):
        model.Comment.insert(
            event=event,
            text=text,
            created_by=self._user,
        )

    def get_events(self, datarun=None):
        events = self._list(
            model.Event,
            datarun=datarun
        )

        if events.empty:
            return events

        comments = list()
        for event in events.event_id:
            events_count = model.Comment.objects(event=event).count()
            comments.append(events_count)

        events['comments'] = comments

        return events

    def get_comments(self, datarun=None, event=None):
        if event is None:
            query = {'datarun': datarun}
        else:
            query = {'id': event}

        events = self._list(
            model.Event,
            exclude_=['insert_time'],
            **query
        )
        comments = self._list(
            model.Comment,
            event__in=list(events.event_id)
        )
        comments = comments.rename(columns={'event': 'event_id'})

        return events.merge(comments, how='inner', on='event_id')
