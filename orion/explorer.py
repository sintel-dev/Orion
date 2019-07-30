import json
import logging
from datetime import datetime

import pandas as pd
from bson import ObjectId
from bson.errors import InvalidId
from mongoengine import connect
from pip._internal.operations import freeze

from mlblocks import MLPipeline
from orion import model
from orion.data import load_signal

LOGGER = logging.getLogger(__name__)


class OrionExplorer:

    def __init__(self, database='orion', **kwargs):
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

    def add_dataset(self, name, entity_id=None):
        return model.Dataset.find_or_insert(
            name=name,
            entity_id=entity_id
        )

    def get_datasets(self, name=None, entity_id=None):
        return self._list(
            model.Dataset,
            name=name,
            entity_id=entity_id
        )

    def get_dataset(self, dataset):
        try:
            _id = ObjectId(dataset)
            return model.Dataset.last(id=_id)
        except InvalidId:
            found_dataset = model.Dataset.last(name=dataset)
            if found_dataset is None:
                raise ValueError('Dataset not found: {}'.format(dataset))
            else:
                return found_dataset

    def add_signal(self, name, dataset, start_time=None, stop_time=None, location=None,
                   timestamp_column=None, value_column=None, user_id=None):

        location = location or name
        data = load_signal(location, None, timestamp_column, value_column)
        timestamps = data['timestamp']
        if not start_time:
            start_time = timestamps.min()

        if not stop_time:
            stop_time = timestamps.max()

        dataset = self.get_dataset(dataset)

        return model.Signal.find_or_insert(
            name=name,
            dataset=dataset,
            start_time=start_time,
            stop_time=stop_time,
            data_location=location,
            timestamp_column=timestamp_column,
            value_column=value_column,
            created_by=user_id
        )

    def get_signals(self, name=None, dataset=None):
        return self._list(
            model.Signal,
            name=name,
            dataset=dataset
        )

    def load_signal(self, signal):
        path_or_name = signal.data_location or signal.name
        LOGGER.info("Loading dataset %s", path_or_name)
        data = load_signal(path_or_name, None, signal.timestamp_column, signal.value_column)
        if signal.start_time:
            data = data[data['timestamp'] >= signal.start_time].copy()

        if signal.stop_time:
            data = data[data['timestamp'] <= signal.stop_time].copy()

        return data

    def add_pipeline(self, name, path, user_id=None):
        with open(path, 'r') as pipeline_file:
            pipeline_json = json.load(pipeline_file)

        return model.Pipeline.find_or_insert(
            name=name,
            mlpipeline=pipeline_json,
            created_by=user_id
        )

    def get_pipelines(self, name=None):
        return self._list(
            model.Pipeline,
            dataset__name=name,
        )

    def get_pipeline(self, pipeline):
        try:
            _id = ObjectId(pipeline)
            return model.Pipeline.last(id=_id)
        except InvalidId:
            found_pipeline = model.Pipeline.last(name=pipeline)
            if found_pipeline is None:
                raise ValueError('Pipeline not found: {}'.format(pipeline))
            else:
                return found_pipeline

    def load_pipeline(self, pipeline):
        LOGGER.info("Loading pipeline %s", pipeline.name)
        return MLPipeline.from_dict(pipeline.mlpipeline)

    def run_experiment(self, name, project, pipeline, dataset, user_id=None):
        project = project
        pipeline = self.get_pipeline(pipeline)
        dataset = self.get_dataset(dataset)

        experiment = model.Experiment.find_or_insert(
            name=name,
            project=project,
            pipeline=pipeline,
            dataset=dataset,
            created_by=user_id
        )

        mlpipeline = self.load_pipeline(pipeline)
        signals = model.Signal.find(dataset=dataset.id)

        for signal in signals:

            data = self.load_signal(signal)
            datarun = self.start_datarun(experiment, signal)

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

            self.end_datarun(datarun, events, status)

            LOGGER.info("%s events found in %s", len(events),
                        datarun.end_time - datarun.start_time)

        return experiment

    def get_experiments(self, name=None):
        return self._list(
            model.Experiment,
            name=name
        )

    def get_dataruns(self, experiment=None):
        return self._list(
            model.Datarun,
            exclude_=['software_versions'],
            experiment=experiment
        )

    def start_datarun(self, experiment, signal):
        return model.Datarun.insert(
            experiment=experiment,
            signal=signal,
            start_time=datetime.utcnow(),
            software_versions=self._software_versions,
            status='running'
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

    def add_comment(self, event, text, user_id):
        model.Comment.insert(
            event=event,
            text=text,
            created_by=user_id,
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

        if events.empty:
            return pd.DataFrame()

        comments = self._list(
            model.Comment,
            event__in=list(events.event_id)
        )
        comments = comments.rename(columns={'event': 'event_id'})

        return events.merge(comments, how='inner', on='event_id')
