"""Orion Database Schema.

This module contains the classes that define the Orion Database Schema:
    * Dataset
    * Signal
    * Template
    * Pipeline
    * Experiment
    * Datarun
    * Signalrun
    * Event
    * EventInteraction
    * Annotation
"""

import logging
from datetime import datetime

from mlblocks import MLPipeline
from mongoengine import CASCADE, fields
from pip._internal.operations import freeze

from orion.data import load_signal
from orion.db.base import OrionDocument, PipelineField, Status

LOGGER = logging.getLogger(__name__)


class Dataset(OrionDocument):
    name = fields.StringField(required=True)
    entity = fields.StringField()
    created_by = fields.StringField()  # New

    unique_key_fields = ['name', 'entity']

    @property
    def signals(self):
        return Signal.find(dataset=self)


class Signal(OrionDocument):
    name = fields.StringField(required=True)
    dataset = fields.ReferenceField(Dataset, reverse_delete_rule=CASCADE)
    data_location = fields.StringField()
    start_time = fields.IntField()
    stop_time = fields.IntField()
    timestamp_column = fields.IntField()
    value_column = fields.IntField()
    created_by = fields.StringField()

    unique_key_fields = ['name', 'dataset']

    def load(self):
        data = load_signal(self.data_location, None, self.timestamp_column, self.value_column)
        if self.start_time:
            data = data[data['timestamp'] >= self.start_time].copy()

        if self.stop_time:
            data = data[data['timestamp'] <= self.stop_time].copy()

        return data


class Template(OrionDocument):   # New - Renamed from Pipeline
    name = fields.StringField(required=True)
    json = PipelineField(required=True)   # New - Renamed from mlpipeline
    created_by = fields.StringField()

    unique_key_fields = ['name']

    def load(self):
        return MLPipeline(self.json)

    @property
    def pipelines(self):
        return Pipeline.find(template=self)


class Pipeline(OrionDocument):   # New
    name = fields.StringField(required=True)
    template = fields.ReferenceField(Template, reverse_delete_rule=CASCADE)  # New
    json = PipelineField(required=True)   # New
    created_by = fields.StringField()

    unique_key_fields = ['name', 'template']

    def load(self):
        return MLPipeline(self.json)


class Experiment(OrionDocument):
    name = fields.StringField(required=True)   # New
    project = fields.StringField()
    template = fields.ReferenceField(Template, reverse_delete_rule=CASCADE)  # New
    dataset = fields.ReferenceField(Dataset, reverse_delete_rule=CASCADE)
    signals = fields.ListField(fields.ReferenceField(Signal, reverse_delete_rule=CASCADE))   # New
    created_by = fields.StringField()

    unique_key_fields = ['name', 'project']

    @property
    def dataruns(self):
        return Datarun.find(experiment=self)


class Datarun(OrionDocument, Status):
    experiment = fields.ReferenceField(Experiment, reverse_delete_rule=CASCADE)
    pipeline = fields.ReferenceField(Pipeline, reverse_delete_rule=CASCADE)
    start_time = fields.DateTimeField()
    end_time = fields.DateTimeField()
    software_versions = fields.ListField(fields.StringField())
    budget_type = fields.StringField()
    budget_amount = fields.IntField()
    events = fields.IntField()

    _software_versions = list(freeze.freeze())

    @property
    def signalruns(self):
        return Signalrun.find(datarun=self)

    def start(self):
        self.start_time = datetime.utcnow()
        self.status = self.STATUS_RUNNING
        self.software_versions = self._software_versions
        self.save()

    def end(self, status):
        self.end_time = datetime.utcnow()
        self.status = status
        self.events = Event.find(signalrun__in=self.signalruns).count()
        self.save()


class Signalrun(OrionDocument, Status):   # New
    datarun = fields.ReferenceField(Datarun, reverse_delete_rule=CASCADE)
    signal = fields.ReferenceField(Signal, reverse_delete_rule=CASCADE)
    start_time = fields.DateTimeField()
    end_time = fields.DateTimeField()
    model_location = fields.StringField()
    metrics_location = fields.StringField()
    events = fields.IntField()

    def start(self):
        self.start_time = datetime.utcnow()
        self.status = self.STATUS_RUNNING
        self.save()

    def end(self, status, events):
        try:
            for start_time, stop_time, severity in events:
                Event.insert(
                    signalrun=self,
                    signal=self.signal,
                    start_time=start_time,
                    stop_time=stop_time,
                    severity=severity
                )
        except Exception:
            LOGGER.exception('Error storing signalrun %s events', self.id)
            status = self.STATUS_ERROR

        self.end_time = datetime.utcnow()
        self.status = status
        self.events = len(events)
        self.save()


class Event(OrionDocument):
    signalrun = fields.ReferenceField(Signalrun, reverse_delete_rule=CASCADE)   # New - renamed
    signal = fields.ReferenceField(Signal, reverse_delete_rule=CASCADE)
    start_time = fields.IntField(required=True)
    stop_time = fields.IntField(required=True)
    severity = fields.FloatField()   # New - renamed
    source = fields.StringField(choices=['orion', 'shape matching', 'manually created'])


class EventInteraction(OrionDocument):   # New
    event = fields.ReferenceField(Event, reverse_delete_rule=CASCADE)
    action = fields.StringField()
    start_time = fields.IntField(required=True)
    stop_time = fields.IntField(required=True)
    created_by = fields.StringField()


class Annotation(OrionDocument):   # New - Renamed
    event = fields.ReferenceField(Event, reverse_delete_rule=CASCADE)
    tag = fields.StringField()   # New
    comment = fields.StringField()    # New - renamed
    created_by = fields.StringField()
