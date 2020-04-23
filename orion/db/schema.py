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
    """Dataset object.

    A **Dataset** represents a group of Signals that are grouped together under a
    common name, which is usually defined by an external entity.
    """
    name = fields.StringField(required=True)
    entity = fields.StringField()
    created_by = fields.StringField()

    unique_key_fields = ['name', 'entity']

    @property
    def signals(self):
        return Signal.find(dataset=self)


class Signal(OrionDocument):
    """Signal object.

    A Signal belongs to a Dataset and contains all the required details to be
    able to load the observations from a timeseries signal, as well as some
    metadata about it, such as the minimum and maximum timestamps that have to
    be used.
    """
    name = fields.StringField(required=True)
    dataset = fields.ReferenceField(Dataset, reverse_delete_rule=CASCADE)
    start_time = fields.IntField()
    stop_time = fields.IntField()
    data_location = fields.StringField()
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


class Template(OrionDocument):
    """Template object.

    A **Template** represents an MLPipeline template from which later on
    Pipelines will be derived to run on Experiments with different hyperparameter
    values.

    The Template includes the complete JSON specification of the MLPipeline it
    represents.
    """
    name = fields.StringField(required=True)
    json = PipelineField(required=True)
    created_by = fields.StringField()

    unique_key_fields = ['name']

    def load(self):
        """Load this Template as an MLPipeline.

        Returns:
            MLPipeline
        """
        return MLPipeline(self.json)

    @property
    def pipelines(self):
        return Pipeline.find(template=self)


class Pipeline(OrionDocument):
    """Pipeline object.

    A **Pipeline** represents an MLPipeline object.
    It is derived from a Template by setting a specific set of
    hyperparameter values.
    """
    name = fields.StringField(required=True)
    template = fields.ReferenceField(Template, reverse_delete_rule=CASCADE)
    json = PipelineField(required=True)
    created_by = fields.StringField()

    unique_key_fields = ['name', 'template']

    def load(self):
        """Load this Pipeline as an MLPipeline.

        Returns:
            MLPipeline
        """
        return MLPipeline(self.json)


class Experiment(OrionDocument):
    """Experiment object.

    An Experiment is associated with a Dataset, a subset of its Signals and a Template.
    It represents a collection of Dataruns, executions of Pipelines generated from the
    Experiment Template over its Signals Set.
    """
    name = fields.StringField(required=True)
    project = fields.StringField()
    template = fields.ReferenceField(Template, reverse_delete_rule=CASCADE)
    dataset = fields.ReferenceField(Dataset, reverse_delete_rule=CASCADE)
    signals = fields.ListField(fields.ReferenceField(Signal, reverse_delete_rule=CASCADE))
    created_by = fields.StringField()

    unique_key_fields = ['name', 'project']

    @property
    def dataruns(self):
        return Datarun.find(experiment=self)


class Datarun(OrionDocument, Status):
    """Datarun object.

    The Datarun object represents a single execution of a Pipeline over a set
    of Signals, within the context of an Experiment.

    It contains all the information about the environment and context where this
    execution took place, which potentially allows to later on reproduce the results
    in a new environment.

    It also contains information about whether the execution was successful or not,
    when it started and ended, and the number of events that were found in this experiment.
    """
    experiment = fields.ReferenceField(Experiment, reverse_delete_rule=CASCADE)
    pipeline = fields.ReferenceField(Pipeline, reverse_delete_rule=CASCADE)
    start_time = fields.DateTimeField()
    end_time = fields.DateTimeField()
    software_versions = fields.ListField(fields.StringField())
    budget_type = fields.StringField()
    budget_amount = fields.IntField()
    num_events = fields.IntField()

    _software_versions = list(freeze.freeze())

    @property
    def signalruns(self):
        return Signalrun.find(datarun=self)

    def start(self):
        """Mark this Datarun as started on DB.

        The ``start_time`` will be set to ``datetime.utcnow()``,
        the ``status`` will be set to RUNNING and the software
        versions will be captured.
        """
        self.start_time = datetime.utcnow()
        self.status = self.STATUS_RUNNING
        self.software_versions = self._software_versions
        self.save()

    def end(self, status):
        """Mark this Datarun as ended on DB.

        The ``end_time`` will be set to ``datetime.utcnow()``, the ``status``
        will be set to the given value, and the ``num_events`` field will be
        populated with the sum of the events detected by the children Signalruns.
        """
        self.end_time = datetime.utcnow()
        self.status = status
        self.num_events = Event.find(signalrun__in=self.signalruns).count()
        self.save()


class Signalrun(OrionDocument, Status):
    """Signalrun object.

    The Signalrun object represents a single executions of a Pipeline on a Signal
    within a Datarun.

    It contains information about whether the execution was successful or not, when
    it started and ended, the number of events that were found by the Pipeline, and
    where the model and metrics are stored.
    """
    datarun = fields.ReferenceField(Datarun, reverse_delete_rule=CASCADE)
    signal = fields.ReferenceField(Signal, reverse_delete_rule=CASCADE)
    start_time = fields.DateTimeField()
    end_time = fields.DateTimeField()
    model_location = fields.StringField()
    metrics_location = fields.StringField()
    num_events = fields.IntField(default=0)

    @property
    def events(self):
        return Event.find(signalrun=self)

    def start(self):
        """Mark this Signalrun as started on DB.

        The ``start_time`` will be set to ``datetime.utcnow()``,
        the ``status`` will be set to RUNNING.
        """
        self.start_time = datetime.utcnow()
        self.status = self.STATUS_RUNNING
        self.save()

    def end(self, status, events):
        """Mark this Signalrun as ended on DB.

        The ``end_time`` will be set to ``datetime.utcnow()``, the ``status``
        will be set to the given value, and the given events will be inserted
        into the Database.
        """
        try:
            for start_time, stop_time, severity in events:
                Event.insert(
                    signalrun=self,
                    signal=self.signal,
                    start_time=start_time,
                    stop_time=stop_time,
                    severity=severity,
                    source=Event.SOURCE_ORION,
                )
        except Exception:
            LOGGER.exception('Error storing signalrun %s events', self.id)
            status = self.STATUS_ERROR

        self.end_time = datetime.utcnow()
        self.status = status
        self.num_events = len(events)
        self.save()


class Event(OrionDocument):
    """Event object.

    An Event represents an anomalous period on a Signal.
    It is descrived by start and stop times and, optionally, a severity score.

    It is always associated to a Signal and, optionally, to a Signalrun.
    """
    SOURCE_ORION = 'ORION'
    SOURCE_SHAPE_MATCHING = 'SHAPE_MATCHING'
    SOURCE_MANUALLY_CREATED = 'MANUALLY_CREATED'
    SOURCE_CHOICES = (SOURCE_ORION, SOURCE_SHAPE_MATCHING, SOURCE_MANUALLY_CREATED)

    signalrun = fields.ReferenceField(Signalrun, reverse_delete_rule=CASCADE)
    signal = fields.ReferenceField(Signal, reverse_delete_rule=CASCADE)
    start_time = fields.IntField(required=True)
    stop_time = fields.IntField(required=True)
    severity = fields.FloatField()
    source = fields.StringField(required=True, choices=SOURCE_CHOICES)
    num_annotations = fields.IntField(default=0)

    @property
    def annotations(self):
        return Annotation.find(event=self)

    @property
    def event_interactions(self):
        return EventInteraction.find(event=self)

    @property
    def latest_event_interaction(self):
        return self.event_interactions.last()


class EventInteraction(OrionDocument):
    """EventInteraction object.

    The EventInteraction object represents an interaction of a user with an
    Event and includes information about the assocaiated Event, the action
    performed and, if changed, the specified start and stop times.
    """
    event = fields.ReferenceField(Event, reverse_delete_rule=CASCADE)
    action = fields.StringField()
    start_time = fields.IntField(required=True)
    stop_time = fields.IntField(required=True)
    created_by = fields.StringField()


class Annotation(OrionDocument):
    """Annotation object.

    User Annotation related to an event. The Annotations consist on a tag and
    a free text field where the user can insert comments about the Event.
    """
    event = fields.ReferenceField(Event, reverse_delete_rule=CASCADE)
    tag = fields.StringField()
    comment = fields.StringField()
    created_by = fields.StringField()

    def save(self):
        super().save()
        self.event.num_annotations = self.event.annotations.count()
        self.event.save()
