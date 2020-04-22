"""Orion Database Schema.

This module contains some utility functions and functions that assist
on the usage of MongoEngine to define a DB Schema.

Afterwards, there are the classes the define the actual Orion Database
Schema:
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

import copy
from datetime import datetime

import pandas as pd
from bson import ObjectId
from mlblocks import MLPipeline
from mongoengine import CASCADE, Document, fields
from mongoengine.base.metaclasses import TopLevelDocumentMetaclass
from pip._internal.operations import freeze

from orion.data import load_signal
from orion.utils import remove_dots, restore_dots


def _merge_meta(base, child):
    """Merge the base and the child meta attributes.

    List entries, such as ``indexes`` are concatenated.
    ``abstract`` value is set to ``True`` only if defined as such
    in the child class.

    Args:
        base (dict):
            ``meta`` attribute from the base class.
        child (dict):
            ``meta`` attribute from the child class.

    Returns:
        dict:
            Merged metadata.
    """
    base = copy.deepcopy(base)
    child.setdefault('abstract', False)
    for key, value in child.items():
        if isinstance(value, list):
            base.setdefault(key, []).extend(value)
        else:
            base[key] = value

    return base


class OrionMeta(TopLevelDocumentMetaclass):
    """Metaclass for the OrionDocument class.

    It ensures that the ``meta`` attribute from the OrionDocument
    parent class is used even if the child class defines a new one
    by merging both of them together.
    """

    def __new__(mcs, name, bases, attrs):
        if 'meta' in attrs:
            meta = attrs['meta']
            for base in bases:
                if base is not Document and hasattr(base, '_meta'):
                    meta = _merge_meta(base._meta, meta)

            attrs['meta'] = meta

        if 'unique_key_fields' in attrs:
            indexes = attrs.setdefault('meta', {}).setdefault('indexes', [])
            indexes.append({
                'fields': attrs['unique_key_fields'],
                'unique': True,
                'sparse': True,
            })

        return super().__new__(mcs, name, bases, attrs)


class OrionDocument(Document, metaclass=OrionMeta):
    """Parent class for all the Document classes in Orion.

    This class defines a few defaults, such as the ``instert_time`` field
    and index, as well as a few utility methods to ease the interaction
    with the database models.
    """

    insert_time = fields.DateTimeField(default=datetime.utcnow)

    meta = {
        'indexes': [
            '$insert_time',
        ],
        'abstract': True,
        'index_background': True,
        'auto_create_index': True,
    }

    @staticmethod
    def _get_id(obj):
        if isinstance(obj, ObjectId):
            return obj
        elif isinstance(obj, Document):
            return obj.id
        elif obj:
            return ObjectId(obj)

    @classmethod
    def find(cls, as_df_=False, only_=None, exclude_=None, **kwargs):
        name = cls.__name__.lower()
        if name in kwargs:
            kwargs['id'] = cls._get_id(kwargs.pop(name))

        query = {
            key: value
            for key, value in kwargs.items()
            if value is not None
        }
        cursor = cls.objects(**query)
        if only_:
            cursor = cursor.only(*only_)

        if exclude_:
            cursor = cursor.exclude(*exclude_)

        if not as_df_:
            return cursor

        df = pd.DataFrame([
            document.to_mongo()
            for document in cursor
        ]).rename(columns={'_id': name + '_id'})

        if exclude_:
            df = df.drop(exclude_, axis=1)

        return df

    @classmethod
    def get(cls, **kwargs):
        query = {
            key: value
            for key, value in kwargs.items()
            if value is not None
        }
        if not query:
            raise ValueError('Empty queries not supported')

        cursor = cls.find(as_df_=False, **query)
        if not cursor:
            raise ValueError('No {} found for query {}'.format(cls.__name__, query))
        elif cursor.count() > 1:
            raise ValueError('Multiple {}s found for query {}'.format(cls.__name__, query))

        return cursor.first()

    @classmethod
    def find_one(cls, **kwargs):
        return cls.find(**kwargs).first()

    @classmethod
    def last(cls, **kwargs):
        return cls.find(**kwargs).order_by('-insert_time').first()

    @classmethod
    def insert(cls, **kwargs):
        document = cls(**kwargs)
        document.save()

        return document

    @classmethod
    def find_or_insert(cls, **kwargs):
        document = cls.find_one(**kwargs)
        if document is None:
            document = cls.insert(**kwargs)

        return document


class Status:
    """Mixin that adds a status field and a method to check its live value."""

    status = fields.StringField()

    STATUS_PENDING = 'PENDING'
    STATUS_RUNNING = 'RUNNING'
    STATUS_SUCCESS = 'SUCCESS'
    STATUS_ERRORED = 'ERRORED'

    def get_status(self):
        self.reload()
        return self.status


def key_has_dollar(d):
    """Recursively check if any key in a dict contains a dollar sign."""
    for k, v in d.items():
        if k.startswith('$') or (isinstance(v, dict) and key_has_dollar(v)):
            return True


class PipelineField(fields.DictField):

    def to_mongo(self, value, use_db_field=True, fields=None):
        value = remove_dots(value)
        return super().to_mongo(value, use_db_field, fields)

    def to_python(self, value):
        value = restore_dots(value)
        return super().to_python(value)

    def validate(self, value):
        """Make sure that a list of valid fields is being used."""
        if not isinstance(value, dict):
            self.error('Only dictionaries may be used in a PipelineField')

        if fields.key_not_string(value):
            msg = ('Invalid dictionary key - documents must '
                   'have only string keys')
            self.error(msg)

        if key_has_dollar(value):
            self.error('Invalid dictionary key name - keys may not start with '
                       '"$" character')

        super(fields.DictField, self).validate(value)


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
