"""Orion Database Schema.

This module contains some utility functions and classes that assist
on the usage of MongoEngine to define a DB Schema.
"""

import copy
from datetime import datetime

import pandas as pd
from bson import ObjectId
from mongoengine import Document, fields
from mongoengine.base.metaclasses import TopLevelDocumentMetaclass


def walk(document, transform):
    if not isinstance(document, dict):
        return document

    new_doc = dict()
    for key, value in document.items():
        if isinstance(value, dict):
            value = walk(value, transform)
        elif isinstance(value, list):
            value = [walk(v, transform) for v in value]

        new_key, new_value = transform(key, value)
        new_doc[new_key] = new_value

    return new_doc


def remove_dots(document):
    return walk(document, lambda key, value: (key.replace('.', '-'), value))


def restore_dots(document):
    return walk(document, lambda key, value: (key.replace('-', '.'), value))


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
            for column in exclude_:
                if column in df:
                    del df[column]

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
