import json
import logging
import os
import pickle
from datetime import datetime

import pandas as pd
from bson import ObjectId
from bson.errors import InvalidId
from gridfs import GridFS
from mlblocks import MLPipeline
from mongoengine import connect, Document
from pymongo.database import Database

from orion import model
from orion.data import load_signal

LOGGER = logging.getLogger(__name__)


class OrionExplorer:
    """User interface for the Orion Database.

    This class provides a user-frienly programming interface to
    interact with the Orion database models.

    Args:
        user (str):
            Unique identifier of the user that creates this OrionExporer
            instance. This username or user ID will be used to populate
            the ``created_by`` field of all the objects created in the
            database during this session.
        database (str):
            Name of the MongoDB database to use. Defaults to ``orion``.
        **kwargs:
            Additional arguments can be passed to provide connection details
            different than the defaults for the MongoDB Database:
                * ``host``: Hostname or IP address of the MongoDB Instance.
                * ``port``: Port to which MongoDB is listening.
                * ``username``: username to authenticate with.
                * ``password``: password to authenticate with.
                * ``authentication_source``: database to authenticate against.

    Examples:
        Simples use case:
        >>> orex = OrionExplorer('my_username')

        Passing all the possible initialization arguments:
        >>> orex = OrionExplorer(
        ...      user='my_username',
        ...      database='orion',
        ...      host='localhost',
        ...      port=27017,
        ...      username='orion',
        ...      password='secret_password',
        ...      authentication_source='admin'
        ... )
    """

    def __init__(self, user, database='orion', **kwargs):
        self.user = user
        self.database = database
        self._db = connect(database, **kwargs)
        self._fs = GridFS(Database(self._db, self.database))

    def drop_database(self):
        """Drop the database.

        This method is used for development purposes and will
        most likely be removed in the future.
        """
        self._db.drop_database(self.database)

    # ####### #
    # Dataset #
    # ####### #

    def add_dataset(self, name, entity=None):
        """Add a new Dataset object to the database.

        Args:
            name (str):
                Name of the Dataset
            entity (str):
                Name or Id of the entity which this Dataset is associated to.
                Defaults to ``None``.

        Returns:
            Dataset
        """
        return model.Dataset.insert(
            name=name,
            entity=entity,
            created_by=self.user
        )

    def get_datasets(self, name=None, entity=None, created_by=None):
        """Query the Dataset collection.

        All the details about the matching datasets will be returned in
        a ``pandas.DataFrame``.

        All the arguments are optional, so a call without arguments will
        return a table with information about all the Datasets availabe.

        Args:
            name (str):
                Name of the Dataset.
            entity (str):
                Name or Id of the entity which this Dataset is associated to.
            created_by (str):
                Unique identifier of the user that created the Dataset.

        Returns:
            pandas.DataFrame
        """
        return model.Dataset.find(
            as_df_=True,
            name=name,
            entity=entity,
            created_by=created_by
        )

    def get_dataset(self, dataset=None, name=None, entity=None, created_by=None):
        """Get a Dataset object from the database.

        Empty queries are not allowed, so at least one argument needs to be passed.

        Args:
            name (str):
                Name of the dataset.
            entity (str):
                Name or Id of the entity which this dataset is associated to.
                Defaults to ``None``.

        Returns:
            Dataset
        """
        return model.Dataset.get(
            dataset=dataset,
            name=name,
            entity=entity,
            created_by=created_by
        )

    # ###### #
    # Signal #
    # ###### #

    def add_signal(self, name, dataset, data_location=None, start_time=None,
                   stop_time=None, timestamp_column=None, value_column=None):
        """Add a new Signal object to the database.

        Args:
            name (str):
                Name of the Signal.
            dataset (Dataset or ObjectID or str):
                Dataset object which the created Signal belongs to or the
                corresponding ObjectId.
            data_location (str):
                Path to the CSV containing the Signal data. If the signal is
                one of the signals provided by Orion, this can be omitted and
                the signal will be loaded based on the signal name.
            start_time (int):
                Optional. Minimum timestamp to use for this signal. If not provided
                this defaults to the minimum timestamp found in the signal data.
            stop_time (int):
                Optional. Maximum timestamp to use for this signal. If not provided
                this defaults to the maximum timestamp found in the signal data.
            timestamp_column (int):
                Optional. Index of the timestamp column.
            value_column (int):
                Optional. Index of the value column.

        Returns:
            Signal
        """

        data_location = data_location or name
        data = load_signal(data_location, None, timestamp_column, value_column)
        timestamps = data['timestamp']
        if not start_time:
            start_time = timestamps.min()

        if not stop_time:
            stop_time = timestamps.max()

        dataset = self.get_dataset(dataset)

        return model.Signal.insert(
            name=name,
            dataset=dataset,
            start_time=start_time,
            stop_time=stop_time,
            data_location=data_location,
            timestamp_column=timestamp_column,
            value_column=value_column,
            created_by=self.user
        )

    def get_signals(self, name=None, dataset=None, created_by=None):
        return model.Signal.find(
            as_df_=True,
            name=name,
            dataset=dataset,
            created_by=created_by
        )

    def get_signal(self, signal=None, name=None, dataset=None, created_by=None):
        return model.Signal.get(
            signal=signal,
            name=name,
            dataset=dataset,
            created_by=created_by
        )

    # ######## #
    # Template #
    # ######## #

    def add_template(self, name, template):
        if isinstance(template, str) and os.path.isfile(template):
            with open(template, 'r') as f:
                template = json.load(f)

        pipeline_dict = MLPipeline(template).to_dict()

        template = model.Template.insert(
            name=name,
            json=pipeline_dict,
            created_by=self.user
        )
        model.Pipeline.insert(
            name=name,
            template=template,
            json=pipeline_dict
        )

        return template

    def get_templates(self, name=None, created_by=None):
        return model.Template.find(
            as_df_=True,
            name=name,
            created_by=created_by,
            exclude_=['json']
        )

    def get_template(self, template=None, name=None, created_by=None):
        return model.Template.get(
            template=template,
            name=name,
            created_by=created_by
        )

    # ######## #
    # Pipeline #
    # ######## #

    def add_pipeline(self, name, template, hyperparameters):
        pipeline = self.get_template(template).load()
        if isinstance(hyperparameters, str):
            with open(hyperparameters, 'r') as f:
                hyperparameters = json.load(f)

        pipeline.set_hyperparameters(hyperparameters)

        return model.Pipeline.insert(
            name=name,
            template=template,
            json=pipeline.to_dict(),
            created_by=self.user
        )

    def get_pipelines(self, name=None, template=None, created_by=None):
        return model.Pipeline.find(
            as_df_=True,
            name=name,
            template=template,
            created_by=created_by,
            exclude_=['json']
        )

    def get_pipeline(self, pipeline=None, name=None, template=None, created_by=None):
        return model.Pipeline.get(
            pipeline=pipeline,
            name=name,
            template=template,
            created_by=created_by,
        )

    # ########## #
    # Experiment #
    # ########## #

    def add_experiment(self, name, template, dataset, signals=None, project=None):
        signals = signals or dataset.signals
        return model.Experiment.insert(
            name=name,
            project=project,
            template=template,
            dataset=dataset,
            signals=signals,
            created_by=self.user
        )

    def get_experiments(self, name=None, project=None, template=None,
                        dataset=None, signals=None, created_by=None):
        return model.Experiment.find(
            as_df_=True,
            name=name,
            project=project,
            template=template,
            dataset=dataset,
            signals=signals,
            created_by=created_by,
        )

    def get_experiment(self, experiment=None, name=None, project=None, template=None,
                       dataset=None, signals=None, created_by=None):
        return model.Experiment.get(
            experiment=experiment,
            name=name,
            project=project,
            template=template,
            dataset=dataset,
            signals=signals,
            created_by=created_by,
        )

    # ####### #
    # Datarun #
    # ####### #

    def add_datarun(self, experiment, pipeline):
        return model.Datarun.insert(
            experiment=experiment,
            pipeline=pipeline,
        )

    def get_dataruns(self, experiment=None, pipeline=None, status=None):
        return model.Datarun.find(
            as_df_=True,
            experiment=experiment,
            pipeline=pipeline,
            status=status,
            exclude_=['software_versions'],
        )

    def get_datarun(self, datarun=None, experiment=None, pipeline=None, status=None):
        return model.Datarun.get(
            experiment=experiment,
            pipeline=pipeline,
            status=status,
        )

    # ##### #
    # Event #
    # ##### #

    def add_event(self, start_time, stop_time, severity=None,
                  signalrun=None, signal=None, source=None):
        if signal is None and signalrun is None:
            raise ValueError('An Event must be associated to either a Signalrun or a Signal')

        return model.Event.insert(
            signalrun=signalrun,
            signal=signal or signalrun.signal,
            start_time=int(start_time),
            stop_time=int(stop_time),
            severity=severity,
            source=source,
        )

    def get_events(self, signalrun=None, signal=None, source=None):
        return model.Event.find(
            as_df_=True,
            signalrun=signalrun,
            signal=signal,
            source=source,
        )

    def get_event(self, event=None, signalrun=None, signal=None, source=None):
        return model.Event.get(
            event=event,
            signalrun=signalrun,
            signal=signal,
            source=source,
        )

    # ######### #
    # Signalrun #
    # ######### #

    def add_signalrun(self, datarun, signal):
        return model.Signalrun.insert(
            experiment=experiment,
            pipeline=pipeline,
        )

    def get_signalruns(self, datarun=None, signal=None, status=None):
        return model.Signalrun.find(
            as_df_=True,
            datarun=datarun,
            signal=signal,
            status=status,
        )

    def get_signalrun(self, signalrun=None, datarun=None, signal=None, status=None):
        return model.Signalrun.get(
            signalrun=signalrun,
            datarun=datarun,
            signal=signal,
            status=status,
        )

    # ########## #
    # Annotation #
    # ########## #

    def add_annotation(self, event, tag=None, comment=None):
        return model.Annotation.insert(
            event=event,
            tag=tag,
            comment=comment,
        )

    def get_annotations(self, event=None, tag=None, comment=None, created_by=None):
        return model.Annotation.find(
            as_df_=True,
            event=event,
            tag=tag,
            created_by=created_by,
        )

    def get_annotation(self, annotation=None, event=None, tag=None, created_by=None):
        return model.Annotation.get(
            annotation=annotation,
            event=event,
            tag=tag,
            created_by=created_by,
        )
