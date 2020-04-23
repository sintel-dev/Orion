"""Orion DB Explorer model.

This model defines the ``orion.db.explorer.OrionDBExplorer``, which provides
a simple programatic access to creating and reading objects in the Orion Database.
"""
import json
import logging
import os

from gridfs import GridFS
from mlblocks import MLPipeline
from mongoengine import connect
from mongoengine.errors import NotUniqueError
from pymongo.database import Database

from orion.data import load_signal
from orion.db import schema

LOGGER = logging.getLogger(__name__)


class OrionDBExplorer:
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
        mongodb_config (dict or str):
            A dict or a path to JSON file with additional arguments can be
            passed to provide connection details different than the defaults
            for the MongoDB Database:
                * ``host``: Hostname or IP address of the MongoDB Instance.
                * ``port``: Port to which MongoDB is listening.
                * ``username``: username to authenticate with.
                * ``password``: password to authenticate with.
                * ``authentication_source``: database to authenticate against.

    Examples:
        Simples use case:
        >>> orex = OrionExplorer('my_username')

        Passing a path to a JSON file with connection details.
        >>> orex = OrionExplorer(
        ...      user='my_username',
        ...      database='orion',
        ...      mongodb_config='/path/to/my/mongodb_config.json',
        ... )

        Passing all the possible initialization arguments as a dict:
        >>> mongodb_config = {
        ...      'host': 'localhost',
        ...      'port': 27017,
        ...      'username': 'orion',
        ...      'password': 'secret_password',
        ...      'authentication_source': 'admin',
        ... }
        >>> orex = OrionExplorer(
        ...      user='my_username',
        ...      database='orion',
        ...      mongodb_config=mongodb_config
        ... )
    """

    def __init__(self, user, database='orion', mongodb_config=None):
        """Initiaize this OrionDBExplorer.

        Args:
            user (str):
                Unique identifier of the user that creates this OrionExporer
                instance. This username or user ID will be used to populate
                the ``created_by`` field of all the objects created in the
                database during this session.
            database (str):
                Name of the MongoDB database to use. Defaults to ``orion``.
            mongodb_config (dict or str):
                A dict or a path to JSON file with additional arguments can be
                passed to provide connection details different than the defaults
                for the MongoDB Database:
                    * ``host``: Hostname or IP address of the MongoDB Instance.
                    * ``port``: Port to which MongoDB is listening.
                    * ``username``: username to authenticate with.
                    * ``password``: password to authenticate with.
                    * ``authentication_source``: database to authenticate against.
        """
        if mongodb_config is None:
            mongodb_config = dict()
        elif isinstance(mongodb_config, str):
            with open(mongodb_config) as config_file:
                mongodb_config = json.load(config_file)
        elif isinstance(mongodb_config, dict):
            mongodb_config = mongodb_config.copy()

        self.user = user
        self.database = mongodb_config.pop('database', database)
        self._db = connect(database, **mongodb_config)
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

        The Dataset needs to be given a name and, optionally, an identitifier,
        name or ID, of the entity which produced the Dataset.

        Args:
            name (str):
                Name of the Dataset
            entity (str):
                Name or Id of the entity which this Dataset is associated to.
                Defaults to ``None``.

        Raises:
            NotUniqueError:
                If a Dataset with the same name and entity values already exists.

        Returns:
            Dataset
        """
        return schema.Dataset.insert(
            name=name,
            entity=entity,
            created_by=self.user
        )

    def get_datasets(self, name=None, entity=None, created_by=None):
        """Query the Datasets collection.

        All the details about the matching Datasets will be returned in
        a ``pandas.DataFrame``.

        All the arguments are optional, so a call without arguments will
        return a table with information about all the Datasets availabe.

        Args:
            name (str):
                Name of the Dataset.
            entity (str):
                Name or Id of the entity which returned Datasets need to be
                associated to.
            created_by (str):
                Unique identifier of the user that created the Datasets.

        Returns:
            pandas.DataFrame
        """
        return schema.Dataset.find(
            as_df_=True,
            name=name,
            entity=entity,
            created_by=created_by
        )

    def get_dataset(self, dataset=None, name=None, entity=None, created_by=None):
        """Get a Dataset object from the database.

        All the arguments are optional but empty queries are not allowed, so at
        least one argument needs to be passed with a value different than ``None``.

        Args:
            dataset (Dataset, ObjectID or str):
                Dataset object (or the corresponding ObjectID, or its string
                representation) that we want to retreive.
            name (str):
                Name of the Dataset.
            entity (str):
                Name or Id of the entity which this Dataset is associated to.
            created_by (str):
                Unique identifier of the user that created the Dataset.

        Raises:
            ValueError:
                If the no arguments are passed with a value different than
                ``None`` or the query resolves to more than one object.

        Returns:
            Dataset
        """
        return schema.Dataset.get(
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

        The signal needs to be given a name and be associated to a Dataset.

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

        Raises:
            NotUniqueError:
                If a Signal with the same name already exists for this Dataset.

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

        return schema.Signal.insert(
            name=name,
            dataset=dataset,
            start_time=start_time,
            stop_time=stop_time,
            data_location=data_location,
            timestamp_column=timestamp_column,
            value_column=value_column,
            created_by=self.user
        )

    def add_signals(self, dataset, signals_path=None, start_time=None,
                    stop_time=None, timestamp_column=None, value_column=None):
        """Add a multiple Signal objects to the database.

        All the signals will be added to the Dataset using the CSV filename
        as their name.

        Args:
            dataset (Dataset or ObjectID or str):
                Dataset object which the created Signals belongs to or the
                corresponding ObjectId.
            signals_path (str):
                Path to the folder where the signals can be found. All the CSV
                files in this folder will be added.
            start_time (int):
                Optional. Minimum timestamp to use for these signals. If not provided
                this defaults to the minimum timestamp found in the signal data.
            stop_time (int):
                Optional. Maximum timestamp to use for these signals. If not provided
                this defaults to the maximum timestamp found in the signal data.
            timestamp_column (int):
                Optional. Index of the timestamp column.
            value_column (int):
                Optional. Index of the value column.
        """
        for filename in os.listdir(signals_path):
            if filename.endswith('.csv'):
                signal_name = filename[:-4]   # remove from filename .csv
                try:
                    self.add_signal(
                        name=signal_name,
                        dataset=dataset,
                        data_location=os.path.join(signals_path, filename),
                        start_time=start_time,
                        stop_time=stop_time,
                        timestamp_column=timestamp_column,
                        value_column=value_column,
                    )
                except NotUniqueError:
                    pass

    def get_signals(self, name=None, dataset=None, created_by=None):
        """Query the Signals collection.

        All the details about the matching Signals will be returned in
        a ``pandas.DataFrame``.

        All the arguments are optional, so a call without arguments will
        return a table with information about all the Signals availabe.

        Args:
            name (str):
                Name of the Signal.
            dataset (Dataset, ObjectID or str):
                Dataset object (or the corresponding ObjectID, or its string
                representation) to which the Signals have to belong.
            created_by (str):
                Unique identifier of the user that created the Signals.

        Returns:
            pandas.DataFrame
        """
        return schema.Signal.find(
            as_df_=True,
            name=name,
            dataset=dataset,
            created_by=created_by
        )

    def get_signal(self, signal=None, name=None, dataset=None, created_by=None):
        """Get a Signal object from the database.

        All the arguments are optional but empty queries are not allowed, so at
        least one argument needs to be passed with a value different than ``None``.

        Args:
            signal (Signal, ObjectID or str):
                Signal object (or the corresponding ObjectID, or its string
                representation) that we want to retreive.
            name (str):
                Name of the Signal.
            dataset (Dataset, ObjectID or str):
                Dataset object (or the corresponding ObjectID, or its string
                representation) to which the Signal has to belong.
            created_by (str):
                Unique identifier of the user that created the Signals.

        Raises:
            ValueError:
                If the no arguments are passed with a value different than
                ``None`` or the query resolves to more than one object.

        Returns:
            Signal
        """
        return schema.Signal.get(
            signal=signal,
            name=name,
            dataset=dataset,
            created_by=created_by
        )

    # ######## #
    # Template #
    # ######## #

    def add_template(self, name, template=None):
        """Add a new Template object to the database.

        The template can be passed as a name of a registered MLPipeline,
        or as a path to an MLPipeline JSON specification, or as a full
        dictionary specification of an MLPipeline or directly as an
        MLPipeline instance.

        If the ``template`` argument is not passed, the given ``name`` will
        be used to load an MLPipeline.

        During this step, apart from the Template object, a new Pipeline object
        using the default hyperparameters and with the same name as the
        Template will also be created.

        Args:
            name (str):
                Name of the Template.
            template (str, dict or MLPipeline):
                Name of the MLBlocks template to load or path to its JSON
                file or dictionary specification or MLPipeline instance.
                If not given, the ``name`` of the template is used.

        Raises:
            NotUniqueError:
                If a Template with the same name already exists.

        Returns:
            Template
        """
        template = template or name
        if isinstance(template, str) and os.path.isfile(template):
            with open(template, 'r') as f:
                template = json.load(f)

        pipeline_dict = MLPipeline(template).to_dict()

        template = schema.Template.insert(
            name=name,
            json=pipeline_dict,
            created_by=self.user
        )
        schema.Pipeline.insert(
            name=name,
            template=template,
            json=pipeline_dict,
            created_by=self.user
        )

        return template

    def get_templates(self, name=None, created_by=None):
        """Query the Templates collection.

        All the details about the matching Templates will be returned in
        a ``pandas.DataFrame``, except for the JSON specification of the
        template, which will be removed from the table for readability.

        In order to access the JSON specification of each Template, please
        retreive them using the ``get_template`` method.

        All the arguments are optional, so a call without arguments will
        return a table with information about all the Templates availabe.

        Args:
            name (str):
                Name of the Template.
            created_by (str):
                Unique identifier of the user that created the Templates.

        Returns:
            pandas.DataFrame
        """
        return schema.Template.find(
            as_df_=True,
            name=name,
            created_by=created_by,
            exclude_=['json']
        )

    def get_template(self, template=None, name=None, created_by=None):
        """Get a Template object from the database.

        All the arguments are optional but empty queries are not allowed, so at
        least one argument needs to be passed with a value different than ``None``.

        Args:
            template (Template, ObjectID or str):
                Template object (or the corresponding ObjectID, or its string
                representation) that we want to retreive.
            name (str):
                Name of the Template.
            created_by (str):
                Unique identifier of the user that created the Templates.

        Raises:
            ValueError:
                If the no arguments are passed with a value different than
                ``None`` or the query resolves to more than one object.

        Returns:
            Template
        """
        return schema.Template.get(
            template=template,
            name=name,
            created_by=created_by
        )

    # ######## #
    # Pipeline #
    # ######## #

    def add_pipeline(self, name, template, hyperparameters):
        """Add a new Pipeline object to the database.

        The Pipeline will consist on a copy of the given Template using
        the indicated hyperparameters.

        The hyperparameters can be passed as a dictionary containing the
        hyperparameter values following the MLBlocks specification format,
        or a path to a JSON file containing the corresponding values.

        Args:
            name (str):
                Name of the Pipeline.
            template (Template or ObjectID or str):
                Template object (or the corresponding ObjectID, or its string
                representation) that we want to use to create this Pipeline.
            hyperparamers (dict or str):
                dict containing the hyperparameter values following the MLBlocks
                specification format, or a path to a JSON file containing the
                corresponding values.

        Raises:
            NotUniqueError:
                If a Pipeline with the same name for this Template already exists.

        Returns:
            Pipeline
        """
        pipeline = self.get_template(template).load()
        if isinstance(hyperparameters, str):
            with open(hyperparameters, 'r') as f:
                hyperparameters = json.load(f)

        pipeline.set_hyperparameters(hyperparameters)

        return schema.Pipeline.insert(
            name=name,
            template=template,
            json=pipeline.to_dict(),
            created_by=self.user
        )

    def get_pipelines(self, name=None, template=None, created_by=None):
        """Query the Pipelines collection.

        All the details about the matching Pipelines will be returned in
        a ``pandas.DataFrame``, except for the JSON specification of the
        pipeline, which will be removed from the table for readability.

        In order to access the JSON specification of each Pipeline, please
        retreive them using the ``get_pipeline`` method.

        All the arguments are optional, so a call without arguments will
        return a table with information about all the Pipelines availabe.

        Args:
            name (str):
                Name of the Pipeline.
            template (Template or ObjectID or str):
                Template object (or the corresponding ObjectID, or its string
                representation) from which the Pipelines have to be derived.
            created_by (str):
                Unique identifier of the user that created the Pipelines.

        Returns:
            pandas.DataFrame
        """
        return schema.Pipeline.find(
            as_df_=True,
            name=name,
            template=template,
            created_by=created_by,
            exclude_=['json']
        )

    def get_pipeline(self, pipeline=None, name=None, template=None, created_by=None):
        """Get a Pipeline object from the database.

        All the arguments are optional but empty queries are not allowed, so at
        least one argument needs to be passed with a value different than ``None``.

        Args:
            pipeline (Template, ObjectID or str):
                Pipeline object (or the corresponding ObjectID, or its string
                representation) that we want to retreive.
            name (str):
                Name of the Pipeline.
            template (Template or ObjectID or str):
                Template object (or the corresponding ObjectID, or its string
                representation) from which the Pipeline has to be derived.
            created_by (str):
                Unique identifier of the user that created the Pipeline.

        Raises:
            ValueError:
                If the no arguments are passed with a value different than
                ``None`` or the query resolves to more than one object.

        Returns:
            Pipeline
        """
        return schema.Pipeline.get(
            pipeline=pipeline,
            name=name,
            template=template,
            created_by=created_by,
        )

    # ########## #
    # Experiment #
    # ########## #

    def add_experiment(self, name, template, dataset, signals=None, project=None):
        """Add a new Experiment object to the database.

        The Experiment will have to be associated to a Template and a Dataset.

        Optionally, a list of Signal objects or the corresponding ObjectIds have to
        can be passed to associate this Experiment to only a subset of Signals from
        the Dataset. In this case, the Signals passed need to be part of the Dataset,
        otherwise an Exception will be raised.

        If no Signals are passed, all the Signals from the Dataset are used.

        A project name can also be passed as a string to group experiments of
        a single project together.

        Args:
            name (str):
                Name of the Experiment.
            template (Template or ObjectID or str):
                Template object (or the corresponding ObjectID, or its string
                representation) that we want to use in this Experiment.
            dataset (Dataset or ObjectID or str):
                Dataset object (or the corresponding ObjectID, or its string
                representation) which will be used for this Experiment.
            signals (list[Signal, ObjectId or str]):
                list of Signals (or their corresponding ObjectIds) to be used for
                this Experiment.
            project (str):
                Name of the project which this Experiment belongs to.

        Raises:
            NotUniqueError:
                If an Experiment with the same name for this Template already exists.

        Returns:
            Experiment
        """
        dataset = self.get_dataset(dataset)

        if not signals:
            signals = dataset.signals
        else:
            for signal in signals:
                if self.get_signal(signal).dataset != dataset:
                    raise ValueError('All Signals must belong to the Dataset')

        return schema.Experiment.insert(
            name=name,
            project=project,
            template=template,
            dataset=dataset,
            signals=signals,
            created_by=self.user
        )

    def get_experiments(self, name=None, template=None, dataset=None,
                        signals=None, project=None, created_by=None):
        """Query the Experiments collection.

        All the details about the matching Experiments will be returned in
        a ``pandas.DataFrame``.

        All the arguments are optional, so a call without arguments will
        return a table with information about all the Experiments availabe.

        Args:
            name (str):
                Name of the Experiment.
            template (Template or ObjectID or str):
                Template that the Experiments must use.
            dataset (Dataset or ObjectID or str):
                Dataset that the Experiments must use.
            signals (list[Signal, ObjectId or str]):
                Signals that the Experiments must use.
            project (str):
                Name of the project which the Experiments must belong to.
            created_by (str):
                Unique identifier of the user that created the Experiments.

        Returns:
            pandas.DataFrame
        """
        return schema.Experiment.find(
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
        """Get an Experiment object from the database.

        All the arguments are optional but empty queries are not allowed, so at
        least one argument needs to be passed with a value different than ``None``.

        Args:
            experiment (Experiment, ObjectID or str):
                Experiment object (or the corresponding ObjectID, or its string
                representation) that we want to retreive.
            name (str):
                Name of the Experiment.
            template (Template or ObjectID or str):
                Template that the Experiment must use.
            dataset (Dataset or ObjectID or str):
                Dataset that the Experiment must use.
            signals (list[Signal, ObjectId or str]):
                Signals that the Experiment must use.
            project (str):
                Name of the project which the Experiment must belong to.
            created_by (str):
                Unique identifier of the user that created the Experiment.

        Raises:
            ValueError:
                If the no arguments are passed with a value different than
                ``None`` or the query resolves to more than one object.

        Returns:
            Experiment
        """
        return schema.Experiment.get(
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
        """Add a new Datarun object to the database.

        The Datarun needs to be associated to an Experiment and a Pipeline.

        Args:
            experiment (Experiment or ObjectID or str):
                Experiment object (or the corresponding ObjectID, or its string
                representation) to which this Datarun belongs.
            pipeline (Pipeline or ObjectID or str):
                Pipeline object (or the corresponding ObjectID, or its string
                representation) used by this Datarun.

        Returns:
            Datarun
        """
        return schema.Datarun.insert(
            experiment=experiment,
            pipeline=pipeline,
        )

    def get_dataruns(self, experiment=None, pipeline=None, status=None):
        """Query the Dataruns collection.

        All the details about the matching Dataruns will be returned in
        a ``pandas.DataFrame``.

        All the arguments are optional, so a call without arguments will
        return a table with information about all the Dataruns availabe.

        Args:
            experiment (Experiment or ObjectID or str):
                Experiment to which the Dataruns must belong.
            pipeline (Pipeline or ObjectID or str):
                Pipeline which the Dataruns must use.
            status (str):
                Status which the Dataruns must have.

        Returns:
            pandas.DataFrame
        """
        return schema.Datarun.find(
            as_df_=True,
            experiment=experiment,
            pipeline=pipeline,
            status=status,
            exclude_=['software_versions'],
        )

    def get_datarun(self, datarun=None, experiment=None, pipeline=None, status=None):
        """Get a Datarun object from the database.

        All the arguments are optional but empty queries are not allowed, so at
        least one argument needs to be passed with a value different than ``None``.

        Args:
            datarun (Datarun, ObjectID or str):
                Datarun object (or the corresponding ObjectID, or its string
                representation) that we want to retreive.
            experiment (Experiment or ObjectID or str):
                Experiment to which the Datarun must belong.
            pipeline (Pipeline or ObjectID or str):
                Pipeline which the Datarun must use.
            status (str):
                Status which the Datarun must have.

        Raises:
            ValueError:
                If the no arguments are passed with a value different than
                ``None`` or the query resolves to more than one object.

        Returns:
            Datarun
        """
        return schema.Datarun.get(
            experiment=experiment,
            pipeline=pipeline,
            status=status,
        )

    # ######### #
    # Signalrun #
    # ######### #

    def add_signalrun(self, datarun, signal):
        """Add a new Signalrun object to the database.

        The Signalrun needs to be associated to a Datarun and a Signal.

        Args:
            datarun (Datarun or ObjectID or str):
                Datarun object (or the corresponding ObjectID, or its string
                representation) to which this Signalrun belongs.
            signal (Signal or ObjectID or str):
                Signal object (or the corresponding ObjectID, or its string
                representation) used by this Signalrun.

        Returns:
            Datarun
        """
        return schema.Signalrun.insert(
            datarun=datarun,
            signal=signal,
        )

    def get_signalruns(self, datarun=None, signal=None, status=None):
        """Query the Signalruns collection.

        All the details about the matching Signalruns will be returned in
        a ``pandas.DataFrame``.

        All the arguments are optional, so a call without arguments will
        return a table with information about all the Dataruns availabe.

        Args:
            datarun (Datarun or ObjectID or str):
                Datarun to which the Signalruns must belong.
            signal (Signal or ObjectID or str):
                Signal which the Signalruns must use.
            status (str):
                Status which the Signalruns must have.

        Returns:
            pandas.DataFrame
        """
        return schema.Signalrun.find(
            as_df_=True,
            datarun=datarun,
            signal=signal,
            status=status,
        )

    def get_signalrun(self, signalrun=None, datarun=None, signal=None, status=None):
        """Get a Signalrun object from the database.

        All the arguments are optional but empty queries are not allowed, so at
        least one argument needs to be passed with a value different than ``None``.

        Args:
            signalrun (Signalrun, ObjectID or str):
                Signalrun object (or the corresponding ObjectID, or its string
                representation) that we want to retreive.
            datarun (Datarun or ObjectID or str):
                Datarun to which the Signalrun must belong.
            signal (Signal or ObjectID or str):
                Signal which the Signalrun must use.
            status (str):
                Status which the Signalrun must have.

        Raises:
            ValueError:
                If the no arguments are passed with a value different than
                ``None`` or the query resolves to more than one object.

        Returns:
            Signalrun
        """
        return schema.Signalrun.get(
            signalrun=signalrun,
            datarun=datarun,
            signal=signal,
            status=status,
        )

    # ##### #
    # Event #
    # ##### #

    def add_event(self, start_time, stop_time, source, severity=None,
                  signalrun=None, signal=None):
        """Add a new Event object to the database.

        The Event needs to have at least a start_time and a stop_time,
        and be associated to either a Signal or a Signalrun.

        If a Signalrun is given but no Signal is, the created Event will
        be associated to the Signalrun signal value.

        If both a Signalrun and a Signal are given, the Signal must be
        the one used by the Signalrun.

        Args:
            start_time (int):
                Timestamp at which the event starts.
            stop_time (int):
                Timestamp at which the event ends.
            source (str):
                Description of where this Event was created. It must be "orion",
                "shape matching" or "manually created".
            severity (float):
                Severity score value. Optional.
            signalrun (Signalrun or ObjectID or str):
                Signalrun object (or the corresponding ObjectID, or its string
                representation) to which this Event is associated.
            signal (Signal or ObjectID or str):
                Signal object (or the corresponding ObjectID, or its string
                representation) to which this Event is associated.

        Raises:
            ValueError:
                if neither a Signal or a Signalrun are given, or if the Signal
                is not the one used by the Signalrun.

        Returns:
            Event
        """
        if signal is None and signalrun is None:
            raise ValueError('An Event must be associated to either a Signalrun or a Signal')
        if signal is not None and signalrun is not None:
            if self.get_signal(signal) != self.get_signalrun(signalrun).signal:
                raise ValueError('Signal cannot be different than Signalrun.signal')

        return schema.Event.insert(
            signalrun=signalrun,
            signal=signal or signalrun.signal,
            start_time=int(start_time),
            stop_time=int(stop_time),
            severity=severity,
            source=source,
        )

    def get_events(self, signalrun=None, signal=None, source=None):
        """Query the Events collection.

        All the details about the matching Signalruns will be returned in
        a ``pandas.DataFrame``.

        All the arguments are optional, so a call without arguments will
        return a table with information about all the Events availabe.

        Args:
            signalrun (Signalrun or ObjectID or str):
                Signalrun to which the Events must belong.
            signal (Signal or ObjectID or str):
                Signal to which the Events be associated.
            source (str):
                Source from which the Events must come.

        Returns:
            pandas.DataFrame
        """
        return schema.Event.find(
            as_df_=True,
            signalrun=signalrun,
            signal=signal,
            source=source,
        )

    def get_event(self, event=None, signalrun=None, signal=None, source=None):
        """Get an Event object from the database.

        All the arguments are optional but empty queries are not allowed, so at
        least one argument needs to be passed with a value different than ``None``.

        Args:
            event (Event, ObjectID or str):
                Event object (or the corresponding ObjectID, or its string
                representation) that we want to retreive.
            signalrun (Signalrun or ObjectID or str):
                Signalrun to which the Events must belong.
            signal (Signal or ObjectID or str):
                Signal to which the Events be associated.
            source (str):
                Source from which the Events must come.

        Raises:
            ValueError:
                If the no arguments are passed with a value different than
                ``None`` or the query resolves to more than one object.

        Returns:
            Event
        """
        return schema.Event.get(
            event=event,
            signalrun=signalrun,
            signal=signal,
            source=source,
        )

    # ########## #
    # Annotation #
    # ########## #

    def add_annotation(self, event, tag=None, comment=None):
        """Add a new Annotation object to the database.

        The Event needs to be associated with an Event, and can be given a
        ``tag`` and a text comment.

        Args:
            event (Event or ObjectID or str):
                Event object (or the corresponding ObjectID, or its string
                representation) to which this Annotation is associated.
            tag (str):
                Tag of this Annotation.
            comment (str):
                Text comment of this Annotation.

        Returns:
            Annotation
        """
        return schema.Annotation.insert(
            event=event,
            tag=tag,
            comment=comment,
        )

    def get_annotations(self, event=None, tag=None, comment=None, created_by=None):
        """Query the Annotations collection.

        All the details about the matching Annotations will be returned in
        a ``pandas.DataFrame``.

        All the arguments are optional, so a call without arguments will
        return a table with information about all the Annotations availabe.

        Args:
            event (Event or ObjectID or str):
                Event to which the Annotations must belong.
            tag (str):
                Tag which the Annotations must have.
            comment (str):
                Comment which the Annotations must have.
            created_by (str):
                Unique identifier of the user that created the Annotations.

        Returns:
            pandas.DataFrame
        """
        return schema.Annotation.find(
            as_df_=True,
            event=event,
            tag=tag,
            created_by=created_by,
        )

    def get_annotation(self, annotation=None, event=None, tag=None, created_by=None):
        """Get an Event object from the database.

        All the arguments are optional but empty queries are not allowed, so at
        least one argument needs to be passed with a value different than ``None``.

        Args:
            annotation (Annotation, ObjectID or str):
                Annotation object (or the corresponding ObjectID, or its string
                representation) that we want to retreive.
            event (Event or ObjectID or str):
                Event to which the Annotation must belong.
            tag (str):
                Tag which the Annotation must have.
            comment (str):
                Comment which the Annotation must have.
            created_by (str):
                Unique identifier of the user that created the Annotation.

        Raises:
            ValueError:
                If the no arguments are passed with a value different than
                ``None`` or the query resolves to more than one object.

        Returns:
            Annotation
        """
        return schema.Annotation.get(
            annotation=annotation,
            event=event,
            tag=tag,
            created_by=created_by,
        )
