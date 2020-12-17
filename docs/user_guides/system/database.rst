.. highlight:: shell

===============
Database Schema
===============

The **Orion Database** contains the following collections and fields:

Dataset
-------

A **Dataset** represents a group of Signals that are grouped together under a common name,
which is usually defined by an external entity.

Fields
~~~~~~

* ``_id (ObjectID)``: Unique Identifier of this Dataset object
* ``name (String)``: Name of the dataset
* ``entity (String)``: Name or Id of the entity which this dataset is associated to
* ``created_by (String)``: Identifier of the user that created this Dataset Object
* ``insert_time (DateTime)``: Time when this Dataset Object was inserted

Signal
------

The **Signal** collection contains all the required details to be able to load the observations
from a timeseries signal, as well as some metadata about it, such as the minimum and maximum
timestamps that want to be used or the user that registered it.

Fields
~~~~~~

* ``_id (ObjectID)``: Unique Identifier of this Signal object
* ``name (String)``: Name of the signal
* ``dataset_id (ObjectID - Foreign Key)``: Unique Identifier of the Dataset which this signal belongs to
* ``start_time (Integer)``: minimum timestamp of this signal
* ``stop_time (Integer)``: maximum timestamp of this signal
* ``data_location (String)``: URI of the dataset
* ``timestamp_column (Integer)``: index of the timestamp column
* ``value_column (Integer)``: index of the value column
* ``created_by (String)``: Identifier of the user that created this Signal Object
* ``insert_time (DateTime)``: Time when this Signal Object was inserted

Template
--------

The **Template** collection contains all the pipeline templates from which the
pipelines that later on will be used to run an experiments are generated.
The template includes all the default hyperparameter values, as well as the tunable hyperparameter
ranges.

Fields
~~~~~~

* ``_id (ObjectID)``: Unique Identifier of this Template object
* ``name (String)``: Name given to this pipeline template
* ``json (SubDocument)``: JSON representation of this pipeline template
* ``created_by (String)``: Identifier of the user that created this Pipeline Template Object
* ``insert_time (DateTime)``: Time when this Pipeline Object was inserted

Pipeline
--------

The **Pipeline** collection contains all the pipelines registered in the system, including
their details, such as the list of primitives and all the configured hyperparameter values.

Fields
~~~~~~

* ``_id (ObjectID)``: Unique Identifier of this Pipeline object
* ``name (String)``: Name given to this pipeline
* ``template_id (ObjectID - Foreign Key)``: Unique Identifier of the Template used to generate this pipeline
* ``json (SubDocument)``: JSON representation of this pipeline object
* ``created_by (String)``: Identifier of the user that created this Pipeline Object
* ``insert_time (DateTime)``: Time when this Pipeline Object was inserted

Experiment
----------

An **Experiment** is associated with a Dataset, a subset of its Signals and a Template,
and represents a collection of Dataruns, executions of Pipelines generated from the Experiment
Template over its Signals Set.

Fields
~~~~~~

* ``_id (ObjectID)``: Unique Identifier of this Experiment object
* ``name (String)``: Name given to describe the Experiment
* ``project (String)``: Name given to describe the project to which the experiment belongs
* ``template_id (ObjectID - Foreign Key)``: Unique Identifier of the Pipeline used
* ``dataset_id (ObjectID - Foreign Key)``: Unique Identifier of the Dataset to which the Signals belong to.
* ``signals (List of Foreign Keys)``: A list of Signal IDs from the Dataset associated with this Experiment
* ``created_by (String)``: Identifier of the user that created this Experiment Object
* ``insert_time (DateTime)``: Time when this Experiment Object was inserted

Datarun
-------

The **Datarun** objects represent single executions of an **Experiment**,
and contain all the information about the environment and context where this execution
took place, which potentially allows to later on reproduce the results in a new environment.

It also contains information about whether the execution was successful or not, when it started
and ended, and the number of events that were found in this experiment.

Fields
~~~~~~

* ``_id (ObjectID)``: Unique Identifier of this Datarun object
* ``experiment_id (ObjectID - Foreign Key)``: Unique Identifier of the Experiment
* ``pipeline_id (ObjectID - Foreign Key)``: Unique Identifier of the Pipeline used
* ``start_time (DateTime)``: When the execution started
* ``end_time (DateTime)``: When the execution ended
* ``software_versions (List of Strings)``: version of each python dependency installed in the *virtualenv* when the execution took place
* ``budget_type (String)``: Type of budget used (time or number of iterations)
* ``budget_amount (Integer)``: Budget amount
* ``num_events (Integer)``: Number of events detected during this Datarun execution
* ``status (String)``: Whether the Datarun is still running, finished successfully or failed
* ``insert_time (DateTime)``: Time when this Datarun Object was inserted

Signalrun
---------

The **Signalrun** objects represent single executions of a **Pipeline** on a **Signal** within a
Datarun.

It contains information about whether the execution was successful or not, when it started
and ended, the number of events that were found by the Pipeline, and where the model and
metrics are stored.

Fields
~~~~~~

* ``_id (ObjectID)``: Unique Identifier of this Datarun object
* ``datarun_id (ObjectID - Foreign Key)``: Unique Identifier of the Datarun to which this Signalrun belongs
* ``signal_id (ObjectID - Foreign Key)``: Unique Identifier of the Signal used
* ``start_time (DateTime)``: When the execution started
* ``end_time (DateTime)``: When the execution ended
* ``model_location (String)``: URI of the fitted model
* ``metrics_location (String)``: URI of the metrics
* ``num_revents (Integer)``: Number of events detected during this Signalrun execution
* ``status (String)``: Whether the Signalrun is still running, finished successfully or failed
* ``insert_time (DateTime)``: Time when this Datarun Object was inserted

Event
-----

Each one of the anomalies detected by the pipelines is stored as an **Event**, which
contains the details about the start time, the stop time and the severity score.

Fields
~~~~~~

* ``_id (ObjectID)``: Unique Identifier of this Event object
* ``signalrun_id (ObjectID - Foreign Key)``: Unique Identifier of the Signalrun during which this Event was detected.
* ``signal_id (ObjectID - Foreign Key)``: Unique Identifier of the Signal to which this Event relates
* ``start_time (Integer)``: Timestamp where the anomalous interval starts
* ``stop_time (Integer)``: Timestamp where the anomalous interval ends
* ``severity (Float)``: Severity score given by the pipeline to this Event
* ``source (String)``: ``ORION``, ``SHAPE_MATCHING``, or ``MANUALLY_CREATED``
* ``num_annotations (int)``: Number of Annotation associated to this Event.
* ``insert_time (DateTime)``: Time when this Event Object was inserted

Event Interaction
-----------------

The **Event Interaction** collection records all the interaction history related to events.

Fields
~~~~~~

* ``_id (ObjectID)``: Unique Identifier of this Interaction object
* ``event_id (ObjectID)``: Unique Identifier of the Event to which this event relates
* ``action (String)``: Action type performed on this event, such as delete, split, and adjust
* ``start_time (Integer)``: Timestamp where the anomalous interval starts
* ``stop_time (Integer)``: Timestamp where the anomalous interval ends
* ``created_by (String)``: Identifier of the user who interacted with the target Object
* ``insert_time (DateTime)``: Time when this Event Interaction Object was inserted

Annotation
----------

Each Event can have multiple **Annotations**, from one or more users.
**Annotations** are expected to be inserted by the domain experts after the Datarun has
finished and they analyze the results.

Fields
~~~~~~

* ``_id (ObjectID)``: Unique Identifier of this Comment object
* ``event_id (ObjectID - Foreign Key)``: Unique Identifier of the Event to which this Annotation relates
* ``tag (String)``: User given tag for this event
* ``comment (String)``: Comment text
* ``created_by (String)``: Identifier of the user that created this Annotation Object
* ``insert_time (DateTime)``: Time when this Annotation Object was inserted
