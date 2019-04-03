# Database Schema

The **Orion Database** contains the following collections and fields:

## Dataset

The **Dataset** collection contains all the required details to be able to load
the observations from a satellite signal, as well as some metadata about it, such as
the minimum and maximum timestamps that want to be used or the user that registered it.

### Fields

* \_id (ObjectID): Unique Identifier of this Dataset object
* name (String): Name of the dataset
* signal_set (String): Identifier of the signal
* satellite_id (String): Identifier of the satellite
* start_time (Integer): minimum timestamp of this dataset
* stop_time (Integer): maximum timestamp of this dataset
* data_location (String): URI of the dataset
* timestamp_column (Integer): index of the timestamp column
* value_column (Integer): index of the value column
* created_by (String): Identifier of the user that created this Dataset Object
* insert_time (DateTime): Time when this Dataset Object was inserted

## Pipeline

The **Pipeline** collection contains all the pipelines registered in the system, including
their details, such as the list of primitives and all the configured hyperparameter values.

### Fields

* \_id (ObjectID): Unique Identifier of this Pipeline object
* name (String): Name given to this pipeline
* mlpipeline (SubDocument): JSON representation of this pipeline
* created_by (String): Identifier of the user that created this Pipeline Object
* insert_time (DateTime): Time when this Pipeline Object was inserted

## Datarun

The **Datarun** objects represent single executions of a **Pipeline** on a **Dataset**,
and contain all the information about the environment and context where this execution
took place, which potentially allows to later on reproduce the results in a new environment.

It also contains information about whether the execution was successful or not, when it started
and ended, and the number of events that were found by the pipeline.

### Fields

* \_id (ObjectID): Unique Identifier of this Datarun object
* dataset_id (ObjectID - Foreign Key): Unique Identifier of the Dataset used
* pipeline_id (ObjectID - Foreign Key): Unique Identifier of the Pipeline used
* start_time (DateTime): When the execution started
* end_time (DateTime): When the execution ended
* software_versions (List of Strings): version of each python dependency installed in the
  *virtualenv* when the execution took place
* budget_type (String): Type of budget used (time or number of iterations)
* budget_amount (Integer): Budget amount
* model_location (String): URI of the fitted model
* metrics_location (String): URI of the metrics
* events (Integer): Number of events detected during this Datarun execution
* status (String): Whether the Datarun is still running, finished successfully or failed
* created_by (String): Identifier of the user that created this Datarun Object
* insert_time (DateTime): Time when this Datarun Object was inserted

## Event

Each one of the anomalies detected by the pipelines is stored as an **Event**, which
contains the details about the start time, the stop time and the severity score.

### Fields

* \_id (ObjectID): Unique Identifier of this Event object
* datarun_id (ObjectID - Foreign Key): Unique Identifier of the Datarun during which this
  Event was detected.
* start_time (Integer): Timestamp where the anomalous interval starts.
* stop_time (Integer): Timestamp where the anomalous interval ends.
* score (Float): Severity score given by the pipeline to this Event
* tag (String): User given tag for this Event
* insert_time (DateTime): Time when this Event Object was inserted

## Comment

Each Event can have multiple **Comments**, from one or more users.
**Comments** are expected to be inserted by the domain experts after the Datarun has
finished and they analyze the results.

### Fields

* \_id (ObjectID): Unique Identifier of this Comment object
* event_id (ObjectID - Foreign Key): Unique Identifier of the Event to which this Comment relates
* text (String): Comment contents
* created_by (String): Identifier of the user that created this Event Object
* insert_time (DateTime): Time when this Event Object was inserted
