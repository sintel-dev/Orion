# Database Schema

The **Orion Database** contains the following collections and fields:

## Dataset

A **Dataset** represents a group of Signals that are grouped together under a common name,
which is usually defined by an external entity.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Dataset object
* `name (String)`: Name of the dataset
* `entity (String)`: Name or Id of the entity which this dataset is associated to
* `created_by (String)`: Identifier of the user that created this Dataset Object
* `insert_time (DateTime)`: Time when this Dataset Object was inserted

## Signal

The **Signal** collection contains all the required details to be able to load the observations
from a timeseries signal, as well as some metadata about it, such as the minimum and maximum
timestamps that want to be used or the user that registered it.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Signal object
* `name (String)`: Name of the signal
* `dataset_id (ObjectID - Foreign Key)`: Unique Identifier of the Dataset which this signal belongs to
* `start_time (Integer)`: minimum timestamp of this signal
* `stop_time (Integer)`: maximum timestamp of this signal
* `data_location (String)`: URI of the dataset
* `timestamp_column (Integer)`: index of the timestamp column
* `value_column (Integer)`: index of the value column
* `created_by (String)`: Identifier of the user that created this Signal Object
* `insert_time (DateTime)`: Time when this Signal Object was inserted

## Template

The **Template** collection contains all the pipeline templates from which the
pipelines that later on will be used to run an experiments are generated.
The template includes all the default hyperparameter values, as well as the tunable hyperparameter
ranges.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Template object
* `name (String)`: Name given to this pipeline template
* `template (SubDocument)`: JSON representation of this pipeline template
* `created_by (String)`: Identifier of the user that created this Pipeline Template Object
* `insert_time (DateTime)`: Time when this Pipeline Object was inserted

## Pipeline

The **Pipeline** collection contains all the pipelines registered in the system, including
their details, such as the list of primitives and all the configured hyperparameter values.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Pipeline object
* `name (String)`: Name given to this pipeline
* `template_id (ObjectID - Foreign Key)`: Unique Identifier of the Template
used to generate this pipeline
* `pipeline (SubDocument)`: JSON representation of this pipeline object
* `created_by (String)`: Identifier of the user that created this Pipeline Object
* `insert_time (DateTime)`: Time when this Pipeline Object was inserted

## Experiment

An **Experiment** is associated with a Dataset, a subset of its Signals and a Template,
and represents a collection of Dataruns, executions of Pipelines generated from the Experiment
Template over its Signals Set.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Experiment object
* `name (String)`: Name given to describe the Experiment
* `project (String)`: Name given to describe the project to which the experiment belongs
* `template_id (ObjectID - Foreign Key)`: Unique Identifier of the Pipeline used
* `dataset_id (ObjectID - Foreign Key)`: Unique Identifier of the Dataset to which the Signals belong to.
* `signal_set (List of Foreign Keys)`: A list of Signal IDs from the Dataset associated with this Experiment
* `created_by (String)`: Identifier of the user that created this Experiment Object
* `insert_time (DateTime)`: Time when this Experiment Object was inserted

## Datarun

The **Datarun** objects represent single executions of an **Experiment**,
and contain all the information about the environment and context where this execution
took place, which potentially allows to later on reproduce the results in a new environment.

It also contains information about whether the execution was successful or not, when it started
and ended, and the number of events that were found in this experiment.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Datarun object
* `experiment_id (ObjectID - Foreign Key)`: Unique Identifier of the Experiment
* `pipeline_id (ObjectID - Foreign Key)`: Unique Identifier of the Pipeline used
* `start_time (DateTime)`: When the execution started
* `end_time (DateTime)`: When the execution ended
* `software_versions (List of Strings)`: version of each python dependency installed in the
  *virtualenv* when the execution took place
* `budget_type (String)`: Type of budget used (time or number of iterations)
* `budget_amount (Integer)`: Budget amount
* `events (Integer)`: Number of events detected during this Datarun execution
* `status (String)`: Whether the Datarun is still running, finished successfully or failed
* `insert_time (DateTime)`: Time when this Datarun Object was inserted

## Signalrun

The **Signalrun** objects represent single executions of a **Pipeline** on a **Signal** within a
Datarun.

It contains information about whether the execution was successful or not, when it started
and ended, the number of events that were found by the Pipeline, and where the model and
metrics are stored.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Datarun object
* `datarun_id (ObjectID - Foreign Key)`: Unique Identifier of the Datarun to which this Signalrun belongs
* `signal_id (ObjectID - Foreign Key)`: Unique Identifier of the Signal used
* `start_time (DateTime)`: When the execution started
* `end_time (DateTime)`: When the execution ended
* `model_location (String)`: URI of the fitted model
* `metrics_location (String)`: URI of the metrics
* `events (Integer)`: Number of events detected during this Signalrun execution
* `status (String)`: Whether the Signalrun is still running, finished successfully or failed
* `insert_time (DateTime)`: Time when this Datarun Object was inserted

## Event

Each one of the anomalies detected by the pipelines is stored as an **Event**, which
contains the details about the start time, the stop time and the severity score.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Event object
* `signalrun_id (ObjectID - Foreign Key)`: Unique Identifier of the Signalrun during which this
Event was detected.
* `signal_id (ObjectID - Foreign Key)`: Unique Identifier of the Signal to which this Event relates
* `start_time (Integer)`: Timestamp where the anomalous interval starts
* `stop_time (Integer)`: Timestamp where the anomalous interval ends
* `severity (Float)`: Severity score given by the pipeline to this Event
* `source (String)`: "orion", "shape matching", or "manually created"
* `insert_time (DateTime)`: Time when this Event Object was inserted

## Event Interaction

The **Event Interaction** collection records all the interaction history related to events.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Interaction object
* `event_id (ObjectID)`: Unique Identifier of the Event to which this event relates
* `action (String)`: Action type performed on this event, such as delete, split, and adjust
* `start_time (Integer)`: Timestamp where the anomalous interval starts
* `stop_time (Integer)`: Timestamp where the anomalous interval ends
* `created_by (String)`: Identifier of the user who interacted with the target Object
* `insert_time (DateTime)`: Time when this Event Interaction Object was inserted

## Annotation

Each Event can have multiple **Annotations**, from one or more users.
**Annotations** are expected to be inserted by the domain experts after the Datarun has
finished and they analyze the results.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Comment object
* `event_id (ObjectID - Foreign Key)`: Unique Identifier of the Event to which this Annotation relates
* `tag (String)`: User given tag for this event
* `comment (String)`: Comment text
* `created_by (String)`: Identifier of the user that created this Annotation Object
* `insert_time (DateTime)`: Time when this Annotation Object was inserted

# OrionExplorer Usage

In the following steps we will learn how to interact with the Orion Database using the
`OrionExplorer` class, which provides following functionalities:

1. Add _Datasets_, _Signals_, _Templates_ and _Experiments_.
2. Create _Pipelines_ and start _Dataruns_, which create _Signalruns_ and _Events_.
3. Explore the _Dataruns_ and their results (i.e. detected _Events_).
4. Add _Annotations_ to the existing _Events_ as well as manual _Events_.

## Creating an instance of the OrionExplorer

In order to connect to the database, all you need to do is import and create an instance of the
`OrionExplorer` class.

Note that, because of the dynamic schema-less nature of MongoDB, no database initialization
or table creation is needed. All you need to do start using a new database is create the
`OrionExplorer` instance with the right connection details and start using it!

```python3
from orion.explorer import OrionExplorer

orex = OrionExplorer()
```

This will directly create a connection to the database named `'orion'` at the default
MongoDB host, `localhost`, and port, `27017`.

In case you wanted to connect to a different database, host or port, or in case user authentication
is enabled in your MongoDB instance, you can pass any required additional arguments:

* `database`: Name of the MongoDB database to connect to. Defaults to `'orion'`.
* `host`: Hostname or IP address of the MongoDB Instance. Defaults to `'localhost'`.
* `port`: Port to which MongoDB is listening. Defaults to `27017`.
* `username`: username to authenticate with.
* `password`: password to authenticate with.
* `authentication_source`: database to authenticate against.

```python3
orex = OrionExplorer(
     database='orion',
     host='localhost',
     port=27017,
     username='orion',
     password='secret_password',
     authentication_source='admin'
)
```

## Setting up the Orion Environment

The first thing that you will need to do to start using **Orion** with a Database will be
to add information about your data and your pipelines.

This can be done by using the methods of the `OrionExplorer` class that are documenteted below,
which allow creating the corresponding objects in the Database.

Additionally, for each `add_{model_name}` method explain, another method call `get_{model_name}s`
exists which allows you to query the database and retreive the created objects. All the details
about the arguments accepted by these methods can be found in the [API Reference](TODO: Add link).

### Add a Dataset

In order to add a dataset you can use the `add_dataset` method, which has the following arguments:

* `name (str)`: Name of the dataset
* `entity (str)`: Name or Id of the entity which this dataset is associated to
* `created_by (str)`: Identifier of the user that created this Dataset Object

For example, if we want to add the demo dataset provided by Orion we could use:

```python3
dataset = orex.add_dataset(
    name='Demo Dataset',
    entity='Orion',
)
```

This call will try to create a new _Dataset_ object in the database and return it.

As mentioned before, we can obtain the list of all the Dataset objects at any point by executing
the `get_datasets` method, which accepts the following arguments:

* `_id (ObjectID)`: Unique Identifier of the Dataset
* `name (str)`: Name of the dataset
* `entity (str)`: Name or Id of the entity which this dataset is associated to
* `created_by (str)`: Identifier of the user that created this Dataset Object

For example, if we want to see all the datasets created by `'my_username`' we can use:

```python3
datasets = orex.get_datasets(created_by='my_username')
```

Which will return a `pymongo` cursor object with all the required datasets.

### Add a Signal

The next step is to add Signals. This can be done with the `add_signal` method, which expects:

* `name (str)`: Name of the signal
* `dataset (Dataset or ObjectID)`: Dataset Object or Dataset Id.
* `start_time (int)`: (Optional) minimum timestamp to be used for this signal. If not given, it
  defaults to the minimum timestamp found in the data.
* `stop_time (int)`: (Optional) maximum timestamp to be used for this signal. If not given, it
  defaults to the maximum timestamp found in the data.
* `data_location (str)`: URI of the dataset
* `timestamp_column (int)`: (Optional) index of the timestamp column. Defaults to 0.
* `value_column (int)`: (Optional) index of the value column. Defaults to 1.

For example, adding the `S-1` signal to the Demo Dataset that we just created could be done like
this:

```python3
signal = orex.add_signal(
    name='S-1',
    dataset=dataset,
    data_location='orion/data/S-1.csv',
    start_time=1483228800,
    stop_time=1514764800,
    timestamp_column=2,
    value_column=4,
)
```

### Add a Template

The next thing we need to add is a _Template_ using the `add_template` method.

This method expects:

* `name (str)`: Name given to this pipeline template
* `template (dict or str)`: dict containing the MLPipeline details or path to the corresponding
  JSON file.

For example, if we want to create a _Template_ using the `lstm_dynamic_threshold` pipeline
included in Orion we can do:

```python3
template = orex.add_template(
    name='lstm_dynamic_threshold',
    template='orion/pipelines/lstm_dynamic_threshold.json',
)
```

### Add a Pipeline

After a _Template_ is created we can add _Pipelines_ with specific hyperparameter values.

In order to do this we will need to call the `add_pipeline` method passing:

* `name (str)`: Name given to this pipeline
* `template (Template or ObjectID)`: Template or the corresponding id.
* `pipeline (dict or str)`: dict containing the MLPipeline details or path to the corresponding
  JSON file. Optional. Raises an error if both this and a hyperparameters dict is given.
* `hyperparameters (dict or str)`: dict containing the hyperparameter details or path to the
  corresponding JSON file. Optional. Raises an error if both this and a pipeline dict is given.

> **NOTE**: When a _Template_ is created, a _Pipeline_ with the default hyperparamter
> values set is automatically added with the same name, so this step needs to be done only to
> add different hyperparameter configurations.

For example, if we want to specify a different number of epochs for the LSTM primitive of the
pipeline that we just created we will run:

```python3
new_hyperparameters = {
    'keras.Sequential.LSTMTimeSeriesRegressor#1': {
        'epochs': 10
    }
}
pipeline = orex.add_pipeline(
    name='lstm_dynamic_threshold_10_epochs',
    template=template,
    hyperparameters=new_hyperparameters,
)
```

Alternatively, if we have stored our modified pipeline in a JSON file, we can execute:

```python3
template = orex.add_pipeline(
    name='lstm_dynamic_threshold_modified',
    template=template,
    pipeline='path/to/my/modified/pipeline.json',
)
```

### Add an Experiment

Once we have a _Dataset_ with _Signals_ and a _Template_, we are ready to add an
_Experiment_.

In order to run an _Experiment_ we will need to:

1. Get the _Dataset_ and the list of _Signals_ that we want to run the _Experiment_ on.
2. Get the _Template_ which we want to use for the _Experiment_
3. Call the `add_experiment` method passing all these with an experiment, a project name and a
   username.

For example, if we want to create an experiment using the _Dataset_, the _Signals_ and the
_Template_ that we just created, we will:

```python3
experiment = orex.add_experiment(
    name='My Experiment',
    project='My Project',
    template=template,
    dataset=dataset,
    signals=signals,
)
```

## Starting a Datarun

Once we have created our _Experiment_ object we are ready to start executing _Pipelines_ on our
_Signals_. For this we will use the `add_datarun` method, which expects:

* `experiment (Experiment or ObjectID)`: Experiment object or the corresponding ID.
* `pipeline (Pipeline or ObjectID)`: Pipeline object or the corresponding ID.

For example, if we want to execute the default `lstm_dynamic_threshold` _Pipeline_ over the
_Experiment_ that we just created we will execute:

```python3
pipeline = orex.get_pipeline(name='lstm_dynamyc_threshold')
datarun = orex.add_datarun(
    experiment=experiment,
    pipeline=pipeline
)
```

This will create the _Datarun_ object in the database and then start creating and executing
_Signalruns_, one for each _Signal_ in the _Experiment_.

## Explore the results

Once a _Datarun_ has started we can see its progress by calling it's `get_status` method:

```python
status = datarun.get_status()
```

This will return the string `RUNNING`, `SUCCESS` or `ERROR` depending on whether it is still
running, it ended successfully or there was some error while running.


**WORK IN PROGRESS**









## 3. Registering a new Signal

In order have signals associated with a data set, we need to add the signals to the respective data set.
Which signals belong to a data set is usually defined by an external entity.

In order to add a signal to a data set, you need to call the `OrionExplorer.add_signal` method passing:

* `name - str`: Name that you want to give to this signal. It must be unique.
* `dataset - str`: Name of the dataset.
* `start_time - int`: (Optional) minimum timestamp to be used for this signal. If not given, it
  defaults to the minimum timestamp found in the data.
* `stop_time - int`: (Optional) maximum timestamp to be used for this signal. If not given, it
  defaults to the maximum timestamp found in the data.
* `location - int`: (Optional) path to the CSV. Skip if using a demo signal hosted on S3.
* `timestamp_column - int`: (Optional) index of the timestamp column. Defaults to 0.
* `value_column - int`: (Optional) index of the value column. Defaults to 1.
* `user - str`: (Optional) Identifier of the user that is creating this signal.

In the simplest scenario, you can register a signal that uses one of the demo signals with the
default start and stop times.

For example, to register a signal (S-1) for the dataset **S-1** used previously you will execute:

```
orex.add_signal(name='S-1', dataset='S-1')
```

In a more complex scenario, we might be loading a CSV file that has the time index in the
third column and the value in the fifth one, and we might want to restrict the timestamps
to a certain range and provide the ID of the user that is registering it.

In this case we will execute:

```
orex.add_dataset('a_dataset_name')
orex.add_signal(
    name='a_signal',
    dataset='a_dataset_name',
    start_time=1483228800,
    stop_time=1514764800,
    location='/path/to/the.csv',
    timestamp_column=2,
    value_column=4,
    user_id='1234',
)
```

**NOTE**: The signal name must be unique, which means that `add_signal` method will fail
if a second signal is added using the same name.

## 4. Exploring the registered datasets and signals

In order to obtain the list of already registered datasets and signals, you can use the
`OrionExplorer.get_datasets` and `OrionExplorer.get_signals` methods.

This method returns a `pandas.DataFrame` containing all the details about the registered
datasets or signals:


```
datasets = orex.get_datasets()
```

```
signals = orex.get_signals()
```

Optionally, you can restrict the results to a particular signal by passing the signal name or dataset name
as arguments to the `get_signals` method:

```
signals = orex.get_signals(name='S-1')
```

## 5. Registering a new Pipeline

Another thing that you will need to do before being able to process the created dataset will be
**registering a new pipeline**.

For this you will need to call the `OrionExplorer.add_pipeline` method passing:

* `name - str`: Name that you want to give to this pipeline.
* `path - str`: Path to the JSON file that contains the pipeline specification.
* `user - str`: (Optional) Identifier of the user that is creating this Pipeline.

For example, to register the LSTM pipeline that we used previously under the `'LSTM'` name
you will execute:

```
orex.add_pipeline('LSTM', 'orion/pipelines/lstm_dynamic_threshold.json', '1234')
```


## 6. Exploring the registered pipelines

Just like datasets, you can obtain the list of registered pipelines using the
`OrionExplorer.get_pipelines` method.

This method returns a `pandas.DataFrame` containing all the details about the registered
pipelines:

```
orex.get_pipelines()
```

## 7. Create an experiment

Once we have datasets and pipelines, we can create an experiment. Experiments are related to a pipeline
and a dataset. We can also optionally define a signal_set, which can be a subset of the signals contained
in the specified dataset.

```
orex.add_experiment(name='LSTM_S1',pipeline='LSTM', dataset='S-1', signal_set=['S-1'], user_id='1234')
```

## 8. Running an experiment

Once we have an experiment stored in the database, we can execute the experiment, which will create a Datarun over all signals specified
in the experiment. One can also optionally specify certain pipeline hyperparameters and their values, otherwise the default values from
the specified pipeline are used.

```
orex.run_experiment(experiment='LSTM_S1', pipeline_specs={"keras.Sequential.LSTMTimeSeriesRegressor": {"epochs": 35}})
```

Once the process has finished, a new Datarun object will have been created in the Database.


# Concept for using the feedback

In order to incorporate the input from the users, we use a shape matching to improve future
anomaly detection tasks. The procedure is the following:

* For each signal in the signalset of the datarun:
    * run pipeline on signal and find anomalies
    * get all known events that are related to the signal and have a annotation tag from database
    * For each known event:
        * get the aggregated signal (in intermediate outputs) from the datarun where the known event was found
        * get the shape of the sequence that was marked as anomalous in the known event
        * compare this shape to aggregated signal of current datarun using Dynamic Time Warping and check if
          some subsequence is significantly closer than others
        * if there is a similar sequence, add an event with source 'shape matching' and a corresponding annotation
          tag that is similar to the tag of the original event
        * if there is any anomaly that was found in the current datarun, which overlaps with the known event,
          remove it from the list of found anomalies
    * add all remaining found anomalies as an event with source 'orion'
