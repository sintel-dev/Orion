# Database Schema

The **Orion Database** contains the following collections and fields:

## Dataset

A **Dataset** represents a group of Signals that are grouped together under a common name
and which are associated with a name and an external entity, such as a Sattelite.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Dataset object
* `name (String)`: Name of the dataset
* `entity_id (String)`: Name or Id of the entity which this dataset is associated to
* `insert_time (DateTime)`: Time when this Dataset Object was inserted


## Signal

A **Signal** object contains all the required details to be able to load the observations
from a timeseries signal, as well as some metadata about it, such as the minimum and maximum
timestamps that want to be used or the user that registered it.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Signal object
* `name (String)`: Name of the signal
* `dataset_id (ObjectID - Foreign Key)`: Unique Identifier of the Dataset which this signals belongs to
* `start_time (Integer)`: minimum timestamp of this signal
* `stop_time (Integer)`: maximum timestamp of this signal
* `data_location (String)`: URI of the dataset
* `timestamp_column (Integer)`: index of the timestamp column
* `value_column (Integer)`: index of the value column
* `created_by (String)`: Identifier of the user that created this Signal Object
* `insert_time (DateTime)`: Time when this Signal Object was inserted

## Pipeline

A **Pipeline** object contains all the details of the MLPipelines registered in the system,
such as the list of primitives and all the configured hyperparameter values.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Pipeline object
* `name (String)`: Name given to this pipeline
* `mlpipeline (SubDocument)`: JSON representation of this pipeline
* `created_by (String)`: Identifier of the user that created this Pipeline Object
* `insert_time (DateTime)`: Time when this Pipeline Object was inserted

## Experiment

An **Experiment** represents an execution of a Pipeline over the Signals of a Dataset.
Within an experiment, one datarun is executed for each signal in the dataset.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Experiment object
* `project (String)`: Name given to describe the project to which the experiment belongs
* `pipeline_id (ObjectID - Foreign Key)`: Unique Identifier of the Pipeline used
* `dataset_id (ObjectID - Foreign Key)`: Unique Identifier of the Dataset used
* `created_by (String)`: Identifier of the user that created this Experiment Object
* `insert_time (DateTime)`: Time when this Experiment Object was inserted

## Datarun

The **Datarun** objects represent single executions of a **Pipeline** on a **Signal**,
and contain all the information about the environment and context where this execution
took place, which potentially allows to later on reproduce the results in a new environment.

It also contains information about whether the execution was successful or not, when it started
and ended, and the number of events that were found by the pipeline.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Datarun object
* `experiment_id (ObjectID - Foreign Key)`: Unique Identifier of the Experiment
* `signal_id (ObjectID - Foreign Key)`: Unique Identifier of the Signal
* `start_time (DateTime)`: When the execution started
* `end_time (DateTime)`: When the execution ended
* `software_versions (List of Strings)`: version of each python dependency installed in the
virtualenv when the execution took place
* `budget_type (String)`: Type of budget used (time or number of iterations)
* `budget_amount (Integer)`: Budget amount
* `model_location (String)`: URI of the fitted model
* `metrics_location (String)`: URI of the metrics
* `events (Integer)`: Number of events detected during this Datarun execution
* `status (String)`: Whether the Datarun is still running, finished successfully or failed
* `insert_time (DateTime)`: Time when this Datarun Object was inserted

## Event

Each one of the anomalies detected by the pipelines is stored as an **Event**, which
contains the details about the start time, the stop time and the severity score.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Event object
* `datarun_id (ObjectID - Foreign Key)`: Unique Identifier of the Datarun during which this
Event was detected.
* `start_time (Integer)`: Timestamp where the anomalous interval starts.
* `stop_time (Integer)`: Timestamp where the anomalous interval ends.
* `score (Float)`: Severity score given by the pipeline to this Event
* `tag (String)`: User given tag for this Event
* `insert_time (DateTime)`: Time when this Event Object was inserted

## Comment

Each Event can have multiple **Comments**, from one or more users.
**Comments** are expected to be inserted by the domain experts after the Datarun has
finished and they analyze the results.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Comment object
* `event_id (ObjectID - Foreign Key)`: Unique Identifier of the Event to which this Comment relates
* `text (String)`: Comment contents
* `created_by (String)`: Identifier of the user that created this Event Object
* `insert_time (DateTime)`: Time when this Event Object was inserted


# Database Usage

In order to make **Orion** interact with the database you will need to use the `OrionExplorer`,
which provides all the required functionality to register and explore all the database objects,
as well as load pipelines and datasets from it in order to start new dataruns and detect events.

In the following steps we will go over a typical session using the `OrionExplorer` to:
* register a new dataset
* register a new pipeline
* create a datarun using the pipeline to detect events on the dataset
* explore the detected events
* add some comments to the detected events

Note that, because of the dynamic schema-less nature of MongoDB, no database initialization
or table creation is needed. All you need to do start using a new database is create the
`OrionExplorer` instance with the right connection details and start using it!

## 1. Connecting to the Database

In order to connect to the database, all you need to do is import and create an instance of the
`OrionExplorer`.

```
In [1]: from orion.explorer import OrionExplorer

In [2]: orex = OrionExplorer()
```

This will directly create a connection to the database named `'orion'` at the default
MongoDB host, `localhost`, and port, `27017`.

In case you want to connect to a different database, host or port, or in case user authentication
is enabled in your MongoDB instance, you can pass any required additional arguments:

* `database`
* `host`
* `port`
* `username`
* `password`
* `authentication_source`

```
In [3]: orex = OrionExplorer(
   ...:     database='orion_database',
   ...:     host='1.2.3.4',
   ...:     port=1234,
   ...:     username='orion',
   ...:     password='secret_password',
   ...:     authentication_source='admin'
   ...: )
```

## 2. Registering a new Dataset

The first thing that you will need to do to start using **Orion** with a Database will be
**registering a new dataset**.

For this you will need to call the `OrionExplorer.add_dataset` method passing:

* `name - str`: Name that you want to give to this dataset. It must be unique.
* `signal_set - str`: Identifier of the signal(s).
* `satellite_id - str`: (Optional) Identifier of the satellite.
* `start_time - int`: (Optional) minimum timestamp to be used for this dataset. If not given, it
  defaults to the minimum timestamp found in the data.
* `stop_time - int`: (Optional) maximum timestamp to be used for this dataset. If not given, it
  defaults to the maximum timestamp found in the data.
* `location - int`: (Optional) path to the CSV. Skip if using a demo signal hosted on S3.
* `timestamp_column - int`: (Optional) index of the timestamp column. Defaults to 0.
* `value_column - int`: (Optional) index of the value column. Defaults to 1.
* `user - str`: (Optional) Identifier of the user that is creating this Dataset.

In the simplest scenario, you can register a dataset that uses one of the demo signals with the
default start and stop times.

For example, to register a dataset for the signal **S-1** used previously you will execute:

```
In [4]: orex.add_dataset('S-1', 'S-1')
```

In a more complex scenario, we might be loading a CSV file that has the time index in the
third column and the value in the fifth one, and we might want to restrict the timestamps
to a certain range and provide the ID of the user that is registering it.

In this case we will execute:

```
In [5]: orex.add_dataset(
   ...:     name='a_dataset',
   ...:     signal_set='a_signal_name',
   ...:     satellite_id='a_satellite_id',
   ...:     start_time=1483228800,
   ...:     stop_time=1514764800,
   ...:     location='/path/to/the.csv',
   ...:     timestamp_column=2,
   ...:     value_column=4,
   ...:     user_id='1234',
   ...: )
```

**NOTE**: The dataset name must be unique, which means that `add_dataset` method will fail
if a second dataset is added using the same name.

## 3. Exploring the registered datasets

In order to obtain the list of already registered datasets, you can use the
`OrionExplorer.get_datasets` method.

This method returns a `pandas.DataFrame` containing all the details about the registered
datasets:

```
In [6]: datasets = orex.get_datasets()

In [7]: datasets[['name', 'signal_set', 'start_time', 'stop_time']]
Out[7]:
        name     signal_set  start_time   stop_time
0        S-1            S-1  1222819200  1442016000
1  a_dataset  a_signal_name  1483228800  1514764800
```

Optionally, you can restrict the results to a particular signal or satellite by passing them
as arguments to the `get_datasets` method:

```
In [8]: datasets = orex.get_datasets(signal='S-1')

In [9]: datasets[['name', 'signal_set', 'start_time', 'stop_time']]
Out[9]:
  name signal_set  start_time   stop_time
0  S-1        S-1  1222819200  1442016000
```

## 4. Registering a new Pipeline

Another thing that you will need to do before being able to process the created dataset will be
**registering a new pipeline**.

For this you will need to call the `OrionExplorer.add_pipeline` method passing:

* `name - str`: Name that you want to give to this pipeline.
* `path - str`: Path to the JSON file that contains the pipeline specification.
* `user - str`: (Optional) Identifier of the user that is creating this Pipeline.

For example, to register the LSTM pipeline that we used previously under the `'LSTM'` name
you will execute:

```
In [10]: orex.add_pipeline('LSTM', 'orion/pipelines/lstm_dynamic_threshold.json')
```

**NOTE**: In this case the name of the pipeline does not need to be unique. If a pipeline is
added more than once using the same name, they will be stored independently and considered
different versions of the same pipeline.


## 5. Exploring the registered pipelines

Just like datasets, you can obtain the list of registered pipelines using the
`OrionExplorer.get_pipelines` method.

This method returns a `pandas.DataFrame` containing all the details about the registered
pipelines:

```
In [11]: pipelines = orex.get_pipelines()

In [12]: pipelines[['pipeline_id', 'name', 'insert_time']]
Out[12]:
                pipeline_id  name             insert_time
0  5c92797a6c1cea7674cf5b48  LSTM 2019-03-20 17:33:46.452
```

## 6. Running a pipeline on a dataset

Once we have at least one dataset and one pipeline registered, you can start analyzing the data
in search for anomalies.

In order to do so, all you need to do is call the `OrionExplorer.analyze` method passing
the name of pipeline and the name of the dataset:

```
In [13]: datarun = orex.analyze('S-1', 'LSTM')
Using TensorFlow backend.
Epoch 1/1
9899/9899 [==============================] - 55s 6ms/step - loss: 0.0561 - mean_squared_error: 0.0561
```

Once the process has finished, a new Datarun object will have been created in the Database
and returned.

```
In [14]: datarun.id
Out[14]: ObjectId('5c927a846c1cea7674cf5b49')

In [15]: dataruns = orex.get_dataruns()

In [16]: dataruns[['datarun_id', 'start_time', 'end_time', 'events']]
Out[16]:
                 datarun_id               start_time                end_time  events
0  5c927a846c1cea7674cf5b49  2019-03-20 17:38:12.133 2019-03-20 17:39:36.279       2
```

## 7. Explore the found Events

A part from visualizing the number of Events found during the pipeline execution, we will want
to see the exact details of each event.

As you might already be guessing, this can be obtained by calling the `OrionExplorer.get_events`
method and passing it the Datarun object returned by the `analyze` method:

```
In [17]: events = orex.get_events(datarun)

In [18]: events[['event_id', 'score', 'start_time', 'stop_time', 'comments']]
Out[18]:
                   event_id     score  start_time   stop_time  comments
0  5c927ad86c1cea7674cf5b4a  0.047956  1398340800  1398621600         0
1  5c927ad86c1cea7674cf5b4b  0.120997  1398686400  1399420800         0
```

Alternatively, the Datarun ID can be passed directly:

```
In [19]: events = orex.get_events('5c927a846c1cea7674cf5b49')

In [20]: events[['event_id', 'score', 'start_time', 'stop_time', 'comments']]
Out[20]:
                   event_id     score  start_time   stop_time  comments
0  5c927ad86c1cea7674cf5b4a  0.047956  1398340800  1398621600         0
1  5c927ad86c1cea7674cf5b4b  0.120997  1398686400  1399420800         0
```

## 8. Add comments to the Events

While visualizing the detected Events, you might want to add some comments about them.

This is done using the `OrionExplorer.add_comment` method, which will expect:
* The Event object or its ID
* The comment text
* The user ID

```
In [21]: orex.add_comment('5c927ad86c1cea7674cf5b4a', 'This needs to be further investigated', '1234')

In [22]: orex.add_comment('5c927ad86c1cea7674cf5b4b', 'This is probably a false positive', '1234')
```

## 9. Retrieving the Event comments

After adding some comments, these can be recovered using the `OrionExplorer.get_comments`.

This method accepts as optional arguments the Datarun or the Event ID in order to filter
the results:

```
In [23]: comments = orex.get_comments('5c927a846c1cea7674cf5b49')

In [24]: comments[['event_id', 'created_by', 'insert_time', 'text']]
Out[24]:
                   event_id created_by             insert_time                                   text
0  5c927ad86c1cea7674cf5b4a       1234 2019-03-21 13:06:33.591  This needs to be further investigated
1  5c927ad86c1cea7674cf5b4b       1234 2019-03-21 13:07:08.935      This is probably a false positive
```
