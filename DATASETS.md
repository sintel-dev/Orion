# Datasets

Orion takes a time series signal and produces an interval of expected anomalies. The input to the framework is a univariate time series and the output is a table denoting the start and end timestamp of the anomalies.

## Data Format

### Input 

Orion Pipelines work on time Series that are provided as a single table of telemetry
observations with two columns:

* ``timestamp``: an INTEGER or FLOAT column with the time of the observation in Unix Time Format
* ``value``: an INTEGER or FLOAT column with the observed value at the indicated timestamp

This is an example of such table:

|  timestamp |     value |
|------------|-----------|
| 1222819200 | -0.366358 |
| 1222840800 | -0.394107 |
| 1222862400 |  0.403624 |
| 1222884000 | -0.362759 |
| 1222905600 | -0.370746 |

### Output

The output of the Orion Pipelines is another table that contains the detected anomalous
intervals and that has at least two columns:

* ``start``: timestamp where the anomalous interval starts
* ``end``: timestamp where the anomalous interval ends

Optionally, a third column called ``severity`` can be included with a value that represents the
severity of the detected anomaly.

An example of such a table is:

|      start |        end | severity |
|------------|------------|----------|
| 1222970400 | 1222992000 | 0.572643 |
| 1223013600 | 1223035200 | 0.572643 |


## Dataset we use in this library

For development, evaluation of pipelines, we include a dataset which includes several signals already formatted as expected by the Orion Pipelines.

These formatted datasets include [NASA](https://github.com/khundman/telemanom) and [NAB](https://github.com/numenta/NAB/tree/master/data) which can be browsed and downloaded directly from the [d3-ai-orion AWS S3 Bucket](https://d3-ai-orion.s3.amazonaws.com/index.html). You can use `load_signal` and pass the name of the signal to directly load the signal into Orion. 

```python3
from orion.data import load_signal

signal_name = 'nyc_taxi'
data = load_signal(signal_name)
```


We thank [NASA](https://github.com/khundman/telemanom) and [NAB](https://github.com/numenta/NAB/tree/master/data) for making this data available for public use. Although we use [Yahoo](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70) dataset for our benchmark evaluation, users must request access and therefore cannot provide a direct link to the dataset.