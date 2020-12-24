<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<p align="left">
<img width=20% src="https://dai.lids.mit.edu/wp-content/uploads/2018/08/orion.png" alt=“Orion” />
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPi Shield](https://img.shields.io/pypi/v/orion-ml.svg)](https://pypi.python.org/pypi/orion-ml)
[![CircleCI](https://circleci.com/gh/signals-dev/Orion.svg?style=shield)](https://circleci.com/gh/signals-dev/Orion)
[![Travis CI Shield](https://travis-ci.org/signals-dev/Orion.svg?branch=master)](https://travis-ci.org/signals-dev/Orion)
[![Downloads](https://pepy.tech/badge/orion-ml)](https://pepy.tech/project/orion-ml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/signals-dev/Orion/master?filepath=notebooks)

# Orion

* License: [MIT](https://github.com/signals-dev/Orion/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
* Homepage: https://github.com/signals-dev/Orion
* Documentation: https://signals-dev.github.io/Orion

# Overview

Orion is a machine learning library built for *unsupervised time series anomaly detection*. With a given time series data, we provide a number of “verified” ML pipelines (a.k.a Orion pipelines) that identify rare patterns and flag them for expert review.

The library makes use of a number of **automated machine learning** tools developed under [Data to AI Lab at MIT](https://dai.lids.mit.edu/).

**Recent news:** Read about using an Orion pipeline on NYC taxi dataset in a blog series:

<table>
  <tr>
    <td><img src="docs/images/tulog-part-1.png" width=270></td>
    <td><img src="docs/images/tulog-part-2.png" width=270></td>
    <td><img src="docs/images/tulog-part-3.png" width=270></td>
  </tr>
  <tr>
    <td><link rel="part 1" href="https://t.co/yIFVM1oRwQ?amp=1"></td>
    <td><link rel="part 2" href="https://link.medium.com/cGsBD0Fevbb"></td>
    <td><link rel="part 3" href="https://link.medium.com/FqCrFXMevbb"></td>
  </tr>
</table>

**Notebooks:** Discover *Orion* through colab by launching our [notebooks](https://drive.google.com/drive/folders/1FAcCEiE1JDsqaMjGcmiw5a5XuGh13c9Q?usp=sharing)!

# Quickstart

## Install with pip

The easiest and recommended way to install **Orion** is using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install orion-ml
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).


In the following example we show how to use one of the **Orion Pipelines**.

## Fit an Orion pipeline

We will load a demo data for this example: 

```python3
from orion.data import load_signal

train_data = load_signal('S-1-train')
train_data.head()
```

which should show a signal with `timestamp` and `value`.
```
    timestamp     value
0  1222819200 -0.366359
1  1222840800 -0.394108
2  1222862400  0.403625
3  1222884000 -0.362759
4  1222905600 -0.370746
```

In this example we use `lstm_dynamic_threshold` pipeline and set some hyperparameters (in this case training epochs as 5).

```python3
from orion import Orion

hyperparameters = {
    'keras.Sequential.LSTMTimeSeriesRegressor#1': {
        'epochs': 5,
        'verbose': True
    }
}

orion = Orion(
    pipeline='lstm_dynamic_threshold',
    hyperparameters=hyperparameters
)

orion.fit(train_data)
```

## Detect anomalies using the fitted pipeline 
Once it is fitted, we are ready to use it to detect anomalies in our incoming time series:

```python3
new_data = load_signal('S-1-new')
anomalies = orion.detect(new_data)
```
> :warning: Depending on your system and the exact versions that you might have installed some *WARNINGS* may be printed. These can be safely ignored as they do not interfere with the proper behavior of the pipeline.

The output of the previous command will be a ``pandas.DataFrame`` containing a table of detected anomalies:

```
        start         end     score
0  1394323200  1399701600  0.673494
```

# Resources

Additional resources that might be of interest:
* Learn about [benchmarking pipelines](BENCHMARK.md). 
* Read about [pipeline evaluation](orion/evaluation/README.md).
* More about [database design](DATABASE.md).
