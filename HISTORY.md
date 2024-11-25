History
=======

## 0.6.1 - 2024-10-04

### Issues resolved
* Update minimum test images – [Issue #563](https://github.com/signals-dev/Orion/issues/563) by @sarahmish
* Update benchmark – [Issue #538](https://github.com/signals-dev/Orion/issues/538) by @sarahmish
* Pin MacOS Version in GH Actions – [Issue #535](https://github.com/signals-dev/Orion/issues/535) by @sarahmish
* Add anomaly transformer method – [Issue #452](https://github.com/signals-dev/Orion/issues/452) by @sarahmish


## 0.6.0 - 2024-2-13

Support for python 3.10 and 3.11

### Issues resolved

* Update ``test_core`` file – [Issue #507](https://github.com/signals-dev/Orion/issues/507) by @sarahmish
* update ARIMA primitive and pipeline – [Issue #503](https://github.com/signals-dev/Orion/issues/503) by @sarahmish
* Update Dependency – [Issue #497](https://github.com/signals-dev/Orion/issues/497) & [Issue #499](https://github.com/signals-dev/Orion/issues/499) by @sarahmish
* Update Python for Dependency Test – [Issue #484](https://github.com/signals-dev/Orion/issues/484) by @sarahmish
* Add python 3.10 and 3.11 & drop 3.6 and 3.7 – [Issue #477](https://github.com/signals-dev/Orion/issues/477) by @sarahmish
* LNN Pipeline – [Issue #475](https://github.com/signals-dev/Orion/issues/475) by @sarahmish


## 0.5.2 - 2023-10-19

Support for python 3.9 and new Matrix Profile pipeline

### Issues resolved

* Send pipeline names in benchmark arguments – [Issue #466](https://github.com/signals-dev/Orion/issues/466) by @sarahmish
* Add continuation parameter for benchmark – [Issue #464](https://github.com/signals-dev/Orion/issues/464) by @sarahmish
* Fix references in documentation – [Issue #453](https://github.com/signals-dev/Orion/issues/453) by @sarahmish
* Update documentation – [Issue #448](https://github.com/signals-dev/Orion/issues/448) by @sarahmish
* Add matrix profiling method – [Issue #446](https://github.com/signals-dev/Orion/issues/446) by @sarahmish
* Support python 3.9 – [Issue #408](https://github.com/signals-dev/Orion/issues/408) by @sarahmish


## 0.5.1 - 2023-08-16

This version introduces a new dataset to the benchmark.

### Issues resolved

* Add UCR dataset to the benchmark – [Issue #443](https://github.com/signals-dev/Orion/issues/443) by @sarahmish
* docker image build failed – [Issue #439](https://github.com/signals-dev/Orion/issues/439) by @sarahmish
* Edit interval settings in ``azure`` pipeline – [Issue #436](https://github.com/signals-dev/Orion/issues/436) by @sarahmish


## 0.5.0 - 2023-05-23

This version uses ``ml-stars`` package instead of ``mlprimitives``.

### Issues resolved

* Migrate to ml-stars – [Issue #418](https://github.com/signals-dev/Orion/issues/418) by @sarahmish
* Updating ``best_cost`` in ``find_anomalies`` primitive – [Issue #403](https://github.com/signals-dev/Orion/issues/403) by @sarahmish
* Retire ``lstm_dynamic_threshold_gpu`` and ``lstm_autoencoder_gpu`` pipeline maintenance – [Issue #373](https://github.com/signals-dev/Orion/issues/373) by @sarahmish
* Typo in xlsxwriter dependency specification – [Issue #394](https://github.com/signals-dev/Orion/issues/394) by @sarahmish
* ``orion.evaluate`` uses fails when fitting – [Issue #384](https://github.com/signals-dev/Orion/issues/384) by @sarahmish
* AER pipeline with visualization option – [Issue #379](https://github.com/signals-dev/Orion/issues/379) by @sarahmish


## 0.4.1 - 2023-01-31

### Issues resolved

* Move VAE from sandbox to verified – [Issue #377](https://github.com/signals-dev/Orion/issues/377) by @sarahmish
* Pin ``opencv`` – [Issue #372](https://github.com/signals-dev/Orion/issues/372) by @sarahmish
* Pin ``scikit-learn`` – [Issue #367](https://github.com/signals-dev/Orion/issues/367) by @sarahmish
* Fix VAE documentation – [Issue #360](https://github.com/signals-dev/Orion/issues/360) by @sarahmish


## 0.4.0 - 2022-11-08

This version introduces several new enhancements:

* Support to python 3.8
* Migrating to Tensorflow 2.0
* New pipeline, namely ``VAE``, a Variational AutoEncoder model.

### Issues resolved

* Add python 3.8 – [Issue #342](https://github.com/signals-dev/Orion/issues/342) by @sarahmish
* VAE (Variational Autoencoders) pipeline implementation – [Issue #349](https://github.com/signals-dev/Orion/issues/349) by @dyuliu
* Add masking option for ``regression_errors`` – [Issue #352](https://github.com/signals-dev/Orion/issues/352) by @dyuliu
* Changes in TadGAN for tensorflow 2.0 – [Issue #161](https://github.com/signals-dev/Orion/issues/161) by @lcwong0928
* Add an automatic dependency checker – [Issue #320](https://github.com/signals-dev/Orion/issues/320) by @sarahmish
* TadGAN ``batch_size`` cannot be changed – [Issue #313](https://github.com/signals-dev/Orion/issues/313) by @sarahmish


## 0.3.2 - 2022-07-04

This version fixes some of the issues in ``aer``, ``ae``, and ``tadgan`` pipelines.

### Issues resolved

* Fix AER model predict error after loading – [Issue #304](https://github.com/signals-dev/Orion/issues/304) by @lcwong0928
* Update AE to work with any `window_size` – [Issue #300](https://github.com/signals-dev/Orion/issues/300) by @sarahmish
* Updated tadgan_viz.json – [Issue #292](https://github.com/signals-dev/Orion/issues/292) by @Hramir


## 0.3.1 - 2022-04-26

This version introduce a new pipeline, namely ``AER``, an AutoEncoder Regressor model.

### Issues resolved
* Add AER Model - [Issue #286](https://github.com/signals-dev/Orion/issues/286) by @lcwong0928


## 0.3.0 - 2022-03-31

This version deprecates the support of ``OrionDBExplorer``, which has been migrated to
[sintel](https://github.com/signals-dev/Orion). As a result, ``Orion`` no longer requires
mongoDB as a dependency.

### Issues resolved
* Update dependency  - [Issue #283](https://github.com/signals-dev/Orion/issues/283) by @sarahmish
* General housekeeping  - [Issue #278](https://github.com/signals-dev/Orion/issues/278) by @sarahmish
* Fix tutorial testing issue - [Issue #276](https://github.com/signals-dev/Orion/issues/276) by @sarahmish
* Migrate OrionExplorer to Sintel - [Issue #275](https://github.com/signals-dev/Orion/issues/275) by @dyuliu
* LSTM viz JSON pipeline added - [Issue #271](https://github.com/signals-dev/Orion/issues/271) by @Hramir


## 0.2.1 - 2022-02-18

This version introduces improvements and more testing.

### Issues resolved
* Adjusting builds for TadGAN - [Issue #261](https://github.com/signals-dev/Orion/issues/261) by @sarahmish
* Testing tutorials, dependencies, and OS - [Issue #251](https://github.com/signals-dev/Orion/issues/251) by @sarahmish


## 0.2.0 - 2021-10-11

This version supports multivariate timeseries as input. In addition to minor improvements
and maintenance.

### Issues resolved
* `setuptools` no longer supports `lib2to3` breaking `mongoengine` - [Issue #252](https://github.com/signals-dev/Orion/issues/252) by @sarahmish
* Supporting multivariate input - [Issue #248](https://github.com/signals-dev/Orion/issues/248) by @sarahmish
* TadGAN pipeline with visualization option - [Issue #240](https://github.com/signals-dev/Orion/issues/240) by @sarahmish
* Support saving absolute path for add_signals and add_signal when using dbExplorer - [Issue #202](https://github.com/signals-dev/Orion/issues/202) by @sarahmish
* dynamic scalability of TadGAN primitive based on `window_size` - [Issue #87](https://github.com/signals-dev/Orion/issues/87) by @sarahmish


## 0.1.7 - 2021-05-04

This version adds new features to the benchmark function where users can now save pipelines, view results as they are being calculated, and allow a single evaluation to be compared multiple times.

### Issues resolved
* Dask issues in benchmark function & improvements - [Issue #225](https://github.com/signals-dev/Orion/issues/225) by @sarahmish
* Numerical overflow when using contextual metrics - [Issue #212](https://github.com/signals-dev/Orion/issues/212) by @kronerte


## 0.1.6 - 2021-03-08

This version introduces two new pipelines: LSTM AE and Dense AE.
In addition to minor improvements, a bit of code refactoring took place to introduce
a new primtive: ``reconstruction_errors``.

### Issues resolved
* Comparison of DTW library performance - [Issue #205](https://github.com/signals-dev/Orion/issues/205) by @sarahmish
* Not able to pickle dump tadgan pipeline - [Issue #200](https://github.com/signals-dev/Orion/issues/200) by @sarahmish
* New pipeline LSTM and Dense autoencoders - [Issue #194](https://github.com/signals-dev/Orion/issues/194) by @sarahmish
* Readme - [Issue #192](https://github.com/signals-dev/Orion/issues/192) by @pvk-developer
* Unable to launch cli - [Issue #186](https://github.com/signals-dev/Orion/issues/186) by @sarahmish
* bullet points not formatted correctly in index.rst - [Issue #178](https://github.com/signals-dev/Orion/issues/178) by @micahjsmith
* Update notebooks - [Issue #176](https://github.com/signals-dev/Orion/issues/176) by @sarahmish
* Inaccuracy in README.md file in orion/evaluation/ - [Issue #157](https://github.com/signals-dev/Orion/issues/157) by @sarahmish
* Dockerfile -- docker does not find orion primitives automatically - [Issue #155](https://github.com/signals-dev/Orion/issues/155) by @sarahmish
* Primitive documentation - [Issue #151](https://github.com/signals-dev/Orion/issues/151) by @sarahmish
* Variable name inconsistency in tadgan - [Issue #150](https://github.com/signals-dev/Orion/issues/150) by @sarahmish
* Sync leaderboard tables between `BENCHMARK.md` and the docs - [Issue #148](https://github.com/signals-dev/Orion/issues/148) by @sarahmish


## 0.1.5 - 2020-12-25

This version includes the new style of documentation and a revamp of the `README.md`. In addition to some minor improvements
in the benchmark code and primitives. This release includes the transfer of `tadgan` pipeline to `verified`.

### Issues resolved
* Link with google colab - [Issue #144](https://github.com/signals-dev/Orion/issues/144) by @sarahmish
* Add `timeseries_anomalies` unittests - [Issue #136](https://github.com/signals-dev/Orion/issues/136) by @sarahmish
* Update `find_sequences` in converting series to arrays - [Issue #135](https://github.com/signals-dev/Orion/issues/135) by @sarahmish
* Definition of error/critic smooth window in score anomalies primitive - [Issue #132](https://github.com/signals-dev/Orion/issues/132) by @sarahmish
* Train-test split in benchmark enhancement - [Issue #130](https://github.com/signals-dev/Orion/issues/130) by @sarahmish


## 0.1.4 - 2020-10-16

Minor enhancements to benchmark

* Load ground truth before try-catch - [Issue #124](https://github.com/signals-dev/Orion/issues/124) by @sarahmish
* Converting timestamp to datetime in Azure primitive - [Issue #123](https://github.com/signals-dev/Orion/issues/123) by @sarahmish
* Benchmark exceptions - [Issue #120](https://github.com/signals-dev/Orion/issues/120) by @sarahmish


## 0.1.3 - 2020-09-29

New benchmark and Azure primitive.

* Implement a benchmarking function new feature - [Issue #94](https://github.com/signals-dev/Orion/issues/94) by @sarahmish
* Add azure anomaly detection as primitive new feature - [Issue #97](https://github.com/signals-dev/Orion/issues/97) by @sarahmish
* Critic and reconstruction error combination - [Issue #99](https://github.com/signals-dev/Orion/issues/99) by @sarahmish
* Fixed threshold for `find_anomalies` - [Issue #101](https://github.com/signals-dev/Orion/issues/101) by @sarahmish
* Add an option to have window size and window step size as percentages of error size - [Issue #102](https://github.com/signals-dev/Orion/issues/102) by @sarahmish
* Organize pipelines into verified and sandbox - [Issue #105](https://github.com/signals-dev/Orion/issues/105) by @sarahmish
* Ground truth parameter name enhancement - [Issue #114](https://github.com/signals-dev/Orion/issues/114) by @sarahmish
* Add benchmark dataset list and parameters to s3 bucket enhancement - [Issue #118](https://github.com/signals-dev/Orion/issues/118) by @sarahmish

## 0.1.2 - 2020-07-03

New Evaluation sub-package and refactor TadGAN.

* Two bugs when saving signalrun if there is no event detected - [Issue #92](https://github.com/signals-dev/Orion/issues/92) by @dyuliu 
* File encoding/decoding issues about `README.md` and `HISTORY.md` - [Issue #88](https://github.com/signals-dev/Orion/issues/88) by @dyuliu
* Fix bottle neck of `score_anomaly` in Cyclegan primitive - [Issue #86](https://github.com/signals-dev/Orion/issues/86) by @dyuliu
* Adjust `epoch` meaning in Cyclegan primitive - [Issue #85](https://github.com/signals-dev/Orion/issues/85) by @sarahmish
* Rename evaluation to benchmark and metrics to evaluation - [Issue #83](https://github.com/signals-dev/Orion/issues/83) by @sarahmish
* Scoring function for intervals of size one - [Issue #76](https://github.com/signals-dev/Orion/issues/76) by @sarahmish

## 0.1.1 - 2020-05-11

New class and function based interfaces.

* Implement the Orion Class - [Issue #79](https://github.com/D3-AI/Orion/issues/79) by @csala
* Implement new functional interface - [Issue #80](https://github.com/D3-AI/Orion/issues/80) by @csala

## 0.1.0 - 2020-04-23

First Orion release to PyPI: https://pypi.org/project/orion-ml/
