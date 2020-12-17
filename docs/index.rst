.. raw:: html

   <p align="left">
   <img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI-Lab" />
   <i>An open source project from Data to AI Lab at MIT.</i>
   </p>

   <p align="left">
   <img width=20% src="https://dai.lids.mit.edu/wp-content/uploads/2018/08/orion.png" alt=“Orion” />
   </p>

|Development Status| |PyPi Shield| |Circle CI| |Travis CI Shield|
|Downloads| |Binder|

Orion
=====

**Date**: |today| **Version**: |version|

-  License: `MIT <https://github.com/signals-dev/Orion/blob/master/LICENSE>`__
-  Development Status:
   `Pre-Alpha <https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha>`__
-  Documentation: https://github.com/signals-dev/Orion
-  Homepage: https://signals-dev.github.io/Orion

Overview
--------

Orion is a machine learning library built for *unsupervised time series anomaly detection*. With a given time series data, we provide a number of “verified” ML pipelines (a.k.a Orion pipelines) that identify rare patterns and flag them for expert review.

The library makes use of a number of **automated machine learning** tools developed under `Data to AI Lab at MIT`_.

**Recent news:** Read about using an Orion pipeline on NYC taxi dataset in a blog series `part 1`_, `part 2`_ and `part 3`_.

Explore Orion
-----------

* `Getting Started <getting_started/index.html>`_
* `User Guides <user_guides/index.html>`_
* `API Reference <api_reference/index.html>`_
* `Developer Guides <developer_guides/index.html>`_
* `Release Notes <history.html>`_

--------------

.. |Development Status| image:: https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow
   :target: https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha
.. |PyPi Shield| image:: https://img.shields.io/pypi/v/orion-ml.svg
   :target: https://pypi.python.org/pypi/orion-ml
.. |Circle CI| image:: https://circleci.com/gh/signals-dev/Orion.svg?style=shield
   :target: https://circleci.com/gh/signals-dev/Orion
.. |Travis CI Shield| image:: https://travis-ci.org/signals-dev/Orion.svg?branch=master
   :target: https://travis-ci.org/signals-dev/Orion
.. |Downloads| image:: https://pepy.tech/badge/orion-ml
   :target: https://pepy.tech/project/orion-ml
.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/signals-dev/Orion/master?filepath=notebooks


.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    getting_started/index
    user_guides/index
    api_reference/index
    developer_guides/index
    Release Notes <history>

.. _Data to AI Lab at MIT: https://dai.lids.mit.edu/
.. _part 1: https://t.co/yIFVM1oRwQ?amp=1
.. _part 2: https://link.medium.com/cGsBD0Fevbb
.. _part 3: https://link.medium.com/FqCrFXMevbb