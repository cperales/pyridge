Functions for experiments and utilities
================================================

Utilities
---------------

Preparing the datasets
........................
.. autofunction:: pyridge.util.preprocess.prepare_data

Activation and kernel functions
...................................
.. autofunction:: pyridge.util.activation.sigmoid
.. autofunction:: pyridge.util.activation.sigmoid_der
.. autofunction:: pyridge.util.activation.linear_kernel
.. autofunction:: pyridge.util.activation.rbf_kernel
.. autofunction:: pyridge.util.activation.u_dot_norm

Cross validation
..................
.. autofunction:: pyridge.util.cross.cross_validation
.. autofunction:: pyridge.util.cross.train_predictor

Metrics
............
.. autofunction:: pyridge.util.metric.accuracy
.. autofunction:: pyridge.util.metric.rmse
.. autofunction:: pyridge.util.metric.diversity

Experiments
------------

In order to perform several experiments and tests de predictors,
generic test function is used for different algorithms and cross-validation
hyperparameters.

.. autofunction::  pyridge.experiment.check.check_fold
.. autofunction::  pyridge.experiment.check.check_algorithm