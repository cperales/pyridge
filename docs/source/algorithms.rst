Predictors
===========


Single predictors
******************

Extreme Learning Machine
--------------------------------

Also known as single-hidden-layer version of Extreme Learning Machine.
Weights for neurons in the hidden layer are chosen randomly.

.. autoclass:: pyridge.neural.ELM
	:members:


PCA ELM
-----------

.. autoclass:: pyridge.neural.PCAELM
	::members:


PCA LDA ELM
------------

.. autoclass:: pyridge.neural.lda.PCALDAELM
	::members:


Sobol ELM
------------

.. autoclass:: pyridge.neural.sobol.SobolELM
	::members:


Parallel Layer ELM
----------------------

.. autoclass:: pyridge.neural.pl.ParallelLayerELM
	::members:


Kernel ELM
--------------------------------

Kernel version of Extreme Learning Machine.

.. autoclass:: pyridge.kernel.KernelELM
	:members:


Artificial Neural Network
--------------------------------

.. autoclass:: pyridge.neural.nn.NeuralNetwork
	:members:


Neural Network Ensembles
******************************

AdaBoost ELM
------------------------------------------

AdaBoost meta-algorithm with ELM as base learner.

.. autoclass:: pyridge.neural.AdaBoostELM
	:members:

Bagging ELM
------------------

Bagging implementation with ELM as base learner.

.. autoclass:: pyridge.neural.BaggingELM
	:members:

Boosting Ridge ELM
--------------------

Boosting Ridge with ELM as base learner.

.. autoclass:: pyridge.neural.BoostingRidgeELM
	:members:

AdaBoost Negative Correlation ELM
------------------------------------

Xin Yao et al. implementation of Negative Correlation and Adaboost.

.. autoclass:: pyridge.neural.AdaBoostNCELM
	:members:

Diverse ELM
------------

Ensemble proposed by `Perales et al (2018)`_.

.. autoclass:: pyridge.neural.DiverseELM
	:members:

.. _Perales et al (2018): http://www.doi.org/10.1007/978-3-319-92639-1_25


Regularized Ensemble ELM (REELM)
----------------------------------

Ensemble proposed by `Perales et al (2019)`_.

.. autoclass:: pyridge.neural.RegularizedEnsembleELM
	:members:


.. _Perales et al (2019): http://www.doi.org/10.1016/j.neucom.2019.06.040


Negative Correlation Ensembles
******************************

Negative Correlation ELM
----------------------------------

Ensemble proposed by `Perales et al (2020)`_.

.. autoclass:: pyridge.negcor.nc_elm.NegativeCorrelationELM
	:members:


.. _Perales et al (2020): http://www.doi.org/10.1007/s00521-020-04788-9


Negative Correlation Neural Network
------------------------------------


.. autoclass:: pyridge.negcor.nc_nn.NegativeCorrelationNN
	:members:

