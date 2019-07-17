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

Kernel ELM
--------------------------------

Kernel version of Extreme Learning Machine.

.. autoclass:: pyridge.kernel.KernelELM
	:members:

Ensembles
**********

AdaBoost ELM
------------------------------------------

AdaBoost meta-algorithm with ELM as base learner.

.. autoclass:: pyridge.neural.AdaBoostELM
	:members:

Bagging ELM
------------------

Bagging implementation with ELM as base learner.

.. autoclass:: pyridge.neural.BaggingELM

Boosting Ridge ELM
--------------------

Boosting Ridge with ELM as base learner.

.. autoclass:: pyridge.neural.BoostingRidgeELM

AdaBoost Negative Correlation ELM
------------------------------------

Xin Yao et al. implementation of Negative Correlation and Adaboost.

.. autoclass:: pyridge.neural.AdaBoostNCELM

Diverse ELM
------------

Ensemble proposed by `Perales et al (2018)`_.

.. autoclass:: pyridge.neural.DiverseELM

.. _Perales et al (2018): http://www.doi.org/10.1007/978-3-319-92639-1_25


Regularized Ensemble ELM (REELM)
----------------------------------

Ensemble proposed by `Perales et al (2019)`_.

.. autoclass:: pyridge.neural.DiverseELM

.. _Perales et al (2019): http://www.doi.org/10.1016/j.neucom.2019.06.040
