Algorithms
===========

Ridge with kernel trick
*************************


Kernel Ridge Classifier
--------------------------------

.. autoclass:: pyridge.kernel.KernelRidge
	:members:


Ridge with neural functions
*****************************

ELM Classifier
--------------------------------

Also known as classical Extreme Learning Machine. Features from training instances are chosen randomly.

.. autoclass:: pyridge.neural.ELM
	:members:

Boosting Ridge ELM Classifier
------------------------------------------

AdaBoost meta-algorithm with ELM as learner base.

.. autoclass:: pyridge.neural.BoostingRidgeELM
	:members:


Bagging ELM Classifier
------------------------------------------

AdaBoost meta-algorithm with ELM as learner base.

.. autoclass:: pyridge.neural.BaggingELM
	:members:

Diverse ELM Classifier
------------------------------------------

AdaBoost meta-algorithm with ELM as learner base.

.. autoclass:: pyridge.neural.DiverseELM
	:members:


AdaBoost ELM Classifier
------------------------------------------

AdaBoost meta-algorithm with ELM as learner base.

.. autoclass:: pyridge.neural.AdaBoostELM
	:members:


AdaBoost Negative Correlation ELM Classifier
-------------------------------------------------------

AdaBoost meta-algorithm with ELM as learner base.

.. autoclass:: pyridge.neural.AdaBoostNCELM
	:members:
