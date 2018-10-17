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

Neural Ridge Classifier
--------------------------------

Also known as classical Extreme Learning Machine. Features from training instances are chosen randomly.

.. autoclass:: pyridge.neural.NeuralRidge
	:members:

Boosting Ridge Neural Ridge Classifier
------------------------------------------

AdaBoost meta-algorithm with Neural Ridge as learner base.

.. autoclass:: pyridge.neural.BoostingRidgeNRidge
	:members:


Bagging Neural Ridge Classifier
------------------------------------------

AdaBoost meta-algorithm with Neural Ridge as learner base.

.. autoclass:: pyridge.neural.BaggingNRidge
	:members:

Diverse Neural Ridge Classifier
------------------------------------------

AdaBoost meta-algorithm with Neural Ridge as learner base.

.. autoclass:: pyridge.neural.DiverseNRidge
	:members:


AdaBoost Neural Ridge Classifier
------------------------------------------

AdaBoost meta-algorithm with Neural Ridge as learner base.

.. autoclass:: pyridge.neural.AdaBoostNRidge
	:members:


AdaBoost Negative Correlation Neural Ridge Classifier
-------------------------------------------------------

AdaBoost meta-algorithm with Neural Ridge as learner base.

.. autoclass:: pyridge.neural.AdaBoostNCNRidge
	:members:
