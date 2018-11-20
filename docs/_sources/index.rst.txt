.. PyRidge documentation master file, created by
   sphinx-quickstart on Wed Jan 10 11:31:17 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyRidge's documentation!
====================================

.. image:: https://travis-ci.org/cperales/pyridge.svg?branch=master
    :target: https://travis-ci.org/cperales/pyridge

.. image:: https://coveralls.io/repos/github/cperales/pyridge/badge.svg
    :target: https://coveralls.io/github/cperales/pyridge


This project is aimed to write an useful machine learning library in Python based on
the Ridge classification algorithms. Public repository is `available in Github
<https://github.com/cperales/PyRidge>`_. These algorithms can be known in the literature
also as Extreme Learning Machine, and they are explained as a type of a feedforward neural
network where some neurons does not require to be tuned by calculating them. Algorithms are the same
and they achieve to a reasonably good solution.

These supervised machine learning algorithms can be known in the literature
as Ridge Classification,
`Tikhonov regularization <https://en.wikipedia.org/wiki/Tikhonov_regularization>`_ or
`Extreme Learning Machine <https://en.wikipedia.org/wiki/Extreme_learning_machine>`_.
A nice discussion about first and second terms can be seen in `this discussion in StackExchange
<https://stats.stackexchange.com/questions/234280/is-tikhonov-regularization-the-same-as-ridge-regression>`_.

Although ELM is a polemic topic
due to the accusations of plagiarism (`see more here <https://github.com/scikit-learn/scikit-learn/pull/10602>`_
and `here <https://www.reddit.com/r/MachineLearning/comments/34y2nk/the_elm_scandal_a_formal_complaint_launched/>`_),
some actual research is done by applying ensemble techniques to Ridge Classification,
thus some some papers are used for implementing algorithms.

Main motivation of this repository is translating from MATLAB to Python 3 what
`I am <https://www.linkedin.com/in/carlos-perales-cperales/>`_ doing in my PhD in Data Science in
`Universidad Loyola Andalucía <https://www.uloyola.es/en/reseach/departments/quantitative-methods-department>`_.


The library is organized in the following way:

.. toctree::
   :maxdepth: 2

   generic
   algorithms
   functions
